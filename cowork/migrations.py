"""One-time .env -> DB settings migration.

On first boot (or after an upgrade from .env-only to DB-backed settings),
this module reads ``~/.cowork/.env`` and seeds any missing settings into the
SQLite database.  A sentinel row (``_env_migrated_v2``) is written to the
``settings`` table so the migration never runs twice.

The v2 sentinel replaced the original ``_env_migrated`` sentinel (which read
from the now-legacy ``~/.anton/.env`` path).  Bumping to v2 lets users who
had the old sentinel fire against the wrong path get a fresh migration run
from the correct ``~/.cowork/.env`` location.

After migration the DB is **authoritative** for all overlapping fields.
The ``.env`` file continues to exist for:
  - The standalone ``anton`` CLI (reads ``AntonSettings`` from ``.env``)
  - Onboarding (writes ``.env`` first, then syncs to DB)
  - Fields that only exist in ``AntonSettings`` (workspace paths, etc.)

Cowork-server runtime code should read from ``get_user_settings()`` (DB),
never from the ``.env`` directly.
"""
from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError
from sqlmodel import Session

from cowork.common.settings import invalidate_user_settings_cache
from cowork.common.settings.user_settings import UserSettings
from cowork.models.setting import Setting
from cowork.services.settings import SettingService

logger = logging.getLogger(__name__)

_ENV_PATH = Path.home() / ".cowork" / ".env"

# v2: path corrected from ~/.anton/.env to ~/.cowork/.env; bumped so users
# with the old sentinel (which may have found nothing) get a fresh run.
_MIGRATION_SENTINEL = "_env_migrated_v2"

# Complete map of .env keys -> DB setting keys for all fields that
# overlap between AntonSettings (.env) and UserSettings (DB).
_ENV_TO_SETTING: dict[str, str] = {
    # API keys
    "ANTON_ANTHROPIC_API_KEY": "anthropic_api_key",
    "ANTON_OPENAI_API_KEY": "openai_api_key",
    "ANTON_MINDS_API_KEY": "minds_api_key",
    # Provider / model
    "ANTON_PLANNING_PROVIDER": "planning_provider",
    "ANTON_PLANNING_MODEL": "planning_model",
    "ANTON_CODING_PROVIDER": "coding_provider",
    "ANTON_CODING_MODEL": "coding_model",
    # URLs
    "ANTON_MINDS_URL": "minds_url",
    "ANTON_OPENAI_BASE_URL": "openai_base_url",
    # Behavioral settings
    "ANTON_MEMORY_ENABLED": "memory_enabled",
    "ANTON_MEMORY_MODE": "memory_mode",
    "ANTON_EPISODIC_MEMORY": "episodic_memory",
    "ANTON_PROACTIVE_DASHBOARDS": "proactive_dashboards",
    "ANTON_ACT_FIRST": "act_first",
    "ANTON_PUBLISH_URL": "publish_url",
}


def _parse_env_file() -> dict[str, str]:
    """Read ``~/.cowork/.env`` into a dict, or return empty if absent."""
    if not _ENV_PATH.exists():
        return {}
    result: dict[str, str] = {}
    try:
        for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip().strip('"').strip("'")
    except Exception:
        logger.warning("Failed to read %s", _ENV_PATH, exc_info=True)
    return result


def _normalize_provider_value(val: str, dotenv: dict[str, str]) -> str:
    """Translate .env provider strings to DB enum values.

    The .env may use hyphens (``openai-compatible``) or underscores
    (``openai_compatible``).  The DB ``Provider`` enum uses underscores
    and has a dedicated ``minds_cloud`` value.
    """
    # Canonicalize to underscores first so both forms are handled.
    canonical = val.replace("-", "_")
    # Detect MindsHub: if the user has a Minds key and the provider is
    # "openai_compatible", this is really minds_cloud.
    if canonical == "openai_compatible" and dotenv.get("ANTON_MINDS_API_KEY"):
        return "minds_cloud"
    return canonical


def sync_env_vars_to_db(session: Session, dotenv: dict[str, str]) -> list[str]:
    """Upsert a dict of ANTON_* env vars into the settings DB.

    Returns the list of DB setting keys that were written.  Skips env
    keys that have no mapping in ``_ENV_TO_SETTING``.
    """
    svc = SettingService(session)
    updates: dict[str, str] = {}
    for env_key, setting_key in _ENV_TO_SETTING.items():
        val = dotenv.get(env_key)
        if not val:
            continue
        if setting_key.endswith("_provider"):
            val = _normalize_provider_value(val, dotenv)
        updates[setting_key] = val

    for key, value in updates.items():
        svc._validate_key(key)
        try:
            UserSettings.model_validate({key: value})
        except ValidationError as e:
            raise ValueError(str(e)) from e

    return svc.bulk_upsert(updates)


def migrate_env_to_db(session: Session) -> bool:
    """Seed DB settings from .env if migration hasn't run yet.

    Returns True if the migration ran, False if it was already done or
    there was nothing to migrate.
    """
    svc = SettingService(session)

    # Check the persistent sentinel — survives server restarts.
    sentinel = svc._fetch_row(_MIGRATION_SENTINEL)
    if sentinel is not None:
        return False

    dotenv = _parse_env_file()

    if dotenv:
        migrated_keys: list[str] = []
        for env_key, setting_key in _ENV_TO_SETTING.items():
            val = dotenv.get(env_key)
            if not val:
                continue
            if setting_key.endswith("_provider"):
                val = _normalize_provider_value(val, dotenv)
            try:
                svc.upsert_setting(setting_key, val)
                migrated_keys.append(setting_key)
            except Exception as e:
                logger.debug("Skipping env migration for %s: %s", env_key, e)

        if migrated_keys:
            logger.info(
                "Migrated %d settings from .env to database: %s",
                len(migrated_keys),
                ", ".join(migrated_keys),
            )

    # Write the sentinel so we never run again, even if we migrated
    # zero keys (empty .env or already-populated DB).
    session.add(Setting(key=_MIGRATION_SENTINEL, value="1"))
    session.commit()
    return True


# Legacy MindsHub host. The default base URL changed from https://mdb.ai to
# https://api.mindshub.ai (cowork "new urls", 2026-05-14) — but only for NEW
# seeds. No migration ever rewrote existing rows, and mdb.ai's /api/v1 path
# now 404s, so any user who configured MindsHub before that flip is stuck
# with a failing provider ("Endpoint not found — check the base URL") and no
# UI field to correct it. This backfill closes that gap.
_LEGACY_MINDS_HOSTS = ("https://mdb.ai", "http://mdb.ai")
_CANONICAL_MINDS_URL = "https://api.mindshub.ai"


def backfill_minds_url(session: Session) -> bool:
    """Rewrite the legacy MindsHub host (mdb.ai) to api.mindshub.ai.

    Touches both ``providers_json`` (the per-provider ``mindsUrl`` the
    Test/ping uses) and the top-level ``minds_url``. Idempotent and
    sentinel-free: it only modifies rows that still contain the legacy
    host, so it is safe to run on every boot and self-heals rows that a
    later stale "Save settings" might re-introduce.

    Returns True if any row was rewritten.
    """
    svc = SettingService(session)
    changed: list[str] = []
    for key in ("providers_json", "minds_url"):
        row = svc._fetch_row(key)
        if row is None or "mdb.ai" not in row.value:
            continue
        new_val = row.value
        for legacy in _LEGACY_MINDS_HOSTS:
            new_val = new_val.replace(legacy, _CANONICAL_MINDS_URL)
        if new_val != row.value:
            row.value = new_val
            session.add(row)
            changed.append(key)
    if changed:
        session.commit()
        invalidate_user_settings_cache()
        logger.info(
            "Backfilled legacy MindsHub host (mdb.ai -> api.mindshub.ai) in: %s",
            ", ".join(changed),
        )
    return bool(changed)
