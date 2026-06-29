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
import shutil
from pathlib import Path

from sqlmodel import Session, select
from pydantic import ValidationError

from anton.core.tools.skill_format import normalize_name, DESC_MAX, SKILL_FILE
from cowork.common.settings import invalidate_user_settings_cache
from cowork.common.settings.user_settings import UserSettings
from cowork.models.setting import Setting
from cowork.models.skill import META_CREATED_AT, META_DISPLAY_NAME, Skill, SkillLegacy
from cowork.services.settings import SettingService
from cowork.harnesses.hermes_harness.settings import HermesHarnessSettings
from cowork.services.skills import SkillService

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



# Packaged skills shipped with cowork. Bump BUILTIN_SKILLS_VERSION when the
# set changes; seeding re-runs only when the stored version is lower, so a
# future release can ship more skills without touching ones the user edited
# or deleted.
BUILTIN_SKILLS_DIR = Path(__file__).parent / "skills_builtin"
BUILTIN_SKILLS_SENTINEL = "_builtin_skills_set"
BUILTIN_SKILLS_VERSION = 1


def seed_builtin_skills(session: Session) -> bool:
    """Copy packaged builtin skills into the canonical skills store.

    Runs only when the stored set version is below ``BUILTIN_SKILLS_VERSION``.
    Existing folders are never overwritten, so user edits/deletes survive.
    Returns True if seeding ran (version advanced), False if skipped.
    """
    svc = SettingService(session)
    row = svc._fetch_row(BUILTIN_SKILLS_SENTINEL)
    current = int(row.value) if row is not None and row.value.isdigit() else 0
    if current >= BUILTIN_SKILLS_VERSION:
        return False

    # remove symlink to global skills if exists
    link = Path(HermesHarnessSettings().root_dir) / "skills"
    if link.is_symlink():
        link.unlink()

    store = SkillService(session)
    copied = 0
    if BUILTIN_SKILLS_DIR.exists():
        store._ensure_root()
        for src in sorted(BUILTIN_SKILLS_DIR.iterdir()):
            if not src.is_dir() or not (src / SKILL_FILE).exists():
                continue
            dest = store._skill_dir(src.name)
            if dest.exists():
                continue  # keep the user-editable copy untouched
            shutil.copytree(src, dest)
            copied += 1

    if copied:
        logger.info("Seeded %d builtin skill(s) into %s", copied, store.root)

    # Attach every skill that has no project yet to the general project
    from cowork.services.projects import GENERAL_PROJECT, ProjectService

    if ProjectService(session).get_project_by_name_or_none(GENERAL_PROJECT) is not None:
        for skill in store.list_skills():
            if not skill.projects:
                store.update_skill(skill.name, projects=[GENERAL_PROJECT])

    # Raw Setting row: the sentinel key isn't a UserSettings field, so it must
    # bypass SettingService validation (same pattern as the other migrations).
    if row is None:
        row = Setting(key=BUILTIN_SKILLS_SENTINEL, value=str(BUILTIN_SKILLS_VERSION))
    else:
        row.value = str(BUILTIN_SKILLS_VERSION)
    session.add(row)
    session.commit()
    return True


def migrate_skills_to_files(session: Session) -> bool:
    """Seed skill files from ``skills_legacy`` if not already done.

    Returns True if the migration ran, False if it was skipped.
    """

    SKILL_MIGRATION_SENTINEL = "_skills_migrated"

    svc = SettingService(session)
    if svc._fetch_row(SKILL_MIGRATION_SENTINEL) is not None:
        return False

    store = SkillService(session)
    rows = list(session.exec(select(SkillLegacy)).all())

    def _unique_slug(svc: SkillService, base: str, taken: set[str]) -> str:
        slug = base
        n = 2
        while slug in taken or svc._skill_dir(slug).exists():
            suffix = f"-{n}"
            slug = base[: 64 - len(suffix)].rstrip("-") + suffix
            n += 1
        return slug

    taken: set[str] = set()
    migrated = 0
    for row in rows:
        base = normalize_name(row.label or row.name or "")
        slug = _unique_slug(store, base, taken)
        taken.add(slug)

        # when_to_use is dropped as a field; fold it into the description.
        description = ". ".join(
            part
            for part in ((row.description or "").strip(), (row.when_to_use or "").strip())
            if part
        )
        description = (row.description or "").strip()
        when_to_use = (row.when_to_use or "").strip()
        if when_to_use:
            description = f"{description}. {when_to_use}" if description else when_to_use
        description = description[:DESC_MAX]

        metadata: dict[str, str] = {"cowork_id": str(row.id)}
        if row.name and row.name != slug:
            metadata[META_DISPLAY_NAME] = row.name
        if row.created_at:
            metadata[META_CREATED_AT] = row.created_at.isoformat()

        try:
            skill = Skill(
                name=slug,
                instructions=row.instructions or "",
                description=description or row.name or slug,
                metadata=metadata,
            )
            store._write(skill)
            migrated += 1
        except (OSError, ValueError):
            logger.warning("Failed to migrate skill %r", slug, exc_info=True)

    if migrated:
        logger.info("Migrated %d skill(s) from skills_legacy to files at %s", migrated, store.root)

    session.add(Setting(key=SKILL_MIGRATION_SENTINEL, value="1"))
    session.commit()
    return True
