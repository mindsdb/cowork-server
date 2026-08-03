"""The ``.env`` <-> DB settings boundary, both directions in one place.

The DB (``UserSettings``) is the source of truth; the standalone ``anton`` CLI
still reads its config from ``.env``, so ``.env`` is a trailing dependency the
server keeps in sync. Everything that knows an ``ANTON_*`` variable name lives
here — the alias map, provider-value normalization, and the two conversions —
so ``user_settings`` stays purely about the DB model.

- inbound  (``.env`` -> DB): ``env_to_db_updates`` maps + normalizes; the caller
  (SettingService / migration) validates, encrypts, and writes.
- outbound (DB -> ``.env``): ``db_to_env`` formats the stored values, then
  ``merge_env_lines`` + ``atomic_write_env`` persist them, preserving unmanaged lines.

Model keys (planning_model / coding_model) are deliberately absent from the alias
map (ENG-739): a model is CLI-only and must never ride a bulk ``.env`` sync.
"""
from __future__ import annotations

import errno
import logging
import os
import tempfile
import time
from enum import Enum
from pathlib import Path

from pydantic import SecretStr

from cowork.common.settings.user_settings import (
    Provider,
    UserSettings,
    provider_api_key_str,
)

logger = logging.getLogger(__name__)

# DB setting key -> its ANTON_* .env variable, for every field that overlaps
# between AntonSettings (.env) and UserSettings (DB). Single canonical map (was
# hand-maintained in two places that drifted — ENG-1125).
SETTING_ENV_ALIASES: dict[str, str] = {
    "anthropic_api_key": "ANTON_ANTHROPIC_API_KEY",
    "openai_api_key": "ANTON_OPENAI_API_KEY",
    "openai_compatible_api_key": "ANTON_OPENAI_API_KEY_CUSTOM",
    "gemini_api_key": "ANTON_GEMINI_API_KEY",
    "minds_api_key": "ANTON_MINDS_API_KEY",
    "planning_provider": "ANTON_PLANNING_PROVIDER",
    "coding_provider": "ANTON_CODING_PROVIDER",
    "router_provider": "ANTON_ROUTER_PROVIDER",
    "minds_url": "ANTON_MINDS_URL",
    "openai_base_url": "ANTON_OPENAI_BASE_URL",
    "memory_enabled": "ANTON_MEMORY_ENABLED",
    "memory_mode": "ANTON_MEMORY_MODE",
    "episodic_memory": "ANTON_EPISODIC_MEMORY",
    "proactive_dashboards": "ANTON_PROACTIVE_DASHBOARDS",
    "act_first": "ANTON_ACT_FIRST",
    "publish_url": "ANTON_PUBLISH_URL",
}

# Inverse view (ANTON_* -> DB key) for the inbound (.env-first) callers.
ENV_ALIAS_TO_SETTING: dict[str, str] = {v: k for k, v in SETTING_ENV_ALIASES.items()}

# The per-role model vars the outbound export owns. Absent from SETTING_ENV_ALIASES
# because the INBOUND (.env->DB) direction must never map a model (ENG-739: a
# stale ``latest:`` line must not re-pin the picker). OUTBOUND they ARE managed —
# a provider is exported WITH its resolved model so the CLI can never end up with
# a provider/model mismatch (ENG-1127 review); being managed also means a stale
# model line is dropped when its provider is cleared.
_MODEL_ENV_VARS: tuple[str, ...] = (
    "ANTON_PLANNING_MODEL",
    "ANTON_CODING_MODEL",
    "ANTON_ROUTER_MODEL",
)

# The ANTON_* vars the outbound export owns; every other .env line is preserved.
# Includes the two key vars the export no longer WRITES (ANTON_GEMINI_API_KEY,
# ANTON_OPENAI_API_KEY_CUSTOM — the pinned CLI has no field for them; gemini/oc
# creds ride the OpenAI slot) so a stale line from an older writer is still
# reconciled away.
MANAGED_ENV_VARS: tuple[str, ...] = tuple(SETTING_ENV_ALIASES.values()) + _MODEL_ENV_VARS


def normalize_provider_value(val: str, *, minds_key_present: bool) -> str:
    """A .env / UI provider string -> the DB ``Provider`` enum value.

    Hyphen->underscore canonicalization plus the "a Minds key is present, so
    ``openai-compatible`` really means ``minds_cloud``" heuristic. The inverse
    (DB -> UI/.env) is ``Provider.ui_value``.
    """
    canonical = val.replace("-", "_")
    if canonical == Provider.OPENAI_COMPATIBLE.value and minds_key_present:
        return Provider.MINDS_CLOUD.value
    return canonical


# ── inbound: .env -> DB ───────────────────────────────────────────────

def env_to_db_updates(dotenv: dict[str, str]) -> dict[str, str]:
    """A parsed ``.env`` dict -> ``{db_key: value}`` ready for the DB.

    Maps ANTON_* names, skips absent/empty vars, normalizes provider fields.
    Pure conversion — validation, encryption and the DB write stay in the caller.
    """
    updates: dict[str, str] = {}
    for env_var, setting_key in ENV_ALIAS_TO_SETTING.items():
        val = dotenv.get(env_var)
        if not val:
            continue
        if setting_key.endswith("_provider"):
            val = normalize_provider_value(
                val, minds_key_present=bool(dotenv.get("ANTON_MINDS_API_KEY"))
            )
        updates[setting_key] = val
    return updates


# ── outbound: DB -> .env ──────────────────────────────────────────────

def _env_str(value: object) -> str:
    """A loaded ``UserSettings`` value -> its ``.env`` string.

    Providers use the dash form the CLI expects (``Provider.ui_value``); secrets
    are the decrypted plaintext; booleans are lowercased.
    """
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, Provider):
        return value.ui_value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _is_dotenv_safe(value: str) -> bool:
    """A value is safe to write as ``VAR=value`` only if it stays on one line.

    A CR/LF in the value would terminate the assignment and turn the remainder
    into a *new* line — e.g. ``minds_url = "https://x\\nDATABASE_URI=…"`` injects
    an unmanaged ``DATABASE_URI`` that survives every later merge and is consumed
    on the next CLI/server start. The exported fields (keys, URLs, providers,
    booleans) never legitimately contain a newline, so we reject rather than
    quote — a poisoned value is dropped, not smuggled into the file.
    """
    return "\n" not in value and "\r" not in value


# The non-provider settings still export by a straight present-gated alias:
# booleans and the publish URL have no cross-field resolution. The provider /
# key / base-url / model cluster is rendered separately (see db_to_env) because
# it needs the SAME resolution the CLI and build_llm_client apply.
_OUTBOUND_FLAG_ALIASES: dict[str, str] = {
    "memory_enabled": "ANTON_MEMORY_ENABLED",
    "memory_mode": "ANTON_MEMORY_MODE",
    "episodic_memory": "ANTON_EPISODIC_MEMORY",
    "proactive_dashboards": "ANTON_PROACTIVE_DASHBOARDS",
    "act_first": "ANTON_ACT_FIRST",
    "publish_url": "ANTON_PUBLISH_URL",
}


def _anton_provider_name(p: Provider) -> str:
    """A cowork ``Provider`` -> the provider name Anton understands ON DISK.

    Anton has no first-class gemini provider: its own ``anton setup`` writes
    gemini as ``openai-compatible`` + Google's base URL + the key in the OpenAI
    slot, and ``LLMClient.from_settings`` maps ``minds-cloud`` -> openai-compatible
    itself. So gemini is translated here; everything else is its kebab ui_value.
    """
    return "openai-compatible" if p is Provider.GEMINI else p.ui_value


def _openai_slot_demand(settings: UserSettings, p: Provider) -> tuple[str, str | None] | None:
    """The ``(key, base)`` a provider needs in Anton's shared OpenAI slot.

    ``None`` for providers that don't touch that slot (anthropic uses its own
    key slot; minds-cloud uses the dedicated ``minds_*`` slots and derives the
    OpenAI creds in ``model_post_init``). openai / openai-compatible / gemini all
    ride ``ANTON_OPENAI_API_KEY`` / ``ANTON_OPENAI_BASE_URL``, so their demands
    must AGREE for a config to be representable — see ``_env_representable``.
    """
    if p in (Provider.ANTHROPIC, Provider.MINDS_CLOUD):
        return None
    from cowork.services.providers import provider_base_url  # lazy: avoid import cycle
    return (
        provider_api_key_str(settings, p),
        provider_base_url(p.ui_value, openai_base_url=settings.openai_base_url or ""),
    )


def _env_representable(settings: UserSettings, providers: list[Provider]) -> bool:
    """Whether the resolved per-role ``providers`` fit Anton's on-disk model.

    Anton has ONE global ``ANTON_OPENAI_API_KEY`` / ``ANTON_OPENAI_BASE_URL`` pair,
    handed to BOTH its openai and openai-compatible factories, and its minds
    derivation only fires when that OpenAI key is unset. So a config is
    representable only when every OpenAI-slot role agrees on the same ``(key,
    base)`` AND minds-cloud never coexists with an explicit OpenAI-slot role
    (planning=OpenAI + coding=Gemini, or MindsHub + OpenAI, otherwise silently
    misroute one role's key to the other's endpoint — ENG-1127 review).
    """
    openai_demands = {d for p in providers if (d := _openai_slot_demand(settings, p)) is not None}
    minds_present = Provider.MINDS_CLOUD in providers
    return len(openai_demands) <= 1 and not (minds_present and openai_demands)


def _emit_provider_creds(out: dict[str, str], settings: UserSettings, p: Provider) -> None:
    """Write ``p``'s API key (and base URL) into the ANTON_* slots the CLI reads.

    Only called for a representable set (see ``_env_representable``), so the
    OpenAI slot is never contended; minds-cloud uses the dedicated ``minds_*``
    slots and derives its base from them.
    """
    from cowork.services.providers import provider_base_url  # lazy: avoid import cycle

    key = provider_api_key_str(settings, p)
    if p is Provider.ANTHROPIC:
        if key:
            out["ANTON_ANTHROPIC_API_KEY"] = key
    elif p is Provider.MINDS_CLOUD:
        if key:
            out["ANTON_MINDS_API_KEY"] = key
        if settings.minds_url:
            out["ANTON_MINDS_URL"] = settings.minds_url
    else:  # OPENAI / OPENAI_COMPATIBLE / GEMINI — all via the OpenAI slot
        if key:
            out["ANTON_OPENAI_API_KEY"] = key
        base = provider_base_url(p.ui_value, openai_base_url=settings.openai_base_url or "")
        if base:
            out["ANTON_OPENAI_BASE_URL"] = base


def db_to_env(settings: UserSettings, present_keys: set[str]) -> dict[str, str]:
    """A loaded ``UserSettings`` -> the ``{ANTON_*: value}`` the CLI can RUN.

    Not a raw field dump: the provider / model / key / base cluster is rendered in
    Anton's on-disk vocabulary via the SAME resolution the server's own
    ``build_llm_client`` uses (``resolved_*`` + ``provider_base_url`` +
    ``provider_api_key``). So a DB ``gemini`` provider exports as
    ``openai-compatible`` + Google's base URL + the key in the OpenAI slot — the
    shape the standalone CLI (and ``anton setup``) understands — instead of the
    literal ``provider=gemini`` the pinned CLI rejects (ENG-1127 review). Each
    role's provider is written WITH its resolved model so the pair is always
    valid; a role whose resolved provider has no key exports nothing, so an
    unconfigured or freshly-cleared install leaves the CLI on its own defaults.

    A mixed cross-role config the CLI cannot represent (planning=OpenAI +
    coding=Gemini, MindsHub + OpenAI, …) exports NO provider cluster rather than
    a silently-misrouting one — the product can run those via per-role clients,
    but the standalone CLI has a single OpenAI slot (ENG-1127 review).

    The remaining settings (memory flags, publish URL) have no cross-field
    resolution and export by a straight present-gated alias.
    """
    out: dict[str, str] = {}

    roles = (
        ("ANTON_PLANNING_PROVIDER", "ANTON_PLANNING_MODEL",
         settings.resolved_planning_provider, settings.resolved_planning_model),
        ("ANTON_CODING_PROVIDER", "ANTON_CODING_MODEL",
         settings.resolved_coding_provider, settings.resolved_coding_model),
        ("ANTON_ROUTER_PROVIDER", "ANTON_ROUTER_MODEL",
         settings.resolved_router_provider, settings.resolved_router_model),
    )
    keyed_roles = [
        (prov_var, model_var, prov, model)
        for prov_var, model_var, prov, model in roles
        if provider_api_key_str(settings, prov)  # resolved provider has a key → runnable
    ]
    unique_providers: list[Provider] = []
    for _, _, prov, _ in keyed_roles:
        if prov not in unique_providers:
            unique_providers.append(prov)  # planning-first order preserved

    if _env_representable(settings, unique_providers):
        for prov_var, model_var, prov, model in keyed_roles:
            out[prov_var] = _anton_provider_name(prov)
            if model:
                out[model_var] = model
        for prov in unique_providers:
            _emit_provider_creds(out, settings, prov)
    elif unique_providers:
        logger.warning(
            "settings: skipping .env provider export — roles resolve to providers the "
            "standalone CLI cannot represent together (%s); leaving the CLI on its own config",
            ", ".join(p.value for p in unique_providers),
        )

    for db_key, env_var in _OUTBOUND_FLAG_ALIASES.items():
        if db_key not in present_keys:
            continue
        value = getattr(settings, db_key, None)
        if value is not None:
            text = _env_str(value)
            if text:
                out[env_var] = text

    # One injection guard over everything emitted (keys, URLs, models, flags): a
    # CR/LF-bearing value is a dotenv-injection vector, never a real setting.
    safe: dict[str, str] = {}
    for var, val in out.items():
        if _is_dotenv_safe(val):
            safe[var] = val
        else:
            logger.warning("settings: refusing to export %s — value spans multiple lines", var)
    return safe


def merge_env_lines(existing: str, managed: dict[str, str]) -> str:
    """Rewrite the managed lines in ``existing``, preserving everything else.

    Managed ANTON_* lines are dropped then re-appended in alias order (byte-stable
    across identical states); unmanaged lines (auth token, CLI model pins,
    comments) keep their place. A managed key absent from ``managed`` loses its
    line — that is how a logout wipes credentials from the CLI's file too.

    Newline-bearing managed values are skipped as a serialization invariant so a
    single assignment can never expand into a second (injected) one, even if a
    caller hands in an unsanitized dict.
    """
    drop = tuple(f"{var}=" for var in MANAGED_ENV_VARS)
    kept = [ln for ln in existing.split("\n") if ln and not ln.startswith(drop)]
    kept.extend(
        f"{var}={value}" for var, value in managed.items() if _is_dotenv_safe(value)
    )
    return "\n".join(kept) + "\n"


# Transient Windows share-mode locks (the CLI or a version-skewed server holding
# ``.env`` open, an AV scan, a delete-pending handle still closing) abort the
# rename with one of these — the exact EPERM class that wedged onboarding on the
# CLIENT before it grew a retry (ENG-1209). Now that the server is the writer
# (ENG-1127), the same hardening has to live here. POSIX has no mandatory
# locking, so these are effectively Windows-only.
_TRANSIENT_LOCK_ERRNOS = frozenset({errno.EPERM, errno.EACCES, errno.EBUSY, errno.ENOTEMPTY})
_REPLACE_ATTEMPTS = 6
_REPLACE_BASE_DELAY_S = 0.06

# Orphaned temps from a hard-kill / power-loss between the write and the rename
# hold the full plaintext key, so they must never linger; sweep only STALE ones
# so a concurrent writer's fresh in-flight temp is spared.
_STALE_TMP_S = 5 * 60


def _is_transient_lock_error(exc: OSError) -> bool:
    return exc.errno in _TRANSIENT_LOCK_ERRNOS


def _sweep_stale_temps(directory: Path) -> None:
    """Remove orphaned ``.env.*.tmp`` files older than ``_STALE_TMP_S``.

    Only stale temps go — a live writer's temp is fresh and spared, so this can't
    yank one out from under a concurrent rename.
    """
    try:
        now = time.time()
        for entry in directory.glob(".env.*.tmp"):
            try:
                if now - entry.stat().st_mtime > _STALE_TMP_S:
                    entry.unlink()
            except OSError:
                pass  # gone already or unreadable — best-effort
    except OSError:
        pass  # dir unreadable — nothing to sweep


def _replace_with_retry(tmp: str, dest: str) -> None:
    """``os.replace`` the finished temp onto ``dest``, retrying transient locks.

    The temp is already written to a fresh, unlocked path; only the rename
    contends with a Windows share-mode lock, so that is all we retry — with a
    widening backoff (~60ms..360ms, ~1.3s total) that mirrors the client's
    ``retryOnTransientLock`` (ENG-1209). A non-lock error (ENOENT, ENOTDIR, a
    genuinely unwritable target) rethrows at once.
    """
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, dest)
            return
        except OSError as exc:
            if attempt < _REPLACE_ATTEMPTS - 1 and _is_transient_lock_error(exc):
                time.sleep(_REPLACE_BASE_DELAY_S * (attempt + 1))
                continue
            raise


def atomic_write_env(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically, owner-only (0o600).

    Temp file + ``os.replace`` so a crash or a concurrent CLI read never sees a
    truncated ``.env``; 0o600 because the file holds plaintext API keys. The
    rename is retried on transient Windows share-mode locks (ENG-1209/ENG-1127).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _sweep_stale_temps(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.chmod(tmp, 0o600)
        _replace_with_retry(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
