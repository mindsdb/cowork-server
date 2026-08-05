"""The ``.env`` -> DB settings boundary (inbound only).

The DB (``UserSettings``) is the source of truth. This module owns the inbound
conversion from a legacy ``.env`` into DB updates: the ANTON_* alias map and
provider-value normalization live here so ``user_settings`` stays purely about
the DB model.

The outbound DB -> ``.env`` export was removed in ENG-1295. The standalone
``anton`` CLI now owns its own ``~/.anton/.env`` (it no longer reads the Cowork
mirror), so the server no longer mirrors settings to ``~/.cowork/.env``. Only
the inbound direction remains, used by the one-time boot migration that seeds
the DB from a legacy ``.env``.

Model keys (planning_model / coding_model) are deliberately absent from the alias
map (ENG-739): a model is CLI-only and must never ride a bulk ``.env`` sync.
"""
from __future__ import annotations

from cowork.common.settings.user_settings import Provider

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
