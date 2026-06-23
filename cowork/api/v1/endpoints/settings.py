"""Canonical settings endpoints.

- GET /          — list all settings with metadata
- PUT /{key}     — upsert a single setting
- DELETE /{key}  — delete a setting
- POST /validate — check if an API key is configured for the active provider
- GET /configured — quick boolean check
- GET /install-status — static "installed" response
- GET /reveal-key/{name} — return the unmasked API key for a provider
- POST /test-providers — ping provider APIs to check connectivity
- POST /validate-provider — validate credentials for a specific provider
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, SecretStr
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.base import CamelRequest
from cowork.schemas.settings import SettingResponse, SettingUpsertRequest
from cowork.services.providers import (
    check_config_status,
    fetch_minds_models,
    ping_providers,
    resolve_stored_key,
    validate_provider as validate_provider_svc,
)
from cowork.services.settings import SettingService
from cowork.common.settings.app_settings import (
    DIRECT_EFFORT_CATALOG,
    RECOMMENDED_MODELS,
    RECOMMENDED_PAIR,
)

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]


# ── CRUD ─────────────────────────────────────────────────────────────


@router.get("/", response_model=list[SettingResponse])
def list_settings(session: SessionDep) -> list[SettingResponse]:
    return SettingService(session).list_settings()


@router.put("/{key}", response_model=SettingResponse)
def upsert_setting(
    key: str,
    body: SettingUpsertRequest,
    session: SessionDep,
) -> SettingResponse:
    try:
        return SettingService(session).upsert_setting(key, body.value)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/{key}")
def delete_setting(key: str, session: SessionDep):
    try:
        deleted = SettingService(session).delete_setting(key)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Setting '{key}' is not set",
        )
    return {"ok": True}


# ── Provider validation & testing ────────────────────────────────────


@router.post("/validate")
def validate_settings(session: SessionDep):
    s = SettingService(session).load()
    cs = check_config_status(s)
    return {
        "status": "ok" if cs["configReady"] else "needs_config",
        "configReady": cs["configReady"],
        "configError": cs["configError"],
        "provider": s.planning_provider.value,
        "model": s.planning_model,
    }


@router.get("/configured")
def check_configured(session: SessionDep):
    s = SettingService(session).load()
    if s.minds_api_key is not None:
        return {"configured": True, "provider": "minds-cloud"}
    if s.anthropic_api_key is not None:
        return {"configured": True, "provider": "anthropic"}
    if s.openai_api_key is not None:
        return {"configured": True, "provider": "openai"}
    return {"configured": False, "provider": ""}


@router.get("/install-status")
def install_status():
    return {"antonInstalled": True, "serverDepsReady": True}


@router.get("/reveal-key/{name}")
def reveal_key(name: str, session: SessionDep):
    field_map = {
        "anthropic": "anthropic_api_key",
        "openai": "openai_api_key",
        "minds": "minds_api_key",
    }
    field = field_map.get(name.lower())
    if field is None:
        raise HTTPException(status_code=404, detail="Unknown key name")
    s = SettingService(session).load()
    val = getattr(s, field)
    return {"value": val.get_secret_value() if isinstance(val, SecretStr) else ""}


class _TestProvidersBody(BaseModel):
    providers: Optional[list[dict[str, Any]]] = None


@router.post("/test-providers")
async def test_providers(session: SessionDep, body: _TestProvidersBody | None = None):
    s = SettingService(session).load()

    if body and body.providers is not None:
        providers = list(body.providers)
    else:
        # Build a minimal providers list from stored keys
        providers = []
        if s.anthropic_api_key is not None:
            providers.append({"type": "anthropic", "apiKey": ""})
        if s.openai_api_key is not None:
            providers.append({"type": "openai", "apiKey": ""})
        if s.minds_api_key is not None:
            providers.append({"type": "minds-cloud", "apiKey": "", "mindsUrl": s.minds_url})

    for p in providers:
        if p.get("apiKey") in ("***", ""):
            p["apiKey"] = resolve_stored_key(s, p.get("type", ""))

    statuses, details = await ping_providers(providers)
    return {"providerStatus": statuses, "providerStatusDetails": details}


class _ValidateProviderBody(CamelRequest):
    provider: str
    api_key: str
    base_url: Optional[str] = None
    model: Optional[str] = None


@router.post("/validate-provider")
async def validate_provider_endpoint(body: _ValidateProviderBody):
    return await validate_provider_svc(body.provider, body.api_key, body.base_url, body.model)


@router.get("/recommended-models")
async def recommended_models(session: SessionDep):
    """Per-provider model picker options for the Settings UI.

    Returns the static `RECOMMENDED_MODELS`/`RECOMMENDED_PAIR` maps, with
    the `minds-cloud` bucket overlaid by MindsHub's live `/v1/models` list
    when a Minds key + URL are configured. Falls back to the static list
    (the `latest:*` aliases) when the key is absent or the endpoint can't
    be reached."""
    recommended = {k: list(v) for k, v in RECOMMENDED_MODELS.items()}
    pair = {k: list(v) for k, v in RECOMMENDED_PAIR.items()}

    # `modelEfforts` maps a model id → {"efforts": [...], "default": "..."} and is
    # the single capability surface for the UI: a model accepts an effort level
    # iff it has an entry here. Direct (BYOK) provider levels are hand-maintained
    # in DIRECT_EFFORT_CATALOG; minds-cloud levels come live from /v1/models and
    # win on any key collision.
    model_efforts: dict[str, dict] = {k: dict(v) for k, v in DIRECT_EFFORT_CATALOG.items()}

    s = SettingService(session).load()
    if s.minds_api_key is not None and s.minds_url:
        live, live_efforts = await fetch_minds_models(
            s.minds_url, s.minds_api_key.get_secret_value()
        )
        if live:
            recommended["minds-cloud"] = live
        model_efforts.update(live_efforts)

    # Overlay a configured custom OpenAI-compatible endpoint the same way as
    # minds-cloud. The provider card's own baseUrl is authoritative — the
    # shared openai_base_url setting is also reused by gemini/openai — so read
    # it from providers_json. fetch_minds_models is just an OpenAI-compatible
    # /models fetch and returns None on failure, leaving the bucket empty so
    # the picker falls back to a free-text model input.
    try:
        provider_cards = json.loads(s.providers_json or "[]")
    except (ValueError, TypeError):
        provider_cards = []
    oc_card = next(
        (
            c for c in provider_cards
            if isinstance(c, dict)
            and c.get("type") in ("openai-compatible", "openai_compatible")
            and (c.get("baseUrl") or "").strip()
        ),
        None,
    )
    if oc_card:
        oc_key = s.openai_api_key.get_secret_value() if s.openai_api_key else ""
        live, live_efforts = await fetch_minds_models(oc_card["baseUrl"].strip(), oc_key)
        if live:
            recommended["openai-compatible"] = live
        model_efforts.update(live_efforts)

    return {
        "recommendedModels": recommended,
        "recommendedPair": pair,
        "modelEfforts": model_efforts,
    }


# ── Raw .env access (legacy, used by Onboarding) ─────────────────────

_ENV_PATH = Path.home() / ".anton" / ".env"


def _parse_dotenv_content(content: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        result[key.strip()] = val.strip().strip('"').strip("'")
    return result


@router.get("/raw")
def read_raw_settings():
    if not _ENV_PATH.exists():
        return {}
    try:
        return _parse_dotenv_content(_ENV_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


class _RawSettingsBody(BaseModel):
    content: str


@router.post("/raw")
def write_raw_settings(body: _RawSettingsBody, session: SessionDep):
    """Merge dotenv content into ~/.anton/.env and sync recognised keys to the DB.

    Uses key-level merge (not full overwrite) because callers like the
    OAuth token refresh only send a subset of keys — a full overwrite
    would wipe model config and other settings from .env.

    The .env persists for the standalone ``anton`` CLI and is read by
    ``GET /raw``; the DB is authoritative for cowork-server.  By syncing
    to both we keep them consistent regardless of which frontend code
    path writes settings (onboarding, OAuth token refresh, etc.)."""
    from cowork.migrations import sync_env_vars_to_db

    incoming = _parse_dotenv_content(body.content)

    try:
        existing = read_raw_settings()
        existing.update(incoming)

        # Sync recognised ANTON_* vars to the DB first. If validation fails,
        # leave the legacy .env untouched so the DB remains authoritative.
        sync_env_vars_to_db(session, existing)

        _ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{k}={v}" for k, v in existing.items()]
        _ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            _ENV_PATH.chmod(0o600)
        except OSError:
            pass
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Settings could not be saved.") from e

    return {"ok": True}


# NOTE: .env → DB migration now runs at server startup via
# cowork.migrations.migrate_env_to_db(), called from dev_setup.
