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
from typing import Annotated, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlmodel import Session

from cowork.api.v1.endpoints.guards import require_local, require_local_tenancy
from cowork.common.paths import cowork_home
from cowork.db.scoped import TenantScope, get_tenant_scope
from cowork.db.session import get_session
from cowork.principal import Principal, can_manage_org, get_principal
from cowork.schemas.base import CamelRequest
from cowork.schemas.settings import (
    SettingResponse,
    SettingsBulkUpsertRequest,
    SettingUpsertRequest,
)
from cowork.services.providers import (
    check_config_status,
    fetch_minds_models,
    fetch_org_model_catalog,
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
from cowork.common.settings.user_settings import (
    Provider,
    provider_api_key_str,
    setting_is_org_scoped,
)

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]
# Tenant scope for every settings read/write (local mode → LOCAL_SCOPE).
ScopeDep = Annotated[TenantScope, Depends(get_tenant_scope)]
PrincipalDep = Annotated[Principal | None, Depends(get_principal)]


def _require_org_admin_for(keys, scope: TenantScope, principal: Principal | None) -> None:
    """Org-level settings (provider keys, model policy, budgets) are admin-owned:
    in org mode, a caller-supplied write to any org-classified key needs the
    manage-organization role. Personal keys and local mode are untouched. 403,
    not 404 — the key names are public schema, only the write is privileged.
    System-derived writes are exempt (see the minds_model_enabled refresh in
    recommended_models) — the caller can't steer the stored value."""
    if not scope.org_mode:
        return
    org_keys = [k for k in keys if setting_is_org_scoped(k)]
    if org_keys and not can_manage_org(principal):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"changing organization settings ({', '.join(sorted(org_keys))}) requires an org admin",
        )


@router.get("/", response_model=list[SettingResponse])
def list_settings(session: SessionDep, scope: ScopeDep) -> list[SettingResponse]:
    return SettingService(session, scope).list_settings()


@router.put("/")
def bulk_upsert_settings(
    body: SettingsBulkUpsertRequest, session: SessionDep, scope: ScopeDep, principal: PrincipalDep
):
    """Upsert many settings in one transaction (all-or-nothing).

    The Settings form saves through here: either every value validates and is
    written, or the request 400s and nothing is written. Replaces the per-key
    PUT loop, which could leave the DB half-written on a partial failure.
    """
    # Skipped values (None/"***") are never written — don't let them trip the gate.
    live_keys = [k for k, v in (body.values or {}).items() if v is not None and v != "***"]
    _require_org_admin_for(live_keys, scope, principal)
    try:
        updated = SettingService(session, scope).save_all(body.values)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"updated": updated}


@router.put("/{key}", response_model=SettingResponse)
def upsert_setting(
    key: str,
    body: SettingUpsertRequest,
    session: SessionDep,
    scope: ScopeDep,
    principal: PrincipalDep,
) -> SettingResponse:
    _require_org_admin_for([key], scope, principal)
    try:
        return SettingService(session, scope).upsert_setting(key, body.value)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/{key}")
def delete_setting(key: str, session: SessionDep, scope: ScopeDep, principal: PrincipalDep):
    _require_org_admin_for([key], scope, principal)
    try:
        deleted = SettingService(session, scope).delete_setting(key)
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
def validate_settings(session: SessionDep, scope: ScopeDep):
    s = SettingService(session, scope).load()
    cs = check_config_status(s)
    return {
        "status": "ok" if cs["configReady"] else "needs_config",
        "configReady": cs["configReady"],
        "configError": cs["configError"],
        "provider": s.planning_provider.value,
        "model": s.planning_model,
    }


@router.get("/configured")
def check_configured(session: SessionDep, scope: ScopeDep):
    s = SettingService(session, scope).load()
    if s.minds_api_key is not None:
        return {"configured": True, "provider": "minds-cloud"}
    if s.anthropic_api_key is not None:
        return {"configured": True, "provider": "anthropic"}
    if s.openai_api_key is not None:
        return {"configured": True, "provider": "openai"}
    # gemini / openai-compatible have dedicated key slots now — a user who
    # configured only one of those (no shared openai key) must still read as
    # configured (this endpoint gates app startup).
    if s.gemini_api_key is not None:
        return {"configured": True, "provider": "gemini"}
    if s.openai_compatible_api_key is not None:
        return {"configured": True, "provider": "openai-compatible"}
    return {"configured": False, "provider": ""}


@router.post("/logout")
def logout_clear_credentials(session: SessionDep, scope: ScopeDep):
    """Clear all stored credentials from the DB (desktop sign-out flow, so
    ``/health`` returns ``config_ready: false``; preferences are kept).

    Org mode: a no-op. Credentials are org-owned there — one member signing
    out must not wipe the keys the whole org runs on.
    """
    if scope.org_mode:
        return {"ok": True, "deleted": []}
    deleted = SettingService(session, scope).clear_credentials()
    return {"ok": True, "deleted": deleted}


@router.get("/install-status")
def install_status():
    return {"antonInstalled": True, "serverDepsReady": True}


@router.get("/reveal-key/{name}")
def reveal_key(name: str, session: SessionDep, scope: ScopeDep, request: Request):
    # reveal-key returns an unmasked provider secret — loopback only (ENG-457).
    require_local(request)
    name_map = {
        "anthropic": Provider.ANTHROPIC,
        "openai": Provider.OPENAI,
        "gemini": Provider.GEMINI,
        "openai-compatible": Provider.OPENAI_COMPATIBLE,
        "minds": Provider.MINDS_CLOUD,
        "minds-cloud": Provider.MINDS_CLOUD,
    }
    provider = name_map.get(name.lower())
    if provider is None:
        raise HTTPException(status_code=404, detail="Unknown key name")
    s = SettingService(session, scope).load()
    # provider_api_key_str applies the gemini/openai-compatible → openai fallback.
    return {"value": provider_api_key_str(s, provider)}


class _TestProvidersBody(BaseModel):
    providers: Optional[list[dict[str, Any]]] = None


@router.post("/test-providers")
async def test_providers(session: SessionDep, scope: ScopeDep, body: _TestProvidersBody | None = None):
    """Ping the given (or all stored) providers and return connectivity results.

    Read-only: a test is a point-in-time check and no longer writes to the DB.
    Persisting made a "test" a silent write, and a stored green dot could
    outlive a revoked key or a drained wallet and read as passing when it no
    longer was (ENG-335). Callers render the returned results directly.
    """
    s = SettingService(session, scope).load()

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


def _fill_missing(target: dict, extra: dict, *, skip: Optional[set[str]] = None) -> None:
    """Add only the keys `target` doesn't already have (first writer wins).

    `skip` reserves ids the later writer may not fill even where `target` has no
    entry for them, for a map whose gaps carry meaning of their own.
    """
    for key, value in extra.items():
        if skip and key in skip:
            continue
        target.setdefault(key, value)


@router.get("/recommended-models")
async def recommended_models(request: Request, session: SessionDep, scope: ScopeDep, refresh: bool = False):
    """Per-provider model picker options for the Settings UI.

    Returns the static `RECOMMENDED_MODELS`/`RECOMMENDED_PAIR` maps, with
    the `minds-cloud` bucket overlaid by MindsHub's live `/v1/models` list
    when a Minds key + URL are configured. Falls back to the static list
    (the `latest:*` aliases) when the key is absent or the endpoint can't
    be reached.

    `refresh=true` bypasses fetch_minds_models's cache (`force_refresh`) — the
    Settings UI passes it when the model dropdown is opened, since the cached
    `enabled` map can otherwise mask a wallet top-up for its 5-minute TTL. The
    plain mount-time load omits it and stays cache-eligible. A cached *failure*
    is honored either way, so an unreachable MindsHub doesn't cost every open
    the full HTTP timeout."""
    recommended = {k: list(v) for k, v in RECOMMENDED_MODELS.items()}
    pair = {k: list(v) for k, v in RECOMMENDED_PAIR.items()}

    # `modelEfforts` maps a model id → {"efforts": [...], "default": "..."} and is
    # the single capability surface for the UI: a model accepts an effort level
    # iff it has an entry here. Direct (BYOK) provider levels are hand-maintained
    # in DIRECT_EFFORT_CATALOG; minds-cloud levels come live from /v1/models and
    # win on any key collision.
    model_efforts: dict[str, dict] = {k: dict(v) for k, v in DIRECT_EFFORT_CATALOG.items()}

    # `modelEnabled` maps a model id → bool. MindsHub lists models the org's
    # wallet can't currently pay for / whose free allowance is exhausted (marked
    # enabled:false) so the picker can show them as locked, with an "add credits"
    # affordance; a model absent from this map is treated as available. Additive
    # alongside modelEfforts — consumers that ignore it keep working.
    model_enabled: dict[str, bool] = {}

    # `modelLabels` maps a model id → MindsHub's human-readable display label.
    # Display-only: the picker uses this to render the option text, but the id
    # remains the value used for selection/storage/resolution everywhere else.
    # A model absent from this map falls back to the client's id-derived label.
    model_labels: dict[str, str] = {}

    # Grouping metadata for the picker:
    #   modelProviders id → the provider serving the model ("anthropic"), which
    #                  decides its section.
    #   modelFamilies  id → the moving alias the model belongs to. Dense for every
    #                  model MindsHub describes, and equal to the id on a moving
    #                  alias, so `families[id] == id` marks a model "latest" and
    #                  anything else names the alias it is a frozen version of.
    #
    # Both are keyed by model id ALONE and are NOT scoped to the minds-cloud
    # bucket — population is gated on a MindsHub key being configured, with no
    # reference to which provider the user actually selected. So they are empty
    # only for a user with no MindsHub key at all, NOT for a user on BYOK.
    #
    # A consumer must therefore decide per model, by looking its id up, and must
    # not read "the map is non-empty" as "this model is described". For
    # `modelLabels`/`modelEnabled` an absent key is a benign fallback, but for
    # `modelFamilies` absence is actively interpreted: read as "is its own head" it
    # tags every BYOK model "latest", including dated snapshots that never move.
    model_providers: dict[str, str] = {}
    model_families: dict[str, str] = {}

    # Every model id MindsHub listed, whether or not it described it. The custom
    # endpoint overlay below reserves these ids for the grouping maps.
    minds_ids: set[str] = set()

    s = SettingService(session, scope).load()
    listing = None
    if scope.org_mode:
        # Org catalog: operator endpoint + the caller's own bearer, per org.
        # Never the stored key or s.minds_url (both tenant-settable), or a
        # member's JWT could be forwarded to an admin-chosen host.
        bearer = request.headers.get("Authorization", "")
        token = bearer[7:].strip() if bearer.lower().startswith("bearer ") else ""
        if token and scope.org_id:
            listing = await fetch_org_model_catalog(
                org_id=scope.org_id, bearer_token=token, refresh=refresh
            )
    elif s.minds_api_key is not None and s.minds_url:
        # Desktop / BYOK: the user's stored key against its configured URL.
        listing = await fetch_minds_models(
            s.minds_url, s.minds_api_key.get_secret_value(), force_refresh=refresh
        )
    if listing is not None:
        live = listing.ids
        live_efforts = listing.efforts
        live_enabled = listing.enabled
        live_labels = listing.labels
        if live:
            recommended["minds-cloud"] = live
            # Cache the availability map so model-default resolution
            # (UserSettings._minds_enabled_map) can avoid locked models without
            # a network call in the turn path. Adding credits re-enables the
            # canonical defaults on the next settings load.
            #
            # Guard on `live_enabled` (the map we actually write), NOT `live`
            # (the id list): a gateway that returns ids without `enabled` flags
            # yields `live` non-empty but `live_enabled == {}`, and writing {}
            # would wipe a previously-good map — re-locking the canonical
            # default, i.e. the exact bug this PR fixes. And only write on a real
            # change: this endpoint is hit on every boot/settings-open, and
            # upsert_setting commits a row + invalidates the settings cache, so
            # an unconditional write churns every UserSettings reader.
            if live_enabled:
                # Persist ORDER-PRESERVING JSON — never sort_keys. The
                # first-enabled default fallback (_enabled_aware_default)
                # iterates the map in insertion order, relying on /v1/models
                # listing the free/baseline model first; an alphabetized map
                # could silently promote the wrong model. The compare is
                # order-sensitive too, so a gateway re-ranking (same set, new
                # baseline first) also counts as a change and refreshes the map.
                desired = json.dumps(live_enabled)
                try:
                    stored = json.dumps(json.loads(s.minds_model_enabled or "{}"))
                except (ValueError, TypeError):
                    stored = "{}"
                if desired != stored:
                    # Intentionally ungated by _require_org_admin_for: the value
                    # is system-derived (MindsHub, via admin-set key/URL), so a
                    # member can trigger this refresh but can't steer what's
                    # stored — and gating it would leave the map stale.
                    SettingService(session, scope).upsert_setting("minds_model_enabled", desired)
        model_efforts.update(live_efforts)
        model_enabled.update(live_enabled)
        model_labels.update(live_labels)
        model_providers.update(listing.providers)
        model_families.update(listing.families)
        minds_ids.update(live or ())

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
        # Read the openai-compatible key via provider_api_key_str so a user who
        # set a dedicated openai_compatible_api_key is used (falls back to the
        # shared openai_api_key), matching how the provider is actually built.
        oc_key = provider_api_key_str(s, Provider.OPENAI_COMPATIBLE)
        oc_listing = await fetch_minds_models(
            oc_card["baseUrl"].strip(), oc_key, force_refresh=refresh,
            tenant_key=scope.org_id if scope.org_mode else None,
        )
        live = oc_listing.ids
        live_efforts = oc_listing.efforts
        live_enabled = oc_listing.enabled
        live_labels = oc_listing.labels
        if live:
            recommended["openai-compatible"] = live
        # These maps are keyed by model id alone, not by (provider, id), so a
        # custom endpoint serving an id MindsHub also serves would other-
        # wise decide that model's label, effort levels and locked state for the
        # minds-cloud bucket too — letting a BYO base URL rename or lock a
        # MindsHub model in the picker. MindsHub is fetched first and wins;
        # the custom endpoint only fills ids MindsHub didn't describe.
        _fill_missing(model_efforts, live_efforts)
        _fill_missing(model_enabled, live_enabled)
        _fill_missing(model_labels, live_labels)
        # Same precedence for the grouping metadata, for the same reason: a BYO
        # endpoint must not be able to move a MindsHub model into another vendor's
        # section or relabel it as an old version of something. But here the
        # reservation is every id MindsHub LISTED, not just the ids it described,
        # because MindsHub can serve a model and say nothing about it:
        #
        #   - deployed ahead of the MindsHub release that publishes these fields,
        #     both MindsHub maps are empty for every model it serves, so a custom
        #     endpoint answering `{"id": "sonnet", "family": "haiku"}` would file
        #     MindsHub's own sonnet under haiku as an older version of it;
        #   - a MindsHub row carrying a family but no usable provider is recorded
        #     in families only (see fetch_minds_models), never in providers, so a
        #     custom endpoint's provider would pick that model's section.
        #
        # In both cases MindsHub's silence is its answer, and the app degrades on
        # it correctly (no section / no "latest" tag); a BYO endpoint's answer for
        # a model it doesn't serve is a wrong one, not a better one.
        #
        # Not covered here, because it can't be: when the MindsHub fetch fails,
        # recommendedModels["minds-cloud"] comes back [], which the app's
        # overlayLists skips so the picker keeps the minds ids it already holds,
        # while these maps come back describing the custom endpoint's ids only and
        # the app's overlayMap swaps its held map out wholesale. Those held minds
        # ids lose their grouping metadata until a fetch succeeds again.
        reserved_ids = minds_ids | model_providers.keys() | model_families.keys()
        _fill_missing(model_providers, oc_listing.providers, skip=reserved_ids)
        _fill_missing(model_families, oc_listing.families, skip=reserved_ids)

    return {
        "recommendedModels": recommended,
        "recommendedPair": pair,
        "modelEfforts": model_efforts,
        "modelEnabled": model_enabled,
        "modelLabels": model_labels,
        "modelProviders": model_providers,
        "modelFamilies": model_families,
    }


# ── Raw .env access (legacy, used by Onboarding) ─────────────────────

_ENV_PATH = cowork_home() / ".env"


def _parse_dotenv_content(content: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def _read_env_dict() -> dict[str, str]:
    if not _ENV_PATH.exists():
        return {}
    try:
        return _parse_dotenv_content(_ENV_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


@router.get("/raw")
def read_raw_settings(request: Request):
    # /raw dumps the dotenv verbatim (all provider secrets) — same loopback
    # restriction as reveal-key, and desktop-only (deployment-global state).
    require_local_tenancy()
    require_local(request)
    return _read_env_dict()


class _RawSettingsBody(BaseModel):
    content: str


@router.post("/raw")
def write_raw_settings(body: _RawSettingsBody, session: SessionDep, request: Request):
    """Merge dotenv content into ~/.cowork/.env and sync recognised keys to the DB.

    Uses key-level merge (not full overwrite) because callers like the
    OAuth token refresh only send a subset of keys — a full overwrite
    would wipe model config and other settings from .env.

    The .env persists for the standalone ``anton`` CLI and is read by
    ``GET /raw``; the DB is authoritative for cowork-server.  By syncing
    to both we keep them consistent regardless of which frontend code
    path writes settings (onboarding, OAuth token refresh, etc.)."""
    # Writing the dotenv lands provider secrets on disk and syncs them into
    # global settings rows — loopback-only, and desktop-only.
    require_local_tenancy()
    require_local(request)

    from cowork.migrations import sync_env_vars_to_db

    incoming = _parse_dotenv_content(body.content)

    try:
        existing = _read_env_dict()
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
