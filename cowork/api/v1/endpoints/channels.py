from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session

from cowork.db.scoped import ScopedSession, ScopedSessionDep
from cowork.db.session import get_session
from cowork.schemas.channels import (
    ChannelAgentResponse,
    ChannelAgentUpdateRequest,
    BindingCreateRequest,
    BindingResponse,
    BindingUpdateRequest,
    ChannelConfigResponse,
    ChannelConfigUpdateRequest,
    ChannelInstallationResponse,
    ChannelLifecycleResponse,
    ChannelReloadResponse,
    ChannelStatusResponse,
    ChannelTestConnectionResponse,
    PluginResponse,
)
from cowork.channels.lifecycle import LifecycleError
from cowork.channels.ingress import sync_channel_ingress
from cowork.services.channel_bindings import (
    BindingConflictError,
    BindingNotFoundError,
    ChannelBindingService,
)
from cowork.services.channel_lifecycle import (
    ChannelLifecycleService,
    LifecycleNotImplementedError,
)
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import TenantScope
from cowork.principal import Principal, can_manage_org, get_principal
from cowork.services.channels import ChannelConfigService, UnknownChannelError
from cowork.harnesses.base import available_harness_ids

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]
PrincipalDep = Annotated[Principal | None, Depends(get_principal)]


def _live_adapters(request: Request):
    return getattr(request.app.state, "channel_adapters", None)


def _require_org_admin(scope: TenantScope, principal: Principal | None) -> None:
    """Configuring channels — credentials, lifecycle, or the shared channel
    agent — is admin-owned in org mode, the same rule settings.py's
    _require_org_admin_for applies to org settings writes. Checked before any
    other gate (readiness, unknown-channel, ...): who may act comes first."""
    if scope.org_mode and not can_manage_org(principal):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="configuring channels requires an org admin",
        )


def _require_org_ready(channel_type: str) -> None:
    """Blocks the mutating config/lifecycle actions for a channel with no
    per-org routing key (Telegram, WhatsApp) in org mode — see
    services.channels.is_org_ready. GET/list stay open; they're read-only."""
    from cowork.channels.registry import get_registry
    from cowork.services.channels import is_org_ready

    plugin = get_registry().get(channel_type)
    if plugin is not None and not is_org_ready(plugin):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"channel {channel_type} is not yet available in org deployments",
        )


async def _reconcile_ingress(request: Request, channel_type: str, org_id: str | None) -> None:
    """Start/stop background ingress (Gateway stream or tunnel-free poll) after
    a change to the channel's live adapter (config/setup/teardown/reload).
    org_id=None in local mode preserves today's single global slot; a real
    org_id in org mode goes through IngressManager's per-org Redis lease."""
    manager = getattr(request.app.state, "channel_ingress", None)
    await sync_channel_ingress(manager, _live_adapters(request), channel_type, org_id)


@router.get("/status", response_model=ChannelStatusResponse)
def channel_status(scoped: ScopedSessionDep) -> ChannelStatusResponse:
    return ChannelConfigService(scoped).status()


@router.get("/plugins", response_model=list[PluginResponse])
def list_plugins(scoped: ScopedSessionDep) -> list[PluginResponse]:
    return ChannelConfigService(scoped).list_plugins()


@router.get("/installations", response_model=list[ChannelInstallationResponse])
def list_installations(scoped: ScopedSessionDep) -> list[ChannelInstallationResponse]:
    return ChannelConfigService(scoped).list_installations()


@router.get("/agent", response_model=ChannelAgentResponse)
def get_channel_agent(scoped: ScopedSessionDep) -> ChannelAgentResponse:
    """The harness that serves channel conversations. Distinct from the desktop
    harness setting and applied to new conversations (existing ones stay pinned
    to whatever first served them)."""
    from cowork.common.settings.user_settings import get_user_settings

    current = (get_user_settings(scoped.scope).channels_harness or "").strip() or "anton"
    return ChannelAgentResponse(harness=current, options=available_harness_ids())


@router.put("/agent", response_model=ChannelAgentResponse)
def set_channel_agent(
    body: ChannelAgentUpdateRequest, session: SessionDep, scoped: ScopedSessionDep,
    principal: PrincipalDep,
) -> ChannelAgentResponse:
    # channels_harness is ORG-marked, so SettingService routes this write to
    # the caller's own org row — two orgs never share or overwrite each other's.
    from cowork.common.settings.user_settings import get_user_settings
    from cowork.services.settings import SettingService

    _require_org_admin(scoped.scope, principal)
    options = available_harness_ids()
    harness = (body.harness or "").strip()
    if harness not in options:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown harness '{harness}'. Available: {', '.join(options) or 'none'}",
        )
    previous = (get_user_settings(scoped.scope).channels_harness or "").strip()
    SettingService(session, scoped.scope).upsert_setting("channels_harness", harness)

    # Changing the agent re-points existing chats: detach their conversations
    # so the next message starts fresh under the new agent (existing pins are
    # per-conversation, so without this only new chats would switch).
    reset = 0
    if harness != previous:
        reset = ChannelBindingService(scoped).reset_conversations()
    return ChannelAgentResponse(harness=harness, options=options, reset_conversations=reset)


@router.get("/{channel_type}/config", response_model=ChannelConfigResponse)
def get_config(channel_type: str, scoped: ScopedSessionDep) -> ChannelConfigResponse:
    try:
        return ChannelConfigService(scoped).get_config(channel_type)
    except UnknownChannelError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown channel: {channel_type}")


@router.put("/{channel_type}/config", response_model=ChannelConfigResponse)
async def set_config(
    channel_type: str,
    body: ChannelConfigUpdateRequest,
    request: Request,
    scoped: ScopedSessionDep,
    principal: PrincipalDep,
) -> ChannelConfigResponse:
    _require_org_admin(scoped.scope, principal)
    _require_org_ready(channel_type)
    try:
        result = ChannelConfigService(scoped).set_config(channel_type, body.values)
    except UnknownChannelError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown channel: {channel_type}")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    adapters = _live_adapters(request)
    if adapters is not None:
        await adapters.refresh(channel_type, session=scoped)
    await _reconcile_ingress(request, channel_type, scoped.scope.org_id)
    return result


@router.delete("/{channel_type}/config", status_code=status.HTTP_204_NO_CONTENT)
async def delete_config(
    channel_type: str, request: Request, scoped: ScopedSessionDep, principal: PrincipalDep
) -> None:
    _require_org_admin(scoped.scope, principal)
    _require_org_ready(channel_type)
    try:
        deleted = ChannelConfigService(scoped).delete_config(channel_type)
    except UnknownChannelError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown channel: {channel_type}")
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no stored config for channel: {channel_type}",
        )
    # Tear the live adapter down — its credentials are gone.
    adapters = _live_adapters(request)
    if adapters is not None:
        await adapters.remove(channel_type)
    await _reconcile_ingress(request, channel_type, scoped.scope.org_id)


@router.post("/{channel_type}/reload", response_model=ChannelReloadResponse)
async def reload_channel(
    channel_type: str, request: Request, scoped: ScopedSessionDep, principal: PrincipalDep
) -> ChannelReloadResponse:
    """Rebuild a channel's live adapter from its currently stored config."""
    _require_org_admin(scoped.scope, principal)
    _require_org_ready(channel_type)
    try:
        ChannelConfigService(scoped).get_config(channel_type)
    except UnknownChannelError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown channel: {channel_type}")
    adapters = _live_adapters(request)
    active = False
    if adapters is not None:
        active = await adapters.refresh(channel_type, session=scoped)
    await _reconcile_ingress(request, channel_type, scoped.scope.org_id)
    return ChannelReloadResponse(channel_type=channel_type, active=active)


@router.post("/{channel_type}/test-connection", response_model=ChannelTestConnectionResponse)
async def test_connection(channel_type: str, scoped: ScopedSessionDep) -> ChannelTestConnectionResponse:
    """Calls the platform to check the STORED credentials actually
    authenticate — not admin-gated, since it reads but never writes."""
    try:
        result = await ChannelConfigService(scoped).test_connection(channel_type)
    except UnknownChannelError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown channel: {channel_type}")
    return ChannelTestConnectionResponse(channel_type=channel_type, ok=result.ok, detail=result.detail)


@router.get("/bindings", response_model=list[BindingResponse])
def list_bindings(scoped: ScopedSessionDep, channel_type: str | None = None) -> list[BindingResponse]:
    return ChannelBindingService(scoped).list(channel_type=channel_type)


@router.post("/bindings", response_model=BindingResponse, status_code=status.HTTP_201_CREATED)
def create_binding(body: BindingCreateRequest, scoped: ScopedSessionDep) -> BindingResponse:
    try:
        return ChannelBindingService(scoped).create(body)
    except BindingConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.patch("/bindings/{binding_id}", response_model=BindingResponse)
def update_binding(binding_id: UUID, body: BindingUpdateRequest, scoped: ScopedSessionDep) -> BindingResponse:
    try:
        return ChannelBindingService(scoped).update(binding_id, body)
    except BindingNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"binding not found: {binding_id}")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/bindings/{binding_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_binding(binding_id: UUID, scoped: ScopedSessionDep) -> None:
    if not ChannelBindingService(scoped).delete(binding_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"binding not found: {binding_id}")


def _lifecycle_service(request: Request, scoped: ScopedSession) -> ChannelLifecycleService:
    adapters = _live_adapters(request)
    if adapters is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="channels runtime not initialized",
        )
    return ChannelLifecycleService(scoped, adapters)


@router.post("/{channel_type}/setup", response_model=ChannelLifecycleResponse)
async def setup_channel(
    channel_type: str, request: Request, scoped: ScopedSessionDep, principal: PrincipalDep
) -> ChannelLifecycleResponse:
    _require_org_admin(scoped.scope, principal)
    _require_org_ready(channel_type)
    svc = _lifecycle_service(request, scoped)
    try:
        result = await svc.setup(channel_type)
    except UnknownChannelError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown channel: {channel_type}")
    except LifecycleNotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"setup not implemented for channel: {channel_type}",
        )
    except LifecycleError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    await _reconcile_ingress(request, channel_type, scoped.scope.org_id)
    return ChannelLifecycleResponse(channel_type=channel_type, action="setup", active=result.active, detail=result.detail)


@router.post("/{channel_type}/teardown", response_model=ChannelLifecycleResponse)
async def teardown_channel(
    channel_type: str, request: Request, scoped: ScopedSessionDep, principal: PrincipalDep
) -> ChannelLifecycleResponse:
    _require_org_admin(scoped.scope, principal)
    _require_org_ready(channel_type)
    svc = _lifecycle_service(request, scoped)
    try:
        result = await svc.teardown(channel_type)
    except UnknownChannelError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown channel: {channel_type}")
    except LifecycleNotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"teardown not implemented for channel: {channel_type}",
        )
    except LifecycleError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    await _reconcile_ingress(request, channel_type, scoped.scope.org_id)
    return ChannelLifecycleResponse(channel_type=channel_type, action="teardown", active=result.active, detail=result.detail)
