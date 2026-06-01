from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.channels import (
    BindingCreateRequest,
    BindingResponse,
    BindingUpdateRequest,
    ChannelConfigResponse,
    ChannelConfigUpdateRequest,
    ChannelInstallationResponse,
    ChannelLifecycleResponse,
    ChannelReloadResponse,
    ChannelStatusResponse,
    PluginResponse,
)
from cowork.channels.lifecycle import LifecycleError
from cowork.services.channel_bindings import (
    BindingConflictError,
    BindingNotFoundError,
    ChannelBindingService,
)
from cowork.services.channel_lifecycle import (
    ChannelLifecycleService,
    LifecycleNotImplementedError,
)
from cowork.services.channels import ChannelConfigService, UnknownChannelError

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]


def _live_adapters(request: Request):
    return getattr(request.app.state, "channel_adapters", None)


@router.get("/status", response_model=ChannelStatusResponse)
def channel_status(session: SessionDep) -> ChannelStatusResponse:
    return ChannelConfigService(session).status()


@router.get("/plugins", response_model=list[PluginResponse])
def list_plugins(session: SessionDep) -> list[PluginResponse]:
    return ChannelConfigService(session).list_plugins()


@router.get("/installations", response_model=list[ChannelInstallationResponse])
def list_installations(session: SessionDep) -> list[ChannelInstallationResponse]:
    return ChannelConfigService(session).list_installations()


@router.get("/{channel_type}/config", response_model=ChannelConfigResponse)
def get_config(channel_type: str, session: SessionDep) -> ChannelConfigResponse:
    try:
        return ChannelConfigService(session).get_config(channel_type)
    except UnknownChannelError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown channel: {channel_type}")


@router.put("/{channel_type}/config", response_model=ChannelConfigResponse)
async def set_config(
    channel_type: str,
    body: ChannelConfigUpdateRequest,
    request: Request,
    session: SessionDep,
) -> ChannelConfigResponse:
    try:
        result = ChannelConfigService(session).set_config(channel_type, body.values)
    except UnknownChannelError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown channel: {channel_type}")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    adapters = _live_adapters(request)
    if adapters is not None:
        await adapters.refresh(channel_type, session=session)
    return result


@router.delete("/{channel_type}/config", status_code=status.HTTP_204_NO_CONTENT)
async def delete_config(channel_type: str, request: Request, session: SessionDep) -> None:
    try:
        deleted = ChannelConfigService(session).delete_config(channel_type)
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


@router.post("/{channel_type}/reload", response_model=ChannelReloadResponse)
async def reload_channel(channel_type: str, request: Request, session: SessionDep) -> ChannelReloadResponse:
    """Rebuild a channel's live adapter from its currently stored config."""
    try:
        ChannelConfigService(session).get_config(channel_type)
    except UnknownChannelError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown channel: {channel_type}")
    adapters = _live_adapters(request)
    active = False
    if adapters is not None:
        active = await adapters.refresh(channel_type, session=session)
    return ChannelReloadResponse(channel_type=channel_type, active=active)


@router.get("/bindings", response_model=list[BindingResponse])
def list_bindings(session: SessionDep, channel_type: str | None = None) -> list[BindingResponse]:
    return ChannelBindingService(session).list(channel_type=channel_type)


@router.post("/bindings", response_model=BindingResponse, status_code=status.HTTP_201_CREATED)
def create_binding(body: BindingCreateRequest, session: SessionDep) -> BindingResponse:
    try:
        return ChannelBindingService(session).create(body)
    except BindingConflictError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.patch("/bindings/{binding_id}", response_model=BindingResponse)
def update_binding(binding_id: UUID, body: BindingUpdateRequest, session: SessionDep) -> BindingResponse:
    try:
        return ChannelBindingService(session).update(binding_id, body)
    except BindingNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"binding not found: {binding_id}")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/bindings/{binding_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_binding(binding_id: UUID, session: SessionDep) -> None:
    if not ChannelBindingService(session).delete(binding_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"binding not found: {binding_id}")


def _lifecycle_service(request: Request, session: Session) -> ChannelLifecycleService:
    adapters = _live_adapters(request)
    if adapters is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="channels runtime not initialized",
        )
    return ChannelLifecycleService(session, adapters)


@router.post("/{channel_type}/setup", response_model=ChannelLifecycleResponse)
async def setup_channel(channel_type: str, request: Request, session: SessionDep) -> ChannelLifecycleResponse:
    svc = _lifecycle_service(request, session)
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
    return ChannelLifecycleResponse(channel_type=channel_type, action="setup", active=result.active, detail=result.detail)


@router.post("/{channel_type}/teardown", response_model=ChannelLifecycleResponse)
async def teardown_channel(channel_type: str, request: Request, session: SessionDep) -> ChannelLifecycleResponse:
    svc = _lifecycle_service(request, session)
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
    return ChannelLifecycleResponse(channel_type=channel_type, action="teardown", active=result.active, detail=result.detail)
