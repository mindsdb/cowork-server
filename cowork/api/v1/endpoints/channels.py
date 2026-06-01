"""Generic channels API — /api/v1/channels.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.channels import (
    ChannelConfigResponse,
    ChannelConfigUpdateRequest,
    ChannelInstallationResponse,
    ChannelStatusResponse,
    PluginResponse,
)
from cowork.services.channels import ChannelConfigService, UnknownChannelError

router = APIRouter()

SessionDep = Annotated[Session, Depends(get_session)]


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
def set_config(
    channel_type: str,
    body: ChannelConfigUpdateRequest,
    session: SessionDep,
) -> ChannelConfigResponse:
    try:
        return ChannelConfigService(session).set_config(channel_type, body.values)
    except UnknownChannelError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown channel: {channel_type}")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/{channel_type}/config", status_code=status.HTTP_204_NO_CONTENT)
def delete_config(channel_type: str, session: SessionDep) -> None:
    try:
        deleted = ChannelConfigService(session).delete_config(channel_type)
    except UnknownChannelError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown channel: {channel_type}")
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no stored config for channel: {channel_type}",
        )
