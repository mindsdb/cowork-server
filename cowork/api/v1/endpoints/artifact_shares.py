"""Per-artifact share API endpoints.

Mirrors the router/auth conventions in
``cowork/api/v1/endpoints/artifacts.py`` (``APIRouter``, ``SessionDep``,
``PrincipalDep``, ``get_request_principal``) for PER-ARTIFACT sharing
with a Google-Drive-style lightweight-signup accept flow.

Create/list/update operations require edit capability on the artifact's
project when that project is owned. ``POST /shares/accept`` remains the
lightweight-signup token flow and validates the invited email against
the grant.
"""
from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import AliasChoices, BaseModel, Field
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.models.artifact import Artifact
from cowork.models.identity import ArtifactShare
from cowork.services.artifact_shares import (
    accept_share as _accept_share,
    create_share as _create_share,
    list_shares as _list_shares,
    revoke_share as _revoke_share,
    set_share_role as _set_share_role,
)
from cowork.services.request_identity import (
    AuthenticationError,
    RequestPrincipal,
    principal_from_authorization_header,
)
from cowork.services.project_permissions import ProjectPermissionError, require_project_permission_if_owned

router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


def get_request_principal(request: Request) -> RequestPrincipal | None:
    try:
        return principal_from_authorization_header(request.headers.get("authorization"))
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


PrincipalDep = Annotated[RequestPrincipal | None, Depends(get_request_principal)]


def _share_creator(principal: RequestPrincipal | None) -> str | None:
    if principal is None:
        return None
    return principal.email or principal.subject


def _require_artifact_capability(
    session: Session,
    artifact_id: UUID,
    principal: RequestPrincipal | None,
    capability: str,
) -> Artifact:
    artifact = session.get(Artifact, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    if artifact.project_id is None:
        return artifact
    try:
        require_project_permission_if_owned(
            session,
            artifact.project_id,
            principal.email if principal is not None else None,
            capability,
        )
    except ProjectPermissionError as exc:
        if principal is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication is required for this shared project",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to share this artifact",
        ) from exc
    return artifact


def _require_share_capability(
    session: Session,
    share_id: UUID,
    principal: RequestPrincipal | None,
    capability: str,
) -> ArtifactShare:
    row = session.get(ArtifactShare, share_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Share not found")
    _require_artifact_capability(session, row.artifact_id, principal, capability)
    return row


class _CreateShareBody(BaseModel):
    model_config = {"populate_by_name": True}

    grantee_email: str = Field(validation_alias=AliasChoices("grantee_email", "granteeEmail", "email"))
    role: str = "viewer"
    ttl_days: int = Field(default=7, validation_alias=AliasChoices("ttl_days", "ttlDays"))


class _AcceptShareBody(BaseModel):
    model_config = {"populate_by_name": True}

    token: str
    email: str
    display_name: str | None = Field(default=None, validation_alias=AliasChoices("display_name", "displayName"))


class _UpdateShareBody(BaseModel):
    model_config = {"populate_by_name": True}

    role: str | None = None
    status: str | None = None


@router.post("/{artifact_id}/shares", status_code=status.HTTP_201_CREATED)
def create_artifact_share(artifact_id: UUID, req: _CreateShareBody, session: SessionDep, principal: PrincipalDep):
    try:
        _require_artifact_capability(session, artifact_id, principal, "edit")
        return _create_share(
            session,
            artifact_id=artifact_id,
            grantee_email=req.grantee_email,
            role=req.role,
            created_by=_share_creator(principal),
            ttl_days=req.ttl_days,
        )
    except ValueError as exc:
        code = status.HTTP_404_NOT_FOUND if "not found" in str(exc).lower() else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@router.get("/{artifact_id}/shares")
def list_artifact_shares(artifact_id: UUID, session: SessionDep, principal: PrincipalDep):
    try:
        _require_artifact_capability(session, artifact_id, principal, "edit")
        return _list_shares(session, artifact_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/shares/accept")
def accept_artifact_share(req: _AcceptShareBody, session: SessionDep, principal: PrincipalDep):
    try:
        return _accept_share(
            session,
            token=req.token,
            email=req.email,
            display_name=req.display_name,
        )
    except ValueError as exc:
        code = status.HTTP_404_NOT_FOUND if "not found" in str(exc).lower() else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@router.patch("/shares/{share_id}")
def update_artifact_share(share_id: UUID, req: _UpdateShareBody, session: SessionDep, principal: PrincipalDep):
    try:
        _require_share_capability(session, share_id, principal, "edit")
        if req.status is not None:
            if req.status != "revoked":
                raise ValueError("status can only be set to 'revoked'")
            return _revoke_share(session, share_id)
        if req.role is not None:
            return _set_share_role(session, share_id, req.role)
        raise ValueError("Provide a role or status to update")
    except ValueError as exc:
        code = status.HTTP_404_NOT_FOUND if "not found" in str(exc).lower() else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=code, detail=str(exc)) from exc
