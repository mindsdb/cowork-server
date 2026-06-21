"""Artifact AI edit pipeline endpoints.

``POST /propose`` returns a reviewable diff for an AI edit (a dry-run that does
not mutate the artifact). ``POST /accept`` applies the edit under optimistic-
concurrency control: it is a base-version compare-and-swap, returning
``{ok: true, versionId}`` on success and HTTP 409 when the artifact has moved
since the client's ``base_version_id``.

This router is intentionally NOT registered here — the integration agent mounts
it in ``cowork/api/v1/router.py`` under the ``/artifacts`` prefix::

    from cowork.api.v1.endpoints import artifact_edits
    api_router.include_router(artifact_edits.router, prefix="/artifacts", tags=["artifact-edits"])

Auth/router conventions mirror ``cowork/api/v1/endpoints/artifacts.py``.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, Field
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.services.artifact_edits import EditConflict, accept_edit, propose_edit
from cowork.services.artifact_versions import _artifact_from_identifier_or_path
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


def _actor_kwargs(principal: RequestPrincipal | None) -> dict[str, str | None]:
    return {
        "actor_name": principal.name if principal is not None else None,
        "actor_email": principal.email if principal is not None else None,
        "actor_subject": principal.subject if principal is not None else None,
    }


def _require_artifact_capability(
    session: Session,
    path: str,
    principal: RequestPrincipal | None,
    capability: str,
) -> None:
    artifact = _artifact_from_identifier_or_path(session, artifact_id=None, path=path)
    if artifact.project_id is None:
        return
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
            detail="You do not have permission to edit this artifact",
        ) from exc


class _ProposeEditBody(BaseModel):
    model_config = {"populate_by_name": True}

    path: str
    target: str
    old_text: str = Field(validation_alias=AliasChoices("old_text", "oldText"))
    new_text: str = Field(validation_alias=AliasChoices("new_text", "newText"))
    base_version_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("base_version_id", "baseVersionId"),
    )


class _AcceptEditBody(BaseModel):
    model_config = {"populate_by_name": True}

    path: str
    target: str
    old_text: str = Field(validation_alias=AliasChoices("old_text", "oldText"))
    new_text: str = Field(validation_alias=AliasChoices("new_text", "newText"))
    base_version_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("base_version_id", "baseVersionId"),
    )


@router.post("/edits/propose")
def propose_artifact_edit(req: _ProposeEditBody, session: SessionDep, principal: PrincipalDep):
    """Dry-run an AI edit and return a reviewable diff (no mutation)."""
    try:
        _require_artifact_capability(session, req.path, principal, "edit")
        return propose_edit(
            session,
            path=req.path,
            target=req.target,
            old_text=req.old_text,
            new_text=req.new_text,
            base_version_id=req.base_version_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/edits/accept")
def accept_artifact_edit(req: _AcceptEditBody, session: SessionDep, principal: PrincipalDep):
    """Apply an AI edit under OCC; HTTP 409 if the base version is stale."""
    try:
        _require_artifact_capability(session, req.path, principal, "edit")
        return accept_edit(
            session,
            path=req.path,
            target=req.target,
            old_text=req.old_text,
            new_text=req.new_text,
            base_version_id=req.base_version_id,
            **_actor_kwargs(principal),
        )
    except EditConflict as conflict:
        # Compare-and-swap lost: the artifact advanced past base_version_id.
        # Shape matches the frontend commitEdit() conflict contract.
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "ok": False,
                "error": "version_conflict",
                "detail": conflict.message,
                "message": conflict.message,
                "baseVersionId": conflict.base_version_id,
                "currentVersionId": conflict.current_version_id,
                "current": conflict.current_version_dict(),
                "conflict": {
                    "message": conflict.message,
                    "currentVersionId": conflict.current_version_id,
                },
            },
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
