from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlmodel import Session

from cowork.db.session import get_session
from cowork.schemas.base import CamelRequest
from cowork.schemas.projects import ProjectCreateRequest, ProjectUpdateRequest
from cowork.services.request_identity import (
    AuthenticationError,
    RequestPrincipal,
    principal_from_authorization_header,
)
from cowork.services.project_collaboration import ProjectCollaborationService
from cowork.services.project_permissions import ProjectPermissionError, require_project_permission_if_owned
from cowork.services.projects import ProjectService
from cowork.services.artifact_activity import list_project_activity as _list_project_activity


router = APIRouter()
SessionDep = Annotated[Session, Depends(get_session)]


def get_request_principal(request: Request) -> RequestPrincipal | None:
    try:
        return principal_from_authorization_header(request.headers.get("authorization"))
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


PrincipalDep = Annotated[RequestPrincipal | None, Depends(get_request_principal)]


class ProjectCollaboratorRequest(CamelRequest):
    email: str
    display_name: str | None = None
    role: str = "viewer"


class ProjectCollaboratorUpdateRequest(CamelRequest):
    display_name: str | None = None
    role: str | None = None


class ProjectInvitationRequest(CamelRequest):
    email: str
    display_name: str | None = None
    role: str = "viewer"
    ttl_days: int = 7


class ProjectInvitationAcceptRequest(CamelRequest):
    token: str
    display_name: str | None = None


class ProjectNotificationHookRequest(CamelRequest):
    kind: str
    target: str = ""
    enabled: bool = True
    events: list[str] | None = None
    secret: str | None = None
    config: dict | None = None


class ProjectNotificationHookUpdateRequest(CamelRequest):
    kind: str | None = None
    target: str | None = None
    enabled: bool | None = None
    events: list[str] | None = None
    secret: str | None = None
    config: dict | None = None


def _raise_project_error(exc: ValueError):
    code = status.HTTP_404_NOT_FOUND if "not found" in str(exc).lower() else status.HTTP_400_BAD_REQUEST
    raise HTTPException(status_code=code, detail=str(exc))


def _require_project_capability_if_owned(
    session: Session,
    project_id: UUID,
    principal: RequestPrincipal | None,
    capability: str,
) -> None:
    try:
        require_project_permission_if_owned(
            session,
            project_id,
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
            detail="You do not have permission to manage this shared project",
        ) from exc


@router.get("/")
def list_projects(session: SessionDep):
    return ProjectService(session).list_projects()


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_project(body: ProjectCreateRequest, session: SessionDep, principal: PrincipalDep):
    project = ProjectService(session).create_project(body.name)
    if principal is not None and principal.email:
        ProjectCollaborationService(session).upsert_collaborator(
            project.id,
            email=principal.email,
            display_name=principal.name,
            role="owner",
        )
        session.refresh(project)
    return project


@router.patch("/{project_id}")
def update_project(project_id: UUID, body: ProjectUpdateRequest, session: SessionDep, principal: PrincipalDep):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        return ProjectService(session).update_project(
            project_id, name=body.name, is_active=body.is_active
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: UUID, session: SessionDep, principal: PrincipalDep):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        found = ProjectService(session).delete_project(project_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")


@router.get("/{project_id}/collaborators")
def list_project_collaborators(project_id: UUID, session: SessionDep, principal: PrincipalDep):
    _require_project_capability_if_owned(session, project_id, principal, "view")
    try:
        return ProjectCollaborationService(session).list_collaborators(project_id)
    except ValueError as exc:
        _raise_project_error(exc)


@router.get("/{project_id}/activity")
def list_project_activity(
    project_id: UUID,
    session: SessionDep,
    principal: PrincipalDep,
    limit: int = Query(default=50, ge=1, le=200),
):
    _require_project_capability_if_owned(session, project_id, principal, "view")
    try:
        return _list_project_activity(session, project_id, limit=limit)
    except ValueError as exc:
        _raise_project_error(exc)


@router.post("/{project_id}/collaborators", status_code=status.HTTP_201_CREATED)
def upsert_project_collaborator(
    project_id: UUID,
    body: ProjectCollaboratorRequest,
    session: SessionDep,
    principal: PrincipalDep,
):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        return ProjectCollaborationService(session).upsert_collaborator(
            project_id,
            email=body.email,
            display_name=body.display_name,
            role=body.role,
        )
    except ValueError as exc:
        _raise_project_error(exc)


@router.get("/{project_id}/invitations")
def list_project_invitations(
    project_id: UUID,
    session: SessionDep,
    principal: PrincipalDep,
    include_closed: bool = Query(default=False),
):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        return ProjectCollaborationService(session).list_invitations(project_id, include_closed=include_closed)
    except ValueError as exc:
        _raise_project_error(exc)


@router.post("/{project_id}/invitations", status_code=status.HTTP_201_CREATED)
def create_project_invitation(
    project_id: UUID,
    body: ProjectInvitationRequest,
    session: SessionDep,
    principal: PrincipalDep,
):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        return ProjectCollaborationService(session).invite_collaborator(
            project_id,
            email=body.email,
            display_name=body.display_name,
            role=body.role,
            invited_by_subject=principal.subject if principal is not None else None,
            invited_by_email=principal.email if principal is not None else None,
            invited_by_name=principal.name if principal is not None else None,
            ttl_days=body.ttl_days,
        )
    except ValueError as exc:
        _raise_project_error(exc)


@router.post("/{project_id}/invitations/accept")
def accept_project_invitation(
    project_id: UUID,
    body: ProjectInvitationAcceptRequest,
    session: SessionDep,
    principal: PrincipalDep,
):
    try:
        return ProjectCollaborationService(session).accept_invitation(
            project_id,
            token=body.token,
            principal_email=principal.email if principal is not None else None,
            principal_subject=principal.subject if principal is not None else None,
            principal_name=principal.name if principal is not None else None,
            display_name=body.display_name or (principal.name if principal is not None else None),
        )
    except ValueError as exc:
        detail = str(exc)
        if "Authentication is required" in detail:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)
        _raise_project_error(exc)


@router.post("/{project_id}/invitations/{invitation_id}/resend")
def resend_project_invitation(
    project_id: UUID,
    invitation_id: UUID,
    session: SessionDep,
    principal: PrincipalDep,
):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        return ProjectCollaborationService(session).resend_invitation(
            project_id,
            invitation_id,
            invited_by_subject=principal.subject if principal is not None else None,
            invited_by_email=principal.email if principal is not None else None,
            invited_by_name=principal.name if principal is not None else None,
        )
    except ValueError as exc:
        _raise_project_error(exc)


@router.post("/{project_id}/invitations/{invitation_id}/revoke")
def revoke_project_invitation(
    project_id: UUID,
    invitation_id: UUID,
    session: SessionDep,
    principal: PrincipalDep,
):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        return ProjectCollaborationService(session).revoke_invitation(project_id, invitation_id)
    except ValueError as exc:
        _raise_project_error(exc)


@router.patch("/{project_id}/collaborators/{collaborator_id}")
def update_project_collaborator(
    project_id: UUID,
    collaborator_id: UUID,
    body: ProjectCollaboratorUpdateRequest,
    session: SessionDep,
    principal: PrincipalDep,
):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        return ProjectCollaborationService(session).update_collaborator(
            project_id,
            collaborator_id,
            display_name=body.display_name,
            role=body.role,
        )
    except ValueError as exc:
        _raise_project_error(exc)


@router.delete("/{project_id}/collaborators/{collaborator_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project_collaborator(
    project_id: UUID,
    collaborator_id: UUID,
    session: SessionDep,
    principal: PrincipalDep,
):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        ProjectCollaborationService(session).delete_collaborator(project_id, collaborator_id)
    except ValueError as exc:
        _raise_project_error(exc)


@router.get("/{project_id}/notification-hooks")
def list_project_notification_hooks(project_id: UUID, session: SessionDep, principal: PrincipalDep):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        return ProjectCollaborationService(session).list_hooks(project_id)
    except ValueError as exc:
        _raise_project_error(exc)


@router.post("/{project_id}/notification-hooks", status_code=status.HTTP_201_CREATED)
def create_project_notification_hook(
    project_id: UUID,
    body: ProjectNotificationHookRequest,
    session: SessionDep,
    principal: PrincipalDep,
):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        return ProjectCollaborationService(session).create_hook(
            project_id,
            kind=body.kind,
            target=body.target,
            enabled=body.enabled,
            events=body.events,
            secret=body.secret,
            config=body.config,
        )
    except ValueError as exc:
        _raise_project_error(exc)


@router.patch("/{project_id}/notification-hooks/{hook_id}")
def update_project_notification_hook(
    project_id: UUID,
    hook_id: UUID,
    body: ProjectNotificationHookUpdateRequest,
    session: SessionDep,
    principal: PrincipalDep,
):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        return ProjectCollaborationService(session).update_hook(
            project_id,
            hook_id,
            kind=body.kind,
            target=body.target,
            enabled=body.enabled,
            events=body.events,
            secret=body.secret,
            config=body.config,
        )
    except ValueError as exc:
        _raise_project_error(exc)


@router.delete("/{project_id}/notification-hooks/{hook_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project_notification_hook(
    project_id: UUID,
    hook_id: UUID,
    session: SessionDep,
    principal: PrincipalDep,
):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        ProjectCollaborationService(session).delete_hook(project_id, hook_id)
    except ValueError as exc:
        _raise_project_error(exc)


@router.post("/{project_id}/notification-hooks/{hook_id}/test")
async def test_project_notification_hook(
    project_id: UUID,
    hook_id: UUID,
    session: SessionDep,
    principal: PrincipalDep,
):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        return await ProjectCollaborationService(session).test_hook(project_id, hook_id)
    except ValueError as exc:
        _raise_project_error(exc)


@router.get("/{project_id}/notification-deliveries")
def list_project_notification_deliveries(
    project_id: UUID,
    session: SessionDep,
    principal: PrincipalDep,
    limit: int = Query(default=50, ge=1, le=200),
):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        return ProjectCollaborationService(session).list_deliveries(project_id, limit=limit)
    except ValueError as exc:
        _raise_project_error(exc)


@router.post("/{project_id}/notification-deliveries/{delivery_id}/retry")
def retry_project_notification_delivery(
    project_id: UUID,
    delivery_id: UUID,
    session: SessionDep,
    principal: PrincipalDep,
):
    _require_project_capability_if_owned(session, project_id, principal, "manage")
    try:
        return ProjectCollaborationService(session).retry_delivery(project_id, delivery_id)
    except ValueError as exc:
        _raise_project_error(exc)
