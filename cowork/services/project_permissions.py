from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlmodel import Session, select

from cowork.models.project_collaboration import ProjectCollaborator
from cowork.services.project_collaboration import normalize_email


ROLE_ORDER = {
    "viewer": 10,
    "commenter": 20,
    "reviewer": 30,
    "editor": 40,
    "owner": 50,
}
ROLE_ALIASES = {
    "admin": "owner",
}
CAPABILITY_ROLES = {
    "view": "viewer",
    "comment": "commenter",
    "review": "reviewer",
    "edit": "editor",
    "manage": "owner",
}


@dataclass(frozen=True)
class ProjectPrincipal:
    email: str
    role: str


class ProjectPermissionError(PermissionError):
    def __init__(self, capability: str, *, project_id: UUID, actor_email: str | None = None) -> None:
        self.capability = capability
        self.project_id = project_id
        self.actor_email = actor_email
        super().__init__(f"Project permission '{capability}' is required")


def normalize_role(role: str | None) -> str:
    clean = (role or "viewer").strip().lower()
    clean = ROLE_ALIASES.get(clean, clean)
    return clean if clean in ROLE_ORDER else "viewer"


def role_allows(role: str | None, capability: str) -> bool:
    required_role = CAPABILITY_ROLES.get((capability or "").strip().lower())
    if required_role is None:
        return False
    return ROLE_ORDER[normalize_role(role)] >= ROLE_ORDER[required_role]


def get_project_principal(
    session: Session,
    project_id: UUID,
    actor_email: str | None,
) -> ProjectPrincipal | None:
    if not actor_email:
        return None
    email = normalize_email(actor_email)
    row = session.exec(
        select(ProjectCollaborator)
        .where(ProjectCollaborator.project_id == project_id)
        .where(ProjectCollaborator.email == email)
    ).first()
    if row is None:
        return None
    return ProjectPrincipal(email=row.email, role=normalize_role(row.role))


def project_has_owner(session: Session, project_id: UUID) -> bool:
    rows = session.exec(
        select(ProjectCollaborator.role)
        .where(ProjectCollaborator.project_id == project_id)
    ).all()
    return any(normalize_role(role) == "owner" for role in rows)


def has_project_permission(
    session: Session,
    project_id: UUID,
    actor_email: str | None,
    capability: str,
) -> bool:
    principal = get_project_principal(session, project_id, actor_email)
    return bool(principal and role_allows(principal.role, capability))


def require_project_permission(
    session: Session,
    project_id: UUID,
    actor_email: str | None,
    capability: str,
) -> ProjectPrincipal:
    principal = get_project_principal(session, project_id, actor_email)
    if principal is None or not role_allows(principal.role, capability):
        raise ProjectPermissionError(capability, project_id=project_id, actor_email=actor_email)
    return principal


def require_project_permission_if_owned(
    session: Session,
    project_id: UUID,
    actor_email: str | None,
    capability: str,
) -> ProjectPrincipal | None:
    if not project_has_owner(session, project_id):
        return None
    return require_project_permission(session, project_id, actor_email, capability)
