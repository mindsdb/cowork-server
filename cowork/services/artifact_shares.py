"""Per-artifact share service.

Mirrors ``cowork/services/project_collaboration.py`` for a single
artifact rather than a project: invite -> hashed token returned once;
accept matches the grantee email, upserts a ``User`` (the lightweight
signup), and flips the matching grant ``pending`` -> ``accepted``.

The lightweight-signup gate: viewing an artifact stays open, but
commenting/reviewing requires an accepted share (a ``User`` identity),
so reviewers are named rather than "Someone".

TODO(keycloak/sso): ``accept_share`` trusts the supplied ``email`` /
    ``display_name``. Once bearer auth is the source of truth, derive
    these from the verified ``RequestPrincipal``
    (``cowork.services.request_identity``) and stamp
    ``User.sso_subject`` from the token ``sub``.
TODO(enforcement): wire these grants into the per-artifact capability
    checks in ``cowork/api/v1/endpoints/artifacts.py`` so an accepted
    ``commenter``/``reviewer``/``editor`` share actually authorizes the
    corresponding artifact action.
"""
from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlmodel import Session, select

from cowork.models.artifact import Artifact
from cowork.models.identity import ArtifactShare, User


# Per-artifact roles ladder (subset of the project COLLABORATOR_ROLES;
# "owner" is a project-level concept and intentionally excluded here).
SHARE_ROLES = {"viewer", "commenter", "reviewer", "editor"}

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(email: str) -> str:
    clean = (email or "").strip().lower()
    if not clean or not _EMAIL_RE.match(clean):
        raise ValueError("A valid grantee email is required")
    return clean


def normalize_optional_email(email: str | None) -> str | None:
    if not email:
        return None
    return normalize_email(email)


def validate_role(role: str) -> str:
    clean = (role or "viewer").strip().lower()
    if clean not in SHARE_ROLES:
        raise ValueError(f"Share role must be one of: {', '.join(sorted(SHARE_ROLES))}")
    return clean


def new_share_token() -> str:
    return secrets.token_urlsafe(32)


def share_token_hash(token: str) -> str:
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def share_expired(row: ArtifactShare) -> bool:
    if row.expires_at is None:
        return False
    expires = row.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return expires <= datetime.now(timezone.utc)


def _artifact(session: Session, artifact_id: UUID) -> Artifact:
    artifact = session.get(Artifact, artifact_id)
    if artifact is None:
        raise ValueError("Artifact not found")
    return artifact


def create_share(
    session: Session,
    *,
    artifact_id: UUID,
    grantee_email: str,
    role: str = "viewer",
    created_by: str | None = None,
    ttl_days: int = 7,
) -> dict:
    """Create or refresh a pending per-artifact share.

    Returns the grant plus the raw accept token, which is exposed only
    here (stored at rest as ``token_hash``), mirroring project invites.
    """
    artifact = _artifact(session, artifact_id)
    normalized_email = normalize_email(grantee_email)
    clean_role = validate_role(role)
    token = new_share_token()
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=max(1, min(int(ttl_days or 7), 30)))
    row = session.exec(
        select(ArtifactShare)
        .where(ArtifactShare.artifact_id == artifact.id)
        .where(ArtifactShare.grantee_email == normalized_email)
        .where(ArtifactShare.status == "pending")
    ).first()
    created = row is None
    if row is None:
        row = ArtifactShare(artifact_id=artifact.id, grantee_email=normalized_email)
    row.role = clean_role
    row.status = "pending"
    row.token_hash = share_token_hash(token)
    row.expires_at = expires_at
    row.accepted_at = None
    row.revoked_at = None
    row.accepted_user_id = None
    row.created_by = created_by or row.created_by
    session.add(row)
    session.commit()
    session.refresh(row)
    return {"created": created, "share": share_to_dict(row, token=token)}


def list_shares(session: Session, artifact_id: UUID, *, include_closed: bool = True) -> dict:
    artifact = _artifact(session, artifact_id)
    query = select(ArtifactShare).where(ArtifactShare.artifact_id == artifact.id)
    if not include_closed:
        query = query.where(ArtifactShare.status == "pending")
    rows = session.exec(query.order_by(ArtifactShare.created_at.desc())).all()
    return {
        "artifactId": str(artifact.id),
        "shares": [share_to_dict(row) for row in rows],
    }


def accept_share(
    session: Session,
    *,
    token: str,
    email: str,
    display_name: str | None = None,
) -> dict:
    """Accept a share: upsert the ``User`` and flip the grant accepted.

    This is the lightweight-signup gate. The accepting ``email`` must
    match the grant's ``grantee_email``.
    """
    accept_email = normalize_email(email)
    row = session.exec(
        select(ArtifactShare)
        .where(ArtifactShare.token_hash == share_token_hash(token))
        .where(ArtifactShare.status == "pending")
    ).first()
    if row is None:
        raise ValueError("Share not found")
    if row.grantee_email != accept_email:
        raise ValueError("Share can only be accepted by the invited email")
    if share_expired(row):
        row.status = "expired"
        session.add(row)
        session.commit()
        raise ValueError("Share has expired")

    user = session.exec(select(User).where(User.email == accept_email)).first()
    user_created = user is None
    if user is None:
        user = User(email=accept_email)
    user.display_name = display_name or user.display_name
    session.add(user)
    session.flush()

    row.status = "accepted"
    row.accepted_at = datetime.now(timezone.utc)
    row.accepted_user_id = user.id
    session.add(row)
    session.commit()
    session.refresh(row)
    session.refresh(user)
    return {
        "userCreated": user_created,
        "share": share_to_dict(row),
        "user": user_to_dict(user),
    }


def set_share_role(session: Session, share_id: UUID, role: str) -> dict:
    row = session.get(ArtifactShare, share_id)
    if row is None:
        raise ValueError("Share not found")
    row.role = validate_role(role)
    session.add(row)
    session.commit()
    session.refresh(row)
    return {"share": share_to_dict(row)}


def revoke_share(session: Session, share_id: UUID) -> dict:
    row = session.get(ArtifactShare, share_id)
    if row is None:
        raise ValueError("Share not found")
    if row.status != "pending":
        raise ValueError("Only pending shares can be revoked")
    row.status = "revoked"
    row.revoked_at = datetime.now(timezone.utc)
    session.add(row)
    session.commit()
    session.refresh(row)
    return {"share": share_to_dict(row)}


def share_to_dict(row: ArtifactShare, *, token: str | None = None) -> dict:
    payload = {
        "id": str(row.id),
        "artifactId": str(row.artifact_id),
        "granteeEmail": row.grantee_email,
        "role": row.role,
        "status": row.status,
        "createdBy": row.created_by,
        "expiresAt": row.expires_at.isoformat() if row.expires_at else None,
        "acceptedAt": row.accepted_at.isoformat() if row.accepted_at else None,
        "revokedAt": row.revoked_at.isoformat() if row.revoked_at else None,
        "acceptedUserId": str(row.accepted_user_id) if row.accepted_user_id else None,
        "createdAt": row.created_at.isoformat() if row.created_at else None,
        "modifiedAt": row.modified_at.isoformat() if row.modified_at else None,
    }
    if token:
        payload["acceptToken"] = token
    return payload


def user_to_dict(row: User) -> dict:
    return {
        "id": str(row.id),
        "email": row.email,
        "displayName": row.display_name,
        "ssoSubject": row.sso_subject,
        "createdAt": row.created_at.isoformat() if row.created_at else None,
        "modifiedAt": row.modified_at.isoformat() if row.modified_at else None,
    }
