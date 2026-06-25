"""Identity and per-artifact share models.

Mirrors the conventions in ``cowork/models/project_collaboration.py``:
``BaseSQLModel`` (UUID pk + ``created_at`` / ``modified_at``), string
``role`` / ``status`` columns with the same roles ladder, a SHA-256
``token_hash`` returned only once at creation, and timezone-aware
lifecycle timestamps.

``User`` is the identity a lightweight (Google-Drive style) signup
creates so reviewers stop being "Someone". ``ArtifactShare`` is a
PER-ARTIFACT grant (the project-level analogue is
``ProjectInvitation`` / ``ProjectCollaborator``); accepting a share
upserts a ``User`` and flips the grant ``pending`` -> ``accepted``.

TODO(keycloak): ``sso_subject`` is the join key to a real Keycloak/SSO
    identity. Today a lightweight signup leaves it ``None``; wire it up
    when bearer auth (see ``cowork.services.request_identity``) is the
    source of truth for accepts.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Field, Relationship

from cowork.models.artifact import Artifact
from cowork.models.base import BaseSQLModel


class User(BaseSQLModel, table=True):
    """A lightweight-signup or SSO identity.

    A guest who completes the lightweight signup becomes a ``User``;
    no separate guest-session table is needed. ``sso_subject`` links to
    a Keycloak/SSO subject once real auth is wired (see module TODO).
    """

    __tablename__ = "users"
    __table_args__ = (
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    email: str = Field(max_length=255, index=True, description="Normalized, unique account email")
    display_name: str | None = Field(default=None, max_length=255, description="Human-facing name")
    sso_subject: str | None = Field(
        default=None,
        max_length=255,
        index=True,
        description="Keycloak/SSO subject identifier; null for lightweight-signup guests",
    )


class ArtifactShare(BaseSQLModel, table=True):
    """A per-artifact access grant.

    The per-artifact analogue of ``ProjectInvitation``: an owner shares
    a single artifact with ``grantee_email`` at ``role``; the raw accept
    token is returned once and stored only as ``token_hash`` (SHA-256).
    Accepting upserts a ``User`` and flips ``status`` to ``accepted``.
    """

    __tablename__ = "artifact_shares"

    artifact_id: UUID = Field(foreign_key="artifacts.id", index=True, description="Artifact this share grants access to")
    grantee_email: str = Field(max_length=255, index=True, description="Normalized invitee email")
    role: str = Field(default="viewer", max_length=64, description="viewer | commenter | reviewer | editor")
    status: str = Field(default="pending", max_length=64, index=True, description="pending | accepted | revoked | expired")
    token_hash: str = Field(max_length=64, index=True, description="SHA-256 hash of the current accept token")
    created_by: str | None = Field(default=None, max_length=255, description="Email or subject of the sharer")
    expires_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True), description="When this share expires")
    accepted_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True), description="When this share was accepted")
    revoked_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True), description="When this share was revoked")
    accepted_user_id: UUID | None = Field(
        default=None,
        foreign_key="users.id",
        index=True,
        description="User who accepted this share (set on accept)",
    )

    artifact: Artifact = Relationship()
    # NOTE: the User who accepted is reachable via ``accepted_user_id``;
    # an optional ORM relationship is intentionally omitted because the
    # ``User | None`` union annotation is not resolvable by SQLAlchemy's
    # string-based mapper lookup. Add one later with an explicit
    # ``Mapped[...]``/back_populates if navigation is needed.
