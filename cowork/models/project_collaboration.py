from datetime import datetime
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import JSON
from sqlmodel import Column, Field, Relationship

from cowork.models.base import BaseSQLModel
from cowork.models.project import Project


class ProjectCollaborator(BaseSQLModel, table=True):
    __tablename__ = "project_collaborators"
    __table_args__ = (
        sa.UniqueConstraint("project_id", "email", name="uq_project_collaborators_email"),
    )

    project_id: UUID = Field(foreign_key="projects.id", index=True, description="Project this collaborator belongs to")
    email: str = Field(max_length=255, index=True, description="Normalized collaborator email")
    display_name: str | None = Field(default=None, max_length=255, description="Human-facing collaborator name")
    role: str = Field(default="viewer", max_length=64, description="owner | editor | reviewer | commenter | viewer")
    notification_state: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    project: Project = Relationship()


class ProjectInvitation(BaseSQLModel, table=True):
    __tablename__ = "project_invitations"

    project_id: UUID = Field(foreign_key="projects.id", index=True, description="Project this invitation grants access to")
    email: str = Field(max_length=255, index=True, description="Normalized invitee email")
    display_name: str | None = Field(default=None, max_length=255, description="Human-facing invitee name")
    role: str = Field(default="viewer", max_length=64, description="Role to grant on accept")
    status: str = Field(default="pending", max_length=64, index=True, description="pending | accepted | revoked | expired")
    token_hash: str = Field(max_length=64, index=True, description="SHA-256 hash of the current accept token")
    expires_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True), description="When this invitation expires")
    accepted_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True), description="When this invitation was accepted")
    revoked_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True), description="When this invitation was revoked")
    last_sent_at: datetime | None = Field(default=None, sa_type=sa.DateTime(timezone=True), description="When this invitation was last emailed")
    send_count: int = Field(default=0, description="Number of invitation emails queued")
    invited_by_subject: str | None = Field(default=None, max_length=255, description="Subject identifier of the inviter")
    invited_by_email: str | None = Field(default=None, max_length=255, description="Email of the inviter")
    invited_by_name: str | None = Field(default=None, max_length=255, description="Display name of the inviter")
    accepted_by_subject: str | None = Field(default=None, max_length=255, description="Subject identifier of the accepting principal")
    accepted_by_email: str | None = Field(default=None, max_length=255, description="Email of the accepting principal")
    accepted_by_name: str | None = Field(default=None, max_length=255, description="Display name of the accepting principal")
    notification_state: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    project: Project = Relationship()


class ProjectNotificationHook(BaseSQLModel, table=True):
    __tablename__ = "project_notification_hooks"

    project_id: UUID = Field(foreign_key="projects.id", index=True, description="Project this hook belongs to")
    kind: str = Field(default="email", max_length=64, description="email | webhook")
    target: str = Field(max_length=1024, description="Non-secret display target, such as team@example.com or Review webhook")
    enabled: bool = Field(default=True, description="Whether this hook receives project notifications")
    events: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    secret_ciphertext: str | None = Field(default=None, sa_type=sa.Text, description="Encrypted hook secret")
    config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    project: Project = Relationship()
    deliveries: list["NotificationDelivery"] = Relationship(back_populates="hook")


class NotificationDelivery(BaseSQLModel, table=True):
    __tablename__ = "notification_deliveries"
    __table_args__ = (
        sa.UniqueConstraint("hook_id", "dedupe_key", name="uq_notification_deliveries_hook_dedupe"),
    )

    project_id: UUID = Field(foreign_key="projects.id", index=True, description="Project that emitted the notification")
    hook_id: UUID | None = Field(
        default=None,
        foreign_key="project_notification_hooks.id",
        index=True,
        description="Notification hook used for delivery",
    )
    event_key: str = Field(max_length=128, index=True, description="Normalized notification event key")
    dedupe_key: str = Field(max_length=255, index=True, description="Stable event+target key")
    status: str = Field(default="queued", max_length=64, description="queued | sending | sent | skipped | failed | exhausted")
    attempts: int = Field(default=0, description="Send attempts made")
    error: str | None = Field(default=None, sa_type=sa.Text, description="Latest delivery error")
    details: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    project: Project = Relationship()
    hook: ProjectNotificationHook | None = Relationship(back_populates="deliveries")
