from typing import TYPE_CHECKING, Any
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import JSON
from sqlmodel import Column, Field, Relationship

from cowork.models.base import BaseSQLModel

if TYPE_CHECKING:
    from cowork.models.project import Project


class Artifact(BaseSQLModel, table=True):
    __tablename__ = "artifacts"
    __table_args__ = (
        sa.UniqueConstraint("path", name="uq_artifacts_path"),
        sa.UniqueConstraint("project_id", "slug", name="uq_artifacts_project_slug"),
    )

    project_id: UUID | None = Field(
        default=None,
        foreign_key="projects.id",
        index=True,
        description="Project that owns this artifact, when known",
    )
    slug: str = Field(index=True, max_length=255, description="Stable folder slug for the artifact")
    title: str = Field(max_length=255, description="Human-facing artifact title")
    description: str | None = Field(default=None, sa_type=sa.Text, description="Artifact summary")
    artifact_type: str | None = Field(default=None, max_length=64, description="Artifact type from metadata.json")
    path: str = Field(max_length=2048, description="Absolute path to the artifact folder")
    current_version_id: UUID | None = Field(
        default=None,
        index=True,
        description="Most recent artifact_versions.id known to the server",
    )
    last_known_good_version_id: UUID | None = Field(
        default=None,
        index=True,
        description="Last version known to preview or publish successfully",
    )

    project: "Project" = Relationship()
    versions: list["ArtifactVersion"] = Relationship(back_populates="artifact")
    drafts: list["ArtifactDraft"] = Relationship(back_populates="artifact")
    deployments: list["ArtifactDeployment"] = Relationship(back_populates="artifact")
    comments: list["ArtifactComment"] = Relationship(back_populates="artifact")
    activity_events: list["ArtifactActivityEvent"] = Relationship(back_populates="artifact")



class ArtifactVersion(BaseSQLModel, table=True):
    __tablename__ = "artifact_versions"
    __table_args__ = (
        sa.UniqueConstraint("artifact_id", "version_number", name="uq_artifact_versions_number"),
    )

    artifact_id: UUID = Field(foreign_key="artifacts.id", index=True, description="Artifact this version belongs to")
    parent_version_id: UUID | None = Field(
        default=None,
        foreign_key="artifact_versions.id",
        index=True,
        description="Previous version in the artifact history graph",
    )
    version_number: int = Field(description="Monotonic version number within an artifact")
    label: str | None = Field(default=None, max_length=255, description="Human-facing checkpoint label")
    manifest_hash: str = Field(max_length=64, index=True, description="SHA-256 hash of the canonical manifest")
    files_hash: str = Field(max_length=64, index=True, description="SHA-256 hash of the file tree")
    file_count: int = Field(default=0, description="Number of files captured")
    total_bytes: int = Field(default=0, description="Total captured byte count")
    store_path: str = Field(max_length=2048, description="Artifact-store relative path to this version manifest")
    source_conversation_id: UUID | None = Field(
        default=None,
        foreign_key="conversations.id",
        index=True,
        description="Conversation that produced this version, when known",
    )
    source_message_id: UUID | None = Field(
        default=None,
        foreign_key="messages.id",
        index=True,
        description="Message that produced this version, when known",
    )
    prompt: str | None = Field(default=None, sa_type=sa.Text, description="Prompt or instruction that produced it")
    operation_type: str = Field(default="snapshot", max_length=64, description="snapshot | edit | restore | import")
    snapshot_role: str | None = Field(
        default=None,
        max_length=64,
        index=True,
        description="Role in a paired snapshot flow, such as pre | post | single",
    )
    pre_snapshot_version_id: UUID | None = Field(
        default=None,
        foreign_key="artifact_versions.id",
        index=True,
        description="Pre-change snapshot paired with this version, when applicable",
    )
    preview_status: str = Field(default="pending", max_length=64, description="Preview lifecycle status")
    publish_status: str = Field(default="unpublished", max_length=64, description="Publish lifecycle status")
    restored_from_version_id: UUID | None = Field(
        default=None,
        foreign_key="artifact_versions.id",
        index=True,
        description="Source version when this row was created by restore",
    )
    branch_name: str | None = Field(default=None, max_length=255, description="Optional branch or remix name")
    forked_from_version_id: UUID | None = Field(
        default=None,
        foreign_key="artifact_versions.id",
        index=True,
        description="Source version when this version starts a fork/remix",
    )

    artifact: Artifact = Relationship(back_populates="versions")
    files: list["ArtifactVersionFile"] = Relationship(back_populates="version")
    drafts: list["ArtifactDraft"] = Relationship(back_populates="base_version")
    deployments: list["ArtifactDeployment"] = Relationship(back_populates="version")


class ArtifactVersionFile(BaseSQLModel, table=True):
    __tablename__ = "artifact_version_files"
    __table_args__ = (
        sa.UniqueConstraint("version_id", "path", name="uq_artifact_version_files_path"),
    )

    version_id: UUID = Field(foreign_key="artifact_versions.id", index=True, description="Captured version")
    path: str = Field(max_length=2048, description="POSIX relative path within the artifact")
    content_hash: str = Field(max_length=64, index=True, description="SHA-256 content digest")
    size: int = Field(description="File size in bytes")
    blob_path: str = Field(max_length=2048, description="Artifact-store relative path to the content blob")

    version: ArtifactVersion = Relationship(back_populates="files")


class ArtifactDraft(BaseSQLModel, table=True):
    __tablename__ = "artifact_drafts"

    artifact_id: UUID = Field(foreign_key="artifacts.id", index=True, description="Artifact being drafted")
    base_version_id: UUID | None = Field(
        default=None,
        foreign_key="artifact_versions.id",
        index=True,
        description="Version the draft started from",
    )
    draft_path: str = Field(max_length=2048, description="Working directory for the draft")
    status: str = Field(default="open", max_length=64, description="open | applied | abandoned")
    details: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    artifact: Artifact = Relationship(back_populates="drafts")
    base_version: ArtifactVersion = Relationship(back_populates="drafts")


class ArtifactDeployment(BaseSQLModel, table=True):
    __tablename__ = "artifact_deployments"

    artifact_id: UUID = Field(foreign_key="artifacts.id", index=True, description="Artifact being deployed")
    version_id: UUID | None = Field(
        default=None,
        foreign_key="artifact_versions.id",
        index=True,
        description="Version deployed, when pinned",
    )
    target: str = Field(max_length=128, description="Deployment target, such as preview or publish")
    status: str = Field(default="unknown", max_length=64, description="Deployment lifecycle status")
    url: str | None = Field(default=None, max_length=2048, description="External URL, when one exists")
    details: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    artifact: Artifact = Relationship(back_populates="deployments")
    version: ArtifactVersion = Relationship(back_populates="deployments")


class ArtifactComment(BaseSQLModel, table=True):
    __tablename__ = "artifact_comments"

    artifact_id: UUID = Field(foreign_key="artifacts.id", index=True, description="Artifact being discussed")
    version_id: UUID | None = Field(
        default=None,
        foreign_key="artifact_versions.id",
        index=True,
        description="Version the comment was anchored to, when known",
    )
    parent_comment_id: UUID | None = Field(
        default=None,
        foreign_key="artifact_comments.id",
        index=True,
        description="Parent comment for a thread reply",
    )
    kind: str = Field(default="comment", max_length=64, description="comment | suggestion | review")
    body: str = Field(sa_type=sa.Text, description="Comment or suggested-change text")
    anchor: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    proposed_patch: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = Field(default="open", max_length=64, description="open | resolved | accepted | rejected")
    review_verdict: str | None = Field(
        default=None,
        max_length=64,
        index=True,
        description="Review verdict such as approved | changes_requested",
    )
    actor_name: str | None = Field(default=None, max_length=255, description="Display name for the commenter")
    notification_state: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    artifact: Artifact = Relationship(back_populates="comments")


class ArtifactActivityEvent(BaseSQLModel, table=True):
    __tablename__ = "artifact_activity_events"

    artifact_id: UUID = Field(foreign_key="artifacts.id", index=True, description="Artifact this event belongs to")
    version_id: UUID | None = Field(
        default=None,
        foreign_key="artifact_versions.id",
        index=True,
        description="Related version, when the event is version-specific",
    )
    event_type: str = Field(max_length=128, description="commented | suggested | resolved | published | restored")
    actor_name: str | None = Field(default=None, max_length=255, description="Display name for the actor")
    details: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    artifact: Artifact = Relationship(back_populates="activity_events")
