"""artifact versions

Revision ID: d3a5c9e7b1f2
Revises: c4e7a1b9d2f0
Create Date: 2026-06-19 23:59:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d3a5c9e7b1f2"
down_revision: Union[str, Sequence[str], None] = "c4e7a1b9d2f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _columns(table_name: str) -> set[str]:
    if not _has_table(table_name):
        return set()
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def _has_index(table_name: str, index_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return index_name in {index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table_name)}


def _copy_column(table_name: str, target_column: str, source_column: str) -> None:
    cols = _columns(table_name)
    if target_column not in cols or source_column not in cols:
        return
    op.execute(
        sa.text(
            f"UPDATE {table_name} "
            f"SET {target_column} = {source_column} "
            f"WHERE {target_column} IS NULL AND {source_column} IS NOT NULL"
        )
    )


def _add_base_timestamps(batch_op, cols: set[str]) -> None:
    if "created_at" not in cols:
        batch_op.add_column(
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True)
        )
    if "modified_at" not in cols:
        batch_op.add_column(
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True)
        )


def _create_missing_artifact_child_tables() -> None:
    if not _has_table("artifact_versions"):
        op.create_table(
            "artifact_versions",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("artifact_id", sa.Uuid(), nullable=False),
            sa.Column("parent_version_id", sa.Uuid(), nullable=True),
            sa.Column("version_number", sa.Integer(), nullable=False),
            sa.Column("label", sa.String(255), nullable=True),
            sa.Column("manifest_hash", sa.String(64), nullable=False),
            sa.Column("files_hash", sa.String(64), nullable=False),
            sa.Column("file_count", sa.Integer(), nullable=False),
            sa.Column("total_bytes", sa.Integer(), nullable=False),
            sa.Column("store_path", sa.String(2048), nullable=False),
            sa.Column("source_conversation_id", sa.Uuid(), nullable=True),
            sa.Column("source_message_id", sa.Uuid(), nullable=True),
            sa.Column("prompt", sa.Text(), nullable=True),
            sa.Column("operation_type", sa.String(64), nullable=False),
            sa.Column("preview_status", sa.String(64), nullable=False),
            sa.Column("publish_status", sa.String(64), nullable=False),
            sa.Column("restored_from_version_id", sa.Uuid(), nullable=True),
            sa.Column("branch_name", sa.String(255), nullable=True),
            sa.Column("forked_from_version_id", sa.Uuid(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
            sa.ForeignKeyConstraint(["parent_version_id"], ["artifact_versions.id"]),
            sa.ForeignKeyConstraint(["restored_from_version_id"], ["artifact_versions.id"]),
            sa.ForeignKeyConstraint(["forked_from_version_id"], ["artifact_versions.id"]),
            sa.ForeignKeyConstraint(["source_conversation_id"], ["conversations.id"]),
            sa.ForeignKeyConstraint(["source_message_id"], ["messages.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("artifact_id", "version_number", name="uq_artifact_versions_number"),
        )
        op.create_index(op.f("ix_artifact_versions_artifact_id"), "artifact_versions", ["artifact_id"], unique=False)
        op.create_index(op.f("ix_artifact_versions_files_hash"), "artifact_versions", ["files_hash"], unique=False)
        op.create_index(op.f("ix_artifact_versions_parent_version_id"), "artifact_versions", ["parent_version_id"], unique=False)
        op.create_index(op.f("ix_artifact_versions_manifest_hash"), "artifact_versions", ["manifest_hash"], unique=False)
        op.create_index(
            op.f("ix_artifact_versions_restored_from_version_id"),
            "artifact_versions",
            ["restored_from_version_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_artifact_versions_source_conversation_id"),
            "artifact_versions",
            ["source_conversation_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_artifact_versions_source_message_id"),
            "artifact_versions",
            ["source_message_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_artifact_versions_forked_from_version_id"),
            "artifact_versions",
            ["forked_from_version_id"],
            unique=False,
        )

    if not _has_table("artifact_version_files"):
        op.create_table(
            "artifact_version_files",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("version_id", sa.Uuid(), nullable=False),
            sa.Column("path", sa.String(2048), nullable=False),
            sa.Column("content_hash", sa.String(64), nullable=False),
            sa.Column("size", sa.Integer(), nullable=False),
            sa.Column("blob_path", sa.String(2048), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.ForeignKeyConstraint(["version_id"], ["artifact_versions.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("version_id", "path", name="uq_artifact_version_files_path"),
        )
        op.create_index(op.f("ix_artifact_version_files_content_hash"), "artifact_version_files", ["content_hash"], unique=False)
        op.create_index(op.f("ix_artifact_version_files_version_id"), "artifact_version_files", ["version_id"], unique=False)

    if not _has_table("artifact_drafts"):
        op.create_table(
            "artifact_drafts",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("artifact_id", sa.Uuid(), nullable=False),
            sa.Column("base_version_id", sa.Uuid(), nullable=True),
            sa.Column("draft_path", sa.String(2048), nullable=False),
            sa.Column("status", sa.String(64), nullable=False),
            sa.Column("details", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
            sa.ForeignKeyConstraint(["base_version_id"], ["artifact_versions.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_artifact_drafts_artifact_id"), "artifact_drafts", ["artifact_id"], unique=False)
        op.create_index(op.f("ix_artifact_drafts_base_version_id"), "artifact_drafts", ["base_version_id"], unique=False)

    if not _has_table("artifact_deployments"):
        op.create_table(
            "artifact_deployments",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("artifact_id", sa.Uuid(), nullable=False),
            sa.Column("version_id", sa.Uuid(), nullable=True),
            sa.Column("target", sa.String(128), nullable=False),
            sa.Column("status", sa.String(64), nullable=False),
            sa.Column("url", sa.String(2048), nullable=True),
            sa.Column("details", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
            sa.ForeignKeyConstraint(["version_id"], ["artifact_versions.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_artifact_deployments_artifact_id"), "artifact_deployments", ["artifact_id"], unique=False)
        op.create_index(op.f("ix_artifact_deployments_version_id"), "artifact_deployments", ["version_id"], unique=False)


def _upgrade_existing_artifact_tables() -> None:
    """Adapt an existing artifact schema from parallel endpoint work.

    The endpoint migration owns a richer table shape. This follow-on keeps its
    columns, adds the snapshot-service columns, and relaxes unmapped non-null
    fields so SQLModel inserts using this service do not fail.
    """
    artifact_cols = _columns("artifacts")
    with op.batch_alter_table("artifacts") as batch_op:
        _add_base_timestamps(batch_op, artifact_cols)
        if "project_id" not in artifact_cols:
            batch_op.add_column(sa.Column("project_id", sa.Uuid(), nullable=True))
        if "path" not in artifact_cols:
            batch_op.add_column(sa.Column("path", sa.String(2048), nullable=True))
        if "current_version_id" not in artifact_cols:
            batch_op.add_column(sa.Column("current_version_id", sa.Uuid(), nullable=True))
        if "description" not in artifact_cols:
            batch_op.add_column(sa.Column("description", sa.Text(), nullable=True))
        if "artifact_type" not in artifact_cols:
            batch_op.add_column(sa.Column("artifact_type", sa.String(64), nullable=True))
        if "last_known_good_version_id" not in artifact_cols:
            batch_op.add_column(sa.Column("last_known_good_version_id", sa.Uuid(), nullable=True))
        if "project_id" in artifact_cols:
            batch_op.alter_column("project_id", existing_type=sa.Uuid(), nullable=True)
        if "artifact_type" in artifact_cols:
            batch_op.alter_column("artifact_type", existing_type=sa.String(64), nullable=True)
        if "folder_path" in artifact_cols:
            batch_op.alter_column("folder_path", existing_type=sa.String(2048), nullable=True)
        if "deleted" in artifact_cols:
            batch_op.alter_column("deleted", existing_type=sa.Boolean(), nullable=True)
    _copy_column("artifacts", "path", "folder_path")
    if not _has_index("artifacts", "uq_artifacts_path"):
        op.create_index("uq_artifacts_path", "artifacts", ["path"], unique=True)
    if not _has_index("artifacts", op.f("ix_artifacts_slug")):
        op.create_index(op.f("ix_artifacts_slug"), "artifacts", ["slug"], unique=False)
    if not _has_index("artifacts", op.f("ix_artifacts_project_id")):
        op.create_index(op.f("ix_artifacts_project_id"), "artifacts", ["project_id"], unique=False)
    if not _has_index("artifacts", op.f("ix_artifacts_current_version_id")):
        op.create_index(op.f("ix_artifacts_current_version_id"), "artifacts", ["current_version_id"], unique=False)
    if not _has_index("artifacts", op.f("ix_artifacts_last_known_good_version_id")):
        op.create_index(
            op.f("ix_artifacts_last_known_good_version_id"),
            "artifacts",
            ["last_known_good_version_id"],
            unique=False,
        )

    _create_missing_artifact_child_tables()

    version_cols = _columns("artifact_versions")
    with op.batch_alter_table("artifact_versions") as batch_op:
        _add_base_timestamps(batch_op, version_cols)
        if "parent_version_id" not in version_cols:
            batch_op.add_column(sa.Column("parent_version_id", sa.Uuid(), nullable=True))
        if "version_number" not in version_cols:
            batch_op.add_column(sa.Column("version_number", sa.Integer(), nullable=True))
        if "label" not in version_cols:
            batch_op.add_column(sa.Column("label", sa.String(255), nullable=True))
        if "file_count" not in version_cols:
            batch_op.add_column(sa.Column("file_count", sa.Integer(), nullable=True))
        if "total_bytes" not in version_cols:
            batch_op.add_column(sa.Column("total_bytes", sa.Integer(), nullable=True))
        if "store_path" not in version_cols:
            batch_op.add_column(sa.Column("store_path", sa.String(2048), nullable=True))
        if "restored_from_version_id" not in version_cols:
            batch_op.add_column(sa.Column("restored_from_version_id", sa.Uuid(), nullable=True))
        if "branch_name" not in version_cols:
            batch_op.add_column(sa.Column("branch_name", sa.String(255), nullable=True))
        if "forked_from_version_id" not in version_cols:
            batch_op.add_column(sa.Column("forked_from_version_id", sa.Uuid(), nullable=True))
        if "manifest_path" in version_cols:
            batch_op.alter_column("manifest_path", existing_type=sa.String(2048), nullable=True)
        if "status" in version_cols:
            batch_op.alter_column("status", existing_type=sa.String(32), nullable=True)
    _copy_column("artifact_versions", "store_path", "manifest_path")
    if not _has_index("artifact_versions", "uq_artifact_versions_number"):
        op.create_index(
            "uq_artifact_versions_number",
            "artifact_versions",
            ["artifact_id", "version_number"],
            unique=True,
        )
    if not _has_index("artifact_versions", op.f("ix_artifact_versions_artifact_id")):
        op.create_index(op.f("ix_artifact_versions_artifact_id"), "artifact_versions", ["artifact_id"], unique=False)
    if not _has_index("artifact_versions", op.f("ix_artifact_versions_parent_version_id")):
        op.create_index(
            op.f("ix_artifact_versions_parent_version_id"),
            "artifact_versions",
            ["parent_version_id"],
            unique=False,
        )
    if not _has_index("artifact_versions", op.f("ix_artifact_versions_restored_from_version_id")):
        op.create_index(
            op.f("ix_artifact_versions_restored_from_version_id"),
            "artifact_versions",
            ["restored_from_version_id"],
            unique=False,
        )
    if not _has_index("artifact_versions", op.f("ix_artifact_versions_forked_from_version_id")):
        op.create_index(
            op.f("ix_artifact_versions_forked_from_version_id"),
            "artifact_versions",
            ["forked_from_version_id"],
            unique=False,
        )
    if not _has_index("artifact_versions", op.f("ix_artifact_versions_source_conversation_id")):
        op.create_index(
            op.f("ix_artifact_versions_source_conversation_id"),
            "artifact_versions",
            ["source_conversation_id"],
            unique=False,
        )
    if not _has_index("artifact_versions", op.f("ix_artifact_versions_source_message_id")):
        op.create_index(
            op.f("ix_artifact_versions_source_message_id"),
            "artifact_versions",
            ["source_message_id"],
            unique=False,
        )

    file_cols = _columns("artifact_version_files")
    with op.batch_alter_table("artifact_version_files") as batch_op:
        _add_base_timestamps(batch_op, file_cols)
        if "content_hash" not in file_cols:
            batch_op.add_column(sa.Column("content_hash", sa.String(64), nullable=True))
        if "blob_path" not in file_cols:
            batch_op.add_column(sa.Column("blob_path", sa.String(2048), nullable=True))
        if "sha256" in file_cols:
            batch_op.alter_column("sha256", existing_type=sa.String(64), nullable=True)
        if "role" in file_cols:
            batch_op.alter_column("role", existing_type=sa.String(64), nullable=True)
        if "object_path" in file_cols:
            batch_op.alter_column("object_path", existing_type=sa.String(2048), nullable=True)
    _copy_column("artifact_version_files", "content_hash", "sha256")
    _copy_column("artifact_version_files", "blob_path", "object_path")
    if not _has_index("artifact_version_files", op.f("ix_artifact_version_files_content_hash")):
        op.create_index(
            op.f("ix_artifact_version_files_content_hash"),
            "artifact_version_files",
            ["content_hash"],
            unique=False,
        )

    draft_cols = _columns("artifact_drafts")
    with op.batch_alter_table("artifact_drafts") as batch_op:
        _add_base_timestamps(batch_op, draft_cols)
        if "draft_path" not in draft_cols:
            batch_op.add_column(sa.Column("draft_path", sa.String(2048), nullable=True))
        if "details" not in draft_cols:
            batch_op.add_column(sa.Column("details", sa.JSON(), nullable=True))
        if "draft_type" in draft_cols:
            batch_op.alter_column("draft_type", existing_type=sa.String(64), nullable=True)
    if not _has_index("artifact_drafts", op.f("ix_artifact_drafts_base_version_id")):
        op.create_index(op.f("ix_artifact_drafts_base_version_id"), "artifact_drafts", ["base_version_id"], unique=False)

    deployment_cols = _columns("artifact_deployments")
    with op.batch_alter_table("artifact_deployments") as batch_op:
        _add_base_timestamps(batch_op, deployment_cols)
        if "target" not in deployment_cols:
            batch_op.add_column(sa.Column("target", sa.String(128), nullable=True))
        if "details" not in deployment_cols:
            batch_op.add_column(sa.Column("details", sa.JSON(), nullable=True))
        if "version_id" in deployment_cols:
            batch_op.alter_column("version_id", existing_type=sa.Uuid(), nullable=True)
        if "url" in deployment_cols:
            batch_op.alter_column("url", existing_type=sa.String(2048), nullable=True)
        if "access_mode" in deployment_cols:
            batch_op.alter_column("access_mode", existing_type=sa.String(32), nullable=True)


def _create_collaboration_tables() -> None:
    if not _has_table("artifact_comments"):
        op.create_table(
            "artifact_comments",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("artifact_id", sa.Uuid(), nullable=False),
            sa.Column("version_id", sa.Uuid(), nullable=True),
            sa.Column("parent_comment_id", sa.Uuid(), nullable=True),
            sa.Column("kind", sa.String(64), nullable=False),
            sa.Column("body", sa.Text(), nullable=False),
            sa.Column("anchor", sa.JSON(), nullable=True),
            sa.Column("proposed_patch", sa.JSON(), nullable=True),
            sa.Column("status", sa.String(64), nullable=False),
            sa.Column("actor_name", sa.String(255), nullable=True),
            sa.Column("notification_state", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
            sa.ForeignKeyConstraint(["version_id"], ["artifact_versions.id"]),
            sa.ForeignKeyConstraint(["parent_comment_id"], ["artifact_comments.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_artifact_comments_artifact_id"), "artifact_comments", ["artifact_id"], unique=False)
        op.create_index(op.f("ix_artifact_comments_version_id"), "artifact_comments", ["version_id"], unique=False)
        op.create_index(
            op.f("ix_artifact_comments_parent_comment_id"),
            "artifact_comments",
            ["parent_comment_id"],
            unique=False,
        )
    else:
        comment_cols = _columns("artifact_comments")
        with op.batch_alter_table("artifact_comments") as batch_op:
            _add_base_timestamps(batch_op, comment_cols)
            if "proposed_patch" not in comment_cols:
                batch_op.add_column(sa.Column("proposed_patch", sa.JSON(), nullable=True))

    if not _has_table("artifact_activity_events"):
        op.create_table(
            "artifact_activity_events",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("artifact_id", sa.Uuid(), nullable=False),
            sa.Column("version_id", sa.Uuid(), nullable=True),
            sa.Column("event_type", sa.String(128), nullable=False),
            sa.Column("actor_name", sa.String(255), nullable=True),
            sa.Column("details", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
            sa.ForeignKeyConstraint(["version_id"], ["artifact_versions.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_artifact_activity_events_artifact_id"),
            "artifact_activity_events",
            ["artifact_id"],
            unique=False,
        )
        op.create_index(
            op.f("ix_artifact_activity_events_version_id"),
            "artifact_activity_events",
            ["version_id"],
            unique=False,
        )


def upgrade() -> None:
    """Upgrade schema."""
    if _has_table("artifacts"):
        _upgrade_existing_artifact_tables()
        _create_collaboration_tables()
        return

    op.create_table(
        "artifacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=True),
        sa.Column("slug", sa.String(255), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("artifact_type", sa.String(64), nullable=True),
        sa.Column("path", sa.String(2048), nullable=False),
        sa.Column("current_version_id", sa.Uuid(), nullable=True),
        sa.Column("last_known_good_version_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("path", name="uq_artifacts_path"),
        sa.UniqueConstraint("project_id", "slug", name="uq_artifacts_project_slug"),
    )
    op.create_index(op.f("ix_artifacts_current_version_id"), "artifacts", ["current_version_id"], unique=False)
    op.create_index(
        op.f("ix_artifacts_last_known_good_version_id"),
        "artifacts",
        ["last_known_good_version_id"],
        unique=False,
    )
    op.create_index(op.f("ix_artifacts_project_id"), "artifacts", ["project_id"], unique=False)
    op.create_index(op.f("ix_artifacts_slug"), "artifacts", ["slug"], unique=False)

    op.create_table(
        "artifact_versions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("artifact_id", sa.Uuid(), nullable=False),
        sa.Column("parent_version_id", sa.Uuid(), nullable=True),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(255), nullable=True),
        sa.Column("manifest_hash", sa.String(64), nullable=False),
        sa.Column("files_hash", sa.String(64), nullable=False),
        sa.Column("file_count", sa.Integer(), nullable=False),
        sa.Column("total_bytes", sa.Integer(), nullable=False),
        sa.Column("store_path", sa.String(2048), nullable=False),
        sa.Column("source_conversation_id", sa.Uuid(), nullable=True),
        sa.Column("source_message_id", sa.Uuid(), nullable=True),
        sa.Column("prompt", sa.Text(), nullable=True),
        sa.Column("operation_type", sa.String(64), nullable=False),
        sa.Column("preview_status", sa.String(64), nullable=False),
        sa.Column("publish_status", sa.String(64), nullable=False),
        sa.Column("restored_from_version_id", sa.Uuid(), nullable=True),
        sa.Column("branch_name", sa.String(255), nullable=True),
        sa.Column("forked_from_version_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["parent_version_id"], ["artifact_versions.id"]),
        sa.ForeignKeyConstraint(["restored_from_version_id"], ["artifact_versions.id"]),
        sa.ForeignKeyConstraint(["forked_from_version_id"], ["artifact_versions.id"]),
        sa.ForeignKeyConstraint(["source_conversation_id"], ["conversations.id"]),
        sa.ForeignKeyConstraint(["source_message_id"], ["messages.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("artifact_id", "version_number", name="uq_artifact_versions_number"),
    )
    op.create_index(op.f("ix_artifact_versions_artifact_id"), "artifact_versions", ["artifact_id"], unique=False)
    op.create_index(op.f("ix_artifact_versions_files_hash"), "artifact_versions", ["files_hash"], unique=False)
    op.create_index(op.f("ix_artifact_versions_parent_version_id"), "artifact_versions", ["parent_version_id"], unique=False)
    op.create_index(op.f("ix_artifact_versions_manifest_hash"), "artifact_versions", ["manifest_hash"], unique=False)
    op.create_index(
        op.f("ix_artifact_versions_restored_from_version_id"),
        "artifact_versions",
        ["restored_from_version_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_artifact_versions_source_conversation_id"),
        "artifact_versions",
        ["source_conversation_id"],
        unique=False,
    )
    op.create_index(op.f("ix_artifact_versions_source_message_id"), "artifact_versions", ["source_message_id"], unique=False)
    op.create_index(
        op.f("ix_artifact_versions_forked_from_version_id"),
        "artifact_versions",
        ["forked_from_version_id"],
        unique=False,
    )

    op.create_table(
        "artifact_version_files",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("version_id", sa.Uuid(), nullable=False),
        sa.Column("path", sa.String(2048), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("blob_path", sa.String(2048), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.ForeignKeyConstraint(["version_id"], ["artifact_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version_id", "path", name="uq_artifact_version_files_path"),
    )
    op.create_index(op.f("ix_artifact_version_files_content_hash"), "artifact_version_files", ["content_hash"], unique=False)
    op.create_index(op.f("ix_artifact_version_files_version_id"), "artifact_version_files", ["version_id"], unique=False)

    op.create_table(
        "artifact_drafts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("artifact_id", sa.Uuid(), nullable=False),
        sa.Column("base_version_id", sa.Uuid(), nullable=True),
        sa.Column("draft_path", sa.String(2048), nullable=False),
        sa.Column("status", sa.String(64), nullable=False),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["base_version_id"], ["artifact_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_artifact_drafts_artifact_id"), "artifact_drafts", ["artifact_id"], unique=False)
    op.create_index(op.f("ix_artifact_drafts_base_version_id"), "artifact_drafts", ["base_version_id"], unique=False)

    op.create_table(
        "artifact_deployments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("artifact_id", sa.Uuid(), nullable=False),
        sa.Column("version_id", sa.Uuid(), nullable=True),
        sa.Column("target", sa.String(128), nullable=False),
        sa.Column("status", sa.String(64), nullable=False),
        sa.Column("url", sa.String(2048), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["version_id"], ["artifact_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_artifact_deployments_artifact_id"), "artifact_deployments", ["artifact_id"], unique=False)
    op.create_index(op.f("ix_artifact_deployments_version_id"), "artifact_deployments", ["version_id"], unique=False)
    _create_collaboration_tables()


def downgrade() -> None:
    """Downgrade schema."""
    # If these tables were created by this migration alone, remove them.
    if _has_table("artifacts") and "folder_path" not in _columns("artifacts"):
        for table in (
            "artifact_activity_events",
            "artifact_comments",
            "artifact_deployments",
            "artifact_drafts",
            "artifact_version_files",
            "artifact_versions",
            "artifacts",
        ):
            if _has_table(table):
                op.drop_table(table)
        return

    if _has_table("artifact_activity_events"):
        if _has_index("artifact_activity_events", op.f("ix_artifact_activity_events_version_id")):
            op.drop_index(op.f("ix_artifact_activity_events_version_id"), table_name="artifact_activity_events")
        if _has_index("artifact_activity_events", op.f("ix_artifact_activity_events_artifact_id")):
            op.drop_index(op.f("ix_artifact_activity_events_artifact_id"), table_name="artifact_activity_events")
        op.drop_table("artifact_activity_events")

    if _has_table("artifact_comments"):
        if _has_index("artifact_comments", op.f("ix_artifact_comments_parent_comment_id")):
            op.drop_index(op.f("ix_artifact_comments_parent_comment_id"), table_name="artifact_comments")
        if _has_index("artifact_comments", op.f("ix_artifact_comments_version_id")):
            op.drop_index(op.f("ix_artifact_comments_version_id"), table_name="artifact_comments")
        if _has_index("artifact_comments", op.f("ix_artifact_comments_artifact_id")):
            op.drop_index(op.f("ix_artifact_comments_artifact_id"), table_name="artifact_comments")
        op.drop_table("artifact_comments")

    if _has_index("artifact_deployments", "ix_artifact_deployments_version_id"):
        op.drop_index("ix_artifact_deployments_version_id", table_name="artifact_deployments")
    if _has_index("artifact_deployments", "ix_artifact_deployments_artifact_id"):
        op.drop_index("ix_artifact_deployments_artifact_id", table_name="artifact_deployments")
    deployment_cols = _columns("artifact_deployments")
    with op.batch_alter_table("artifact_deployments") as batch_op:
        if "details" in deployment_cols:
            batch_op.drop_column("details")
        if "target" in deployment_cols:
            batch_op.drop_column("target")

    if _has_index("artifact_drafts", op.f("ix_artifact_drafts_base_version_id")):
        op.drop_index(op.f("ix_artifact_drafts_base_version_id"), table_name="artifact_drafts")
    draft_cols = _columns("artifact_drafts")
    with op.batch_alter_table("artifact_drafts") as batch_op:
        if "details" in draft_cols:
            batch_op.drop_column("details")
        if "draft_path" in draft_cols:
            batch_op.drop_column("draft_path")

    if _has_index("artifact_version_files", op.f("ix_artifact_version_files_content_hash")):
        op.drop_index(op.f("ix_artifact_version_files_content_hash"), table_name="artifact_version_files")
    file_cols = _columns("artifact_version_files")
    with op.batch_alter_table("artifact_version_files") as batch_op:
        if "blob_path" in file_cols:
            batch_op.drop_column("blob_path")
        if "content_hash" in file_cols:
            batch_op.drop_column("content_hash")

    for index_name in (
        op.f("ix_artifact_versions_source_message_id"),
        op.f("ix_artifact_versions_source_conversation_id"),
        op.f("ix_artifact_versions_restored_from_version_id"),
        op.f("ix_artifact_versions_forked_from_version_id"),
        op.f("ix_artifact_versions_parent_version_id"),
        op.f("ix_artifact_versions_artifact_id"),
        "uq_artifact_versions_number",
    ):
        if _has_index("artifact_versions", index_name):
            op.drop_index(index_name, table_name="artifact_versions")
    version_cols = _columns("artifact_versions")
    with op.batch_alter_table("artifact_versions") as batch_op:
        for column in (
            "forked_from_version_id",
            "branch_name",
            "restored_from_version_id",
            "store_path",
            "total_bytes",
            "file_count",
            "label",
            "version_number",
            "parent_version_id",
        ):
            if column in version_cols:
                batch_op.drop_column(column)

    for index_name in (op.f("ix_artifacts_slug"), "uq_artifacts_path"):
        if _has_index("artifacts", index_name):
            op.drop_index(index_name, table_name="artifacts")
    artifact_cols = _columns("artifacts")
    with op.batch_alter_table("artifacts") as batch_op:
        if "last_known_good_version_id" in artifact_cols:
            batch_op.drop_column("last_known_good_version_id")
        if "description" in artifact_cols:
            batch_op.drop_column("description")
        if "path" in artifact_cols:
            batch_op.drop_column("path")

    return
