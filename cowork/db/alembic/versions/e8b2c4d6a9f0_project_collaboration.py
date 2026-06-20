"""project collaboration

Revision ID: e8b2c4d6a9f0
Revises: d3a5c9e7b1f2
Create Date: 2026-06-20 01:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e8b2c4d6a9f0"
down_revision: Union[str, Sequence[str], None] = "d3a5c9e7b1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _has_index(table_name: str, index_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return index_name in {index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_table("project_collaborators"):
        op.create_table(
            "project_collaborators",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("project_id", sa.Uuid(), nullable=False),
            sa.Column("email", sa.String(length=255), nullable=False),
            sa.Column("display_name", sa.String(length=255), nullable=True),
            sa.Column("role", sa.String(length=64), nullable=False),
            sa.Column("notification_state", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("project_id", "email", name="uq_project_collaborators_email"),
        )
        op.create_index(op.f("ix_project_collaborators_project_id"), "project_collaborators", ["project_id"], unique=False)
        op.create_index(op.f("ix_project_collaborators_email"), "project_collaborators", ["email"], unique=False)

    if not _has_table("project_notification_hooks"):
        op.create_table(
            "project_notification_hooks",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("project_id", sa.Uuid(), nullable=False),
            sa.Column("kind", sa.String(length=64), nullable=False),
            sa.Column("target", sa.String(length=1024), nullable=False),
            sa.Column("enabled", sa.Boolean(), nullable=False),
            sa.Column("events", sa.JSON(), nullable=True),
            sa.Column("secret_ciphertext", sa.Text(), nullable=True),
            sa.Column("config", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            op.f("ix_project_notification_hooks_project_id"),
            "project_notification_hooks",
            ["project_id"],
            unique=False,
        )

    if not _has_table("notification_deliveries"):
        op.create_table(
            "notification_deliveries",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("project_id", sa.Uuid(), nullable=False),
            sa.Column("hook_id", sa.Uuid(), nullable=True),
            sa.Column("event_key", sa.String(length=128), nullable=False),
            sa.Column("dedupe_key", sa.String(length=255), nullable=False),
            sa.Column("status", sa.String(length=64), nullable=False),
            sa.Column("attempts", sa.Integer(), nullable=False),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("details", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
            sa.ForeignKeyConstraint(["hook_id"], ["project_notification_hooks.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("hook_id", "dedupe_key", name="uq_notification_deliveries_hook_dedupe"),
        )
        op.create_index(op.f("ix_notification_deliveries_project_id"), "notification_deliveries", ["project_id"], unique=False)
        op.create_index(op.f("ix_notification_deliveries_hook_id"), "notification_deliveries", ["hook_id"], unique=False)
        op.create_index(op.f("ix_notification_deliveries_event_key"), "notification_deliveries", ["event_key"], unique=False)
        op.create_index(op.f("ix_notification_deliveries_dedupe_key"), "notification_deliveries", ["dedupe_key"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    if _has_table("notification_deliveries"):
        for index_name in (
            op.f("ix_notification_deliveries_dedupe_key"),
            op.f("ix_notification_deliveries_event_key"),
            op.f("ix_notification_deliveries_hook_id"),
            op.f("ix_notification_deliveries_project_id"),
        ):
            if _has_index("notification_deliveries", index_name):
                op.drop_index(index_name, table_name="notification_deliveries")
        op.drop_table("notification_deliveries")

    if _has_table("project_notification_hooks"):
        if _has_index("project_notification_hooks", op.f("ix_project_notification_hooks_project_id")):
            op.drop_index(op.f("ix_project_notification_hooks_project_id"), table_name="project_notification_hooks")
        op.drop_table("project_notification_hooks")

    if _has_table("project_collaborators"):
        if _has_index("project_collaborators", op.f("ix_project_collaborators_email")):
            op.drop_index(op.f("ix_project_collaborators_email"), table_name="project_collaborators")
        if _has_index("project_collaborators", op.f("ix_project_collaborators_project_id")):
            op.drop_index(op.f("ix_project_collaborators_project_id"), table_name="project_collaborators")
        op.drop_table("project_collaborators")
