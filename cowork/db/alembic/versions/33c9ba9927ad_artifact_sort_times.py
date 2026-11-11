"""artifact_sort_times: the time the artifacts list sorts each folder by

The list reads the newest rows of the roots it may open, and opens only
those folders, instead of every folder's metadata.json. One row per (org_id,
project_id, conversation_id, slug), conversation_id NULL for a project's
shared root; a NULL never equals another, so the shared root has a partial
unique index of its own. conversation_id is indexed for its ON DELETE
CASCADE.

Revision ID: 33c9ba9927ad
Revises: 7e636946b5f8
Create Date: 2026-11-09 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "33c9ba9927ad"
down_revision: Union[str, Sequence[str], None] = "7e636946b5f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    """Upgrade schema."""
    if _has_table("artifact_sort_times"):
        return
    op.create_table(
        "artifact_sort_times",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("org_id", sa.String(length=36), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=True),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("sort_time", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.Column(
            "modified_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_artifact_sort_times_conversation_id", "artifact_sort_times", ["conversation_id"])
    op.create_index("ix_artifact_sort_times_project_time", "artifact_sort_times", ["project_id", "sort_time"])
    op.create_index(
        "uq_artifact_sort_times_shared",
        "artifact_sort_times",
        ["org_id", "project_id", "slug"],
        unique=True,
        sqlite_where=sa.text("conversation_id IS NULL"),
        postgresql_where=sa.text("conversation_id IS NULL"),
    )
    op.create_index(
        "uq_artifact_sort_times_legacy",
        "artifact_sort_times",
        ["org_id", "project_id", "conversation_id", "slug"],
        unique=True,
        sqlite_where=sa.text("conversation_id IS NOT NULL"),
        postgresql_where=sa.text("conversation_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    # The settings row that marks the rows as complete
    # (cowork.services.artifact_sort_times.SENTINEL_KEY, kept as a literal so
    # this revision stands alone) goes with the rows it vouches for: left
    # behind, a later upgrade would find it over an empty table and the list
    # would show no card.
    if _has_table("settings"):
        op.execute(sa.text("DELETE FROM settings WHERE key = '_artifact_sort_times_backfill_v1'"))
    if not _has_table("artifact_sort_times"):
        return
    op.drop_index("uq_artifact_sort_times_legacy", table_name="artifact_sort_times")
    op.drop_index("uq_artifact_sort_times_shared", table_name="artifact_sort_times")
    op.drop_index("ix_artifact_sort_times_project_time", table_name="artifact_sort_times")
    op.drop_index("ix_artifact_sort_times_conversation_id", table_name="artifact_sort_times")
    op.drop_table("artifact_sort_times")
