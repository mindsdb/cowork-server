"""held_artifact_roots: the artifact roots the server has seen hold a folder

The artifacts list opens only the roots these rows name, instead of every
project's shared root and every conversation workspace of the caller. One row
per (org_id, project_id, conversation_id), conversation_id NULL for a
project's shared root; a NULL never equals another, so the shared root has a
partial unique index of its own. conversation_id is indexed for its ON DELETE
CASCADE, so deleting a conversation does not scan the table.

Revision ID: 7e636946b5f8
Revises: f93151b5bad1
Create Date: 2026-11-09 07:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7e636946b5f8"
down_revision: Union[str, Sequence[str], None] = "f93151b5bad1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    """Upgrade schema."""
    if _has_table("held_artifact_roots"):
        return
    op.create_table(
        "held_artifact_roots",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("org_id", sa.String(length=36), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=True),
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
    op.create_index("ix_held_artifact_roots_project_id", "held_artifact_roots", ["project_id"])
    op.create_index("ix_held_artifact_roots_conversation_id", "held_artifact_roots", ["conversation_id"])
    op.create_index(
        "uq_held_artifact_roots_shared",
        "held_artifact_roots",
        ["org_id", "project_id"],
        unique=True,
        sqlite_where=sa.text("conversation_id IS NULL"),
        postgresql_where=sa.text("conversation_id IS NULL"),
    )
    op.create_index(
        "uq_held_artifact_roots_legacy",
        "held_artifact_roots",
        ["org_id", "project_id", "conversation_id"],
        unique=True,
        sqlite_where=sa.text("conversation_id IS NOT NULL"),
        postgresql_where=sa.text("conversation_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    # The settings row that marks the rows as complete
    # (cowork.services.held_artifact_roots.SENTINEL_KEY, kept as a literal so
    # this revision stands alone) goes with the rows it vouches for: left
    # behind, a later upgrade would find it over an empty table, never fill
    # the table again, and the list would show no card.
    if _has_table("settings"):
        op.execute(sa.text("DELETE FROM settings WHERE key = '_held_artifact_roots_backfill_v1'"))
    if not _has_table("held_artifact_roots"):
        return
    op.drop_index("uq_held_artifact_roots_legacy", table_name="held_artifact_roots")
    op.drop_index("uq_held_artifact_roots_shared", table_name="held_artifact_roots")
    op.drop_index("ix_held_artifact_roots_conversation_id", table_name="held_artifact_roots")
    op.drop_index("ix_held_artifact_roots_project_id", table_name="held_artifact_roots")
    op.drop_table("held_artifact_roots")
