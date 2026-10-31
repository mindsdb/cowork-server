"""queued_agent_repairs: the artifact folders holding a repair queued to a conversation

The end of a conversation's turn reads these rows to check only the folders
that hold one of its queued agent repairs, instead of every artifact folder.
Rows are keyed by org: one row per conversation and folder in an org, or one
among the rows with no org (desktop and local).

Revision ID: c7d1e3f5a8b2
Revises: b5c8e2f4a9d1
Create Date: 2026-10-30 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c7d1e3f5a8b2"
down_revision: Union[str, Sequence[str], None] = "b5c8e2f4a9d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    """Upgrade schema."""
    if _has_table("queued_agent_repairs"):
        return
    op.create_table(
        "queued_agent_repairs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("org_id", sa.String(length=36), nullable=True),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_queued_agent_repairs_org_id", "queued_agent_repairs", ["org_id"])
    op.create_index(
        "ix_queued_agent_repairs_conversation_id", "queued_agent_repairs", ["conversation_id"]
    )
    op.create_index(
        "uq_queued_agent_repairs_conversation_slug_global",
        "queued_agent_repairs",
        ["conversation_id", "slug"],
        unique=True,
        sqlite_where=sa.text("org_id IS NULL"),
        postgresql_where=sa.text("org_id IS NULL"),
    )
    op.create_index(
        "uq_queued_agent_repairs_conversation_slug_org",
        "queued_agent_repairs",
        ["org_id", "conversation_id", "slug"],
        unique=True,
        sqlite_where=sa.text("org_id IS NOT NULL"),
        postgresql_where=sa.text("org_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("uq_queued_agent_repairs_conversation_slug_org", table_name="queued_agent_repairs")
    op.drop_index("uq_queued_agent_repairs_conversation_slug_global", table_name="queued_agent_repairs")
    op.drop_index("ix_queued_agent_repairs_conversation_id", table_name="queued_agent_repairs")
    op.drop_index("ix_queued_agent_repairs_org_id", table_name="queued_agent_repairs")
    op.drop_table("queued_agent_repairs")
