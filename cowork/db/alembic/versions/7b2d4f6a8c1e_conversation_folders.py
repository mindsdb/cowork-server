"""conversation_folders: working folders attached to a desktop chat

Revision ID: 7b2d4f6a8c1e
Revises: 3e4b5f7586d3
Create Date: 2026-10-06 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7b2d4f6a8c1e"
down_revision: Union[str, Sequence[str], None] = "3e4b5f7586d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    """Upgrade schema."""
    if _has_table("conversation_folders"):
        return
    op.create_table(
        "conversation_folders",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("path", sa.String(length=1024), nullable=False),
        sa.Column("org_id", sa.String(length=36), nullable=True),
        sa.Column("created_by", sa.String(length=36), nullable=True),
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
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id", "path", name="uq_conversation_folders_conversation_path"
        ),
    )
    op.create_index(
        "ix_conversation_folders_conversation_id", "conversation_folders", ["conversation_id"]
    )
    op.create_index("ix_conversation_folders_org_id", "conversation_folders", ["org_id"])


def downgrade() -> None:
    """Downgrade schema."""
    if not _has_table("conversation_folders"):
        return
    op.drop_index("ix_conversation_folders_org_id", table_name="conversation_folders")
    op.drop_index("ix_conversation_folders_conversation_id", table_name="conversation_folders")
    op.drop_table("conversation_folders")
