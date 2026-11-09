"""add conversation channel_type

The channel a conversation came from (discord, slack, ...), set by the channel
runtime when it creates the conversation; NULL for every other conversation.
64 wide to match channel_bindings.channel_type, which it is copied from.

Revision ID: c69488788ac7
Revises: 2a366c1f4b9f
Create Date: 2026-11-03 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c69488788ac7"
down_revision: Union[str, Sequence[str], None] = "2a366c1f4b9f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table_name: str, column_name: str) -> bool:
    """Whether `table_name` already has `column_name` on the bound connection."""
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    """Add the nullable conversations.channel_type column."""
    if not _has_column("conversations", "channel_type"):
        op.add_column("conversations", sa.Column("channel_type", sa.String(64), nullable=True))


def downgrade() -> None:
    """Drop conversations.channel_type."""
    with op.batch_alter_table("conversations") as batch_op:
        if _has_column("conversations", "channel_type"):
            batch_op.drop_column("channel_type")
