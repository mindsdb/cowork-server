"""channel approvals

Adds per-binding gated_tools and the channel_pending_actions audit table.

Revision ID: d4e5f6a7b8c9
Revises: b7c1d2e3f4a5
Create Date: 2026-06-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "b7c1d2e3f4a5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("channel_bindings") as batch:
        batch.add_column(sa.Column("gated_tools", sa.JSON(), nullable=True))

    op.create_table(
        "channel_pending_actions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("channel_type", sa.String(64), nullable=False),
        sa.Column("binding_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=True),
        sa.Column("tool_name", sa.String(255), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("responder_id", sa.String(255), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
        sa.ForeignKeyConstraint(["binding_id"], ["channel_bindings.id"]),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_channel_pending_actions_binding_id"), "channel_pending_actions", ["binding_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_channel_pending_actions_binding_id"), table_name="channel_pending_actions")
    op.drop_table("channel_pending_actions")
    with op.batch_alter_table("channel_bindings") as batch:
        batch.drop_column("gated_tools")
