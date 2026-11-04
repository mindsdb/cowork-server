"""sidekick_contexts: a conversation's Chimera sidekick between turns

One row per conversation, keyed by its id: anton's sidekick state as the last
turn left it (its id, model, own message list, summary, status, when the main
agent last delegated, and its signal count), handed back to the next turn
(cowork.models.sidekick_context). No org column: the row is reached only
through its conversation.

The downgrade drops the table. Losing it costs a sidekick its earlier messages,
and the next delegation starts it fresh.

Revision ID: f6c2d8a4b1e9
Revises: 2a366c1f4b9f
Create Date: 2026-11-04 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "f6c2d8a4b1e9"
down_revision: Union[str, Sequence[str], None] = "2a366c1f4b9f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "sidekick_contexts"


def _has_table() -> bool:
    return _TABLE in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    # Bootstrap can create SQLModel metadata before Alembic adopts the schema.
    if _has_table():
        return
    op.create_table(
        _TABLE,
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("sidekick_id", sa.String(length=128), nullable=True),
        sa.Column("model", sa.String(length=256), nullable=True),
        sa.Column("messages", sa.JSON(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("last_call_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("signals", sa.JSON(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.PrimaryKeyConstraint("conversation_id"),
    )


def downgrade() -> None:
    if _has_table():
        op.drop_table(_TABLE)
