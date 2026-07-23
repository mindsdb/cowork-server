"""approval_tokens: one-shot execution tokens for resolved approvals

Issued at resolve time, bound to the full approved payload (tool + args +
snapshot version), hash-stored, single-use — an approval can't be
double-spent or aimed at changed page state.

Revision ID: b4e9d3a2c6f1
Revises: a3f8c2d1e5b9
Create Date: 2026-07-23 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b4e9d3a2c6f1"
down_revision: Union[str, Sequence[str], None] = "a3f8c2d1e5b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    """Upgrade schema."""
    if _has_table("approval_tokens"):
        return
    op.create_table(
        "approval_tokens",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("approval_id", sa.Uuid(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
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
        sa.ForeignKeyConstraint(["approval_id"], ["approvals.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("ix_approval_tokens_approval_id", "approval_tokens", ["approval_id"])


def downgrade() -> None:
    """Downgrade schema."""
    if not _has_table("approval_tokens"):
        return
    op.drop_index("ix_approval_tokens_approval_id", table_name="approval_tokens")
    op.drop_table("approval_tokens")
