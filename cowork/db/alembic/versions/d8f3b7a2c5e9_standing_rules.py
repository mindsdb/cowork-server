"""standing_rules: durable "Always" permissions for agent actions

scope = origin + action_kind; the browser gate bypasses proposals on an
exact match. Granted from evidence (3+ identical unmodified approvals),
revocable with one click — a grant never exists without visible revocation.

Revision ID: d8f3b7a2c5e9
Revises: c7d4e1f3a8b5
Create Date: 2026-07-23 15:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d8f3b7a2c5e9"
down_revision: Union[str, Sequence[str], None] = "c7d4e1f3a8b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    """Upgrade schema."""
    if _has_table("standing_rules"):
        return
    op.create_table(
        "standing_rules",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("origin", sa.String(length=255), nullable=False),
        sa.Column("action_kind", sa.String(length=255), nullable=False),
        sa.Column("source_approval_id", sa.Uuid(), nullable=False),
        sa.Column("hit_count", sa.Integer(), nullable=False),
        sa.Column("last_fired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(["source_approval_id"], ["approvals.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_standing_rules_origin", "standing_rules", ["origin"])
    op.create_index("ix_standing_rules_action_kind", "standing_rules", ["action_kind"])


def downgrade() -> None:
    """Downgrade schema."""
    if not _has_table("standing_rules"):
        return
    op.drop_index("ix_standing_rules_action_kind", table_name="standing_rules")
    op.drop_index("ix_standing_rules_origin", table_name="standing_rules")
    op.drop_table("standing_rules")
