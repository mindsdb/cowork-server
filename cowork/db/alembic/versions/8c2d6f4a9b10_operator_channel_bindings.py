"""Require explicit operator setup for trusted-local channel bindings.

Revision ID: 8c2d6f4a9b10
Revises: 3e4b5f7586d3
"""
from alembic import op
import sqlalchemy as sa

revision = "8c2d6f4a9b10"
down_revision = "3e4b5f7586d3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Legacy database adoption can start from a schema created by the models.
    if "operator_configured" in {column["name"] for column in sa.inspect(op.get_bind()).get_columns("channel_bindings")}:
        return
    op.add_column("channel_bindings", sa.Column(
        "operator_configured", sa.Boolean(), nullable=False, server_default=sa.false(),
    ))


def downgrade() -> None:
    op.drop_column("channel_bindings", "operator_configured")
