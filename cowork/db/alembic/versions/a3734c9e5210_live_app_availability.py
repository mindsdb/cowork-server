"""Keep full-slot refusals and availability notices across browser reloads.

Revision ID: a3734c9e5210
Revises: e3618a102301
"""
from alembic import op
import sqlalchemy as sa

revision = "a3734c9e5210"
down_revision = "e3618a102301"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("live_app_availability_waits"):
        return
    op.create_table(
        "live_app_availability_waits",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("partition_id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("generation", sa.Uuid(), nullable=False),
        sa.Column("notified", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("partition_id", "user_id", name="uq_live_app_availability_person"),
    )


def downgrade() -> None:
    op.drop_table("live_app_availability_waits")
