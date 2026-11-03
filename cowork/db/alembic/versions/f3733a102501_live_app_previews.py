"""Add bounded, versioned browser preview leases.

Revision ID: f3733a102501
Revises: e3618a102301
"""
from alembic import op
import sqlalchemy as sa

revision = "f3733a102501"
down_revision = "e3618a102301"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not sa.inspect(op.get_bind()).has_table("live_app_preview_owners"):
        op.create_table(
            "live_app_preview_owners",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column("marker", sa.Integer(), nullable=False),
            sa.Column("revision", sa.Integer(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
    if sa.inspect(op.get_bind()).has_table("live_app_previews"):
        return
    op.create_table(
        "live_app_previews",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("owner_partition_id", sa.String(36), nullable=False),
        sa.Column("owner_user_id", sa.String(36), nullable=False),
        sa.Column("owner_email", sa.String(254), nullable=False),
        sa.Column("viewer_cookie", sa.Text()),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("launch_id", sa.Uuid(), nullable=False),
        sa.Column("origin", sa.String(256), nullable=False),
        sa.Column("intent_hash", sa.String(64), nullable=False),
        sa.Column("grant_hash", sa.String(64)),
        sa.Column("session_hash", sa.String(64)),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("owner_partition_id", "owner_user_id", "position", name="uq_live_app_preview_position"),
        sa.CheckConstraint("position >= 0 AND position < 8", name="ck_live_app_preview_position"),
    )
    for field in ("intent_hash", "grant_hash", "session_hash"):
        op.create_index(f"ix_live_app_previews_{field}", "live_app_previews", [field], unique=True)


def downgrade() -> None:
    op.drop_table("live_app_previews")
    op.drop_table("live_app_preview_owners")
