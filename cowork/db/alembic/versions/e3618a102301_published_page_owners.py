"""Persist shared-publisher ownership independently of editable artifact files.

Revision ID: e3618a102301
Revises: f3b9d6a2c8e4
"""
from alembic import op
import sqlalchemy as sa

revision = "e3618a102301"
down_revision = "f3b9d6a2c8e4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("published_page_owners"):
        return
    op.create_table(
        "published_page_owners",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("publisher_key", sa.String(64), nullable=False),
        sa.Column("report_id", sa.String(255), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("owner_user_id", sa.String(36), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("publisher_key", "report_id", name="uq_published_page_owner"),
    )


def downgrade() -> None:
    # Retain authorization history across rollback and a subsequent upgrade.
    pass
