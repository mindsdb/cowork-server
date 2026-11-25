"""Private content-report queue.
Revision ID: a5c8721e09dd
Revises: f6a2c4e8b1d3
"""
from alembic import op
import sqlalchemy as sa
revision = "a5c8721e09dd"
down_revision = "f6a2c4e8b1d3"
branch_labels = None
depends_on = None

def upgrade():
    if "content_reports" in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table("content_reports",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("org_id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(36), nullable=False),
        sa.Column("target_key", sa.String(128), nullable=False),
        sa.Column("conversation_id", sa.Uuid()),
        sa.Column("message_id", sa.Uuid()),
        sa.Column("project_id", sa.Uuid()),
        sa.Column("artifact_id", sa.Uuid()),
        sa.Column("evidence", sa.String(65536), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("reviewed_by", sa.String(128)),
        sa.Column("resolution", sa.String(2000)),
        sa.UniqueConstraint("org_id", "user_id", "target_key", name="uq_content_report_target"),
    )
    op.create_index("ix_content_reports_queue", "content_reports", ["status", "created_at"])
    op.create_index("ix_content_reports_reporter", "content_reports", ["org_id", "user_id", "created_at"])

def downgrade():
    # SQLite can retain the DROP when a later downgrade step refuses to run.
    op.drop_table("content_reports", if_exists=True)
