"""Index historical artifact notification URLs by organization and URL digest."""
from alembic import op
import sqlalchemy as sa

revision = "b97cd8ab410f"
down_revision = "fabea0eac5e2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("artifact_share_links"):
        return
    op.create_table(
        "artifact_share_links",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("org_id", sa.String(36), nullable=False),
        sa.Column("url_digest", sa.String(64), nullable=False),
        sa.Column("project_id", sa.Uuid(), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("artifact_id", sa.String(32), nullable=False),
        sa.UniqueConstraint("org_id", "url_digest", "project_id", "artifact_id", name="uq_artifact_share_target"),
    )
    op.create_index("ix_artifact_share_links_project_id", "artifact_share_links", ["project_id"])


def downgrade() -> None:
    if sa.inspect(op.get_bind()).has_table("artifact_share_links"):
        op.drop_table("artifact_share_links")
