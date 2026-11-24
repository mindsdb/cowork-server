"""artifact_datasource_approvals: connections approved for a full-stack publish

A full-stack publish sends credentials only of the connections a person
approved for the artifact's folder. Rows are keyed by org, or have none on
desktop and local deployments.

Revision ID: a8d3f6c1e924
Revises: fabea0eac5e2
Create Date: 2026-11-23 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a8d3f6c1e924"
down_revision: Union[str, Sequence[str], None] = "fabea0eac5e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "artifact_datasource_approvals"


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    """Upgrade schema."""
    if _has_table(TABLE):
        return
    op.create_table(
        TABLE,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("org_id", sa.String(length=36), nullable=True),
        sa.Column("artifact_key", sa.String(length=64), nullable=False),
        sa.Column("artifact_dir", sa.Text(), nullable=False),
        sa.Column("datasources", sa.JSON(), nullable=False),
        sa.Column("approved_by", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_artifact_datasource_approvals_org_id", TABLE, ["org_id"])
    op.create_index(
        "uq_artifact_datasource_approvals_global",
        TABLE,
        ["artifact_key", "artifact_dir"],
        unique=True,
        sqlite_where=sa.text("org_id IS NULL"),
        postgresql_where=sa.text("org_id IS NULL"),
    )
    op.create_index(
        "uq_artifact_datasource_approvals_org",
        TABLE,
        ["org_id", "artifact_key", "artifact_dir"],
        unique=True,
        sqlite_where=sa.text("org_id IS NOT NULL"),
        postgresql_where=sa.text("org_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    if not _has_table(TABLE):
        return
    op.drop_index("uq_artifact_datasource_approvals_org", table_name=TABLE)
    op.drop_index("uq_artifact_datasource_approvals_global", table_name=TABLE)
    op.drop_index("ix_artifact_datasource_approvals_org_id", table_name=TABLE)
    op.drop_table(TABLE)
