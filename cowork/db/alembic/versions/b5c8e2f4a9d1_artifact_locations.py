"""artifact_locations: the folder of each artifact id, per organization

A lookup by artifact id reads its row and opens that one folder, instead of
listing every artifacts root in the project. One row per (org_id,
artifact_id); the desktop writes none.

Revision ID: b5c8e2f4a9d1
Revises: a6e3c9f2b4d8
Create Date: 2026-10-30 11:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b5c8e2f4a9d1"
down_revision: Union[str, Sequence[str], None] = "a6e3c9f2b4d8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    """Upgrade schema."""
    if _has_table("artifact_locations"):
        return
    op.create_table(
        "artifact_locations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("org_id", sa.String(length=36), nullable=False),
        sa.Column("artifact_id", sa.String(length=32), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=True),
        sa.Column("slug", sa.String(length=255), nullable=False),
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
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("org_id", "artifact_id", name="uq_artifact_locations_org_artifact"),
    )
    op.create_index("ix_artifact_locations_project_id", "artifact_locations", ["project_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_artifact_locations_project_id", table_name="artifact_locations")
    op.drop_table("artifact_locations")
