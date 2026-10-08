"""custom_connectors: connector definitions built in Cowork

Stores a connector built through the agent (form, display metadata, agent
usage notes) so it is listed beside the static registry. One definition per
connector id per org, or one local definition when org_id is NULL.

Revision ID: a6e3c9f2b4d8
Revises: 3e4b5f7586d3
Create Date: 2026-10-08 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a6e3c9f2b4d8"
down_revision: Union[str, Sequence[str], None] = "3e4b5f7586d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    """Upgrade schema."""
    if _has_table("custom_connectors"):
        return
    op.create_table(
        "custom_connectors",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("connector_id", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=True),
        sa.Column("logo_color", sa.String(length=32), nullable=True),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("usage_notes", sa.String(), nullable=True),
        sa.Column("featured", sa.Boolean(), nullable=False),
        sa.Column("org_id", sa.String(length=36), nullable=True),
        sa.Column("created_by", sa.String(length=36), nullable=True),
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_custom_connectors_org_id", "custom_connectors", ["org_id"])
    op.create_index(
        "uq_custom_connectors_connector_id_global",
        "custom_connectors",
        ["connector_id"],
        unique=True,
        sqlite_where=sa.text("org_id IS NULL"),
        postgresql_where=sa.text("org_id IS NULL"),
    )
    op.create_index(
        "uq_custom_connectors_connector_id_org",
        "custom_connectors",
        ["connector_id", "org_id"],
        unique=True,
        sqlite_where=sa.text("org_id IS NOT NULL"),
        postgresql_where=sa.text("org_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    if not _has_table("custom_connectors"):
        return
    op.drop_index("uq_custom_connectors_connector_id_org", table_name="custom_connectors")
    op.drop_index("uq_custom_connectors_connector_id_global", table_name="custom_connectors")
    op.drop_index("ix_custom_connectors_org_id", table_name="custom_connectors")
    op.drop_table("custom_connectors")
