"""shared resource attribution and mutation audit

Revision ID: e2a6c4f8b1d3
Revises: a4c8e1f6b3d9
Create Date: 2026-08-29 12:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "e2a6c4f8b1d3"
down_revision: Union[str, Sequence[str], None] = "a4c8e1f6b3d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if not _has_table("shared_resource_attributions"):
        op.create_table(
            "shared_resource_attributions",
            sa.Column("id", sa.Uuid(), nullable=False),
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
            sa.Column("org_id", sa.String(length=36), nullable=False),
            sa.Column("resource_kind", sa.String(length=32), nullable=False),
            sa.Column("resource_key", sa.String(length=255), nullable=False),
            sa.Column("created_by_id", sa.String(length=36), nullable=True),
            sa.Column("created_by_email", sa.String(length=320), nullable=True),
            sa.Column("updated_by_id", sa.String(length=36), nullable=True),
            sa.Column("updated_by_email", sa.String(length=320), nullable=True),
            sa.Column("pending_claim_token", sa.String(length=36), nullable=True),
            sa.Column(
                "pending_claim_expires_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "org_id",
                "resource_kind",
                "resource_key",
                name="uq_shared_resource_attribution_key",
            ),
        )
        op.create_index(
            "ix_shared_resource_attributions_org_id",
            "shared_resource_attributions",
            ["org_id"],
        )
        op.create_index(
            "ix_shared_resource_attributions_resource_kind",
            "shared_resource_attributions",
            ["resource_kind"],
        )

    if not _has_table("shared_resource_mutations"):
        op.create_table(
            "shared_resource_mutations",
            sa.Column("id", sa.Uuid(), nullable=False),
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
            sa.Column("org_id", sa.String(length=36), nullable=False),
            sa.Column("resource_kind", sa.String(length=32), nullable=False),
            sa.Column("resource_key", sa.String(length=255), nullable=False),
            sa.Column("action", sa.String(length=32), nullable=False),
            sa.Column("actor_id", sa.String(length=36), nullable=False),
            sa.Column("actor_email", sa.String(length=320), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_shared_resource_mutations_org_id",
            "shared_resource_mutations",
            ["org_id"],
        )
        op.create_index(
            "ix_shared_resource_mutations_resource",
            "shared_resource_mutations",
            ["org_id", "resource_kind", "resource_key"],
        )


def downgrade() -> None:
    if _has_table("shared_resource_mutations"):
        op.drop_index(
            "ix_shared_resource_mutations_resource",
            table_name="shared_resource_mutations",
        )
        op.drop_index(
            "ix_shared_resource_mutations_org_id",
            table_name="shared_resource_mutations",
        )
        op.drop_table("shared_resource_mutations")
    if _has_table("shared_resource_attributions"):
        op.drop_index(
            "ix_shared_resource_attributions_resource_kind",
            table_name="shared_resource_attributions",
        )
        op.drop_index(
            "ix_shared_resource_attributions_org_id",
            table_name="shared_resource_attributions",
        )
        op.drop_table("shared_resource_attributions")
