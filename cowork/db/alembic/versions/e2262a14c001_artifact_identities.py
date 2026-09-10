"""Persist trusted artifact authorization aliases.

Revision ID: e2262a14c001
Revises: d5b28c4a91e7
"""

from alembic import op
import sqlalchemy as sa

revision = "e2262a14c001"
down_revision = "d5b28c4a91e7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Bootstrap can create SQLModel metadata before Alembic adopts the schema.
    # A downgrade also retains this security history for later re-upgrades.
    if sa.inspect(op.get_bind()).has_table("artifact_identities"):
        return
    op.create_table(
        "artifact_identities",
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
        sa.Column("org_id", sa.String(36), nullable=False),
        sa.Column("owner_keycloak_id", sa.String(36), nullable=False),
        sa.Column("local_artifact_id", sa.String(32), nullable=False),
        sa.Column("canonical_artifact_id", sa.String(64), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "org_id", "local_artifact_id", name="uq_artifact_identity_org_local"
        ),
    )
    op.create_index("ix_artifact_identities_org_id", "artifact_identities", ["org_id"])


def downgrade() -> None:
    # These rows are authorization history, not a disposable cache. An older
    # application can ignore the extra table; dropping it would release names.
    pass
