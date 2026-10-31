"""artifact_owner_choices: the sharing an artifact's owner last chose

One row per artifact per organization, keyed by the artifact's authorization
key: the mode the owner chose (restricted, password or public) and whether the
artifact is published, written only by the owner's own save, so a re-publish
that isn't the owner's never goes wider than it
(cowork.models.artifact_owner_choice). The org and owner ids are UUID
columns. MindsHub mode writes it; the desktop's SQLite gets the empty table
and never reads it.

The downgrade drops the table. A row kept across a downgrade would miss every
choice the owner made on the older version, and a re-upgrade would then judge
re-publishes against a choice the owner had since changed.

Revision ID: d9c4e7a2b5f1
Revises: c5a8e2f1d7b3
Create Date: 2026-10-31 09:36:01.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "d9c4e7a2b5f1"
down_revision: Union[str, Sequence[str], None] = "c5a8e2f1d7b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "artifact_owner_choices"


def _has_table() -> bool:
    return _TABLE in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    # Bootstrap can create SQLModel metadata before Alembic adopts the schema.
    if _has_table():
        return
    op.create_table(
        _TABLE,
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
        sa.Column("org_id", sa.Uuid(), nullable=False),
        sa.Column("artifact_key", sa.String(length=64), nullable=False),
        sa.Column("access_mode", sa.String(length=16), nullable=False),
        sa.Column("published", sa.Boolean(), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("org_id", "artifact_key", name="uq_artifact_owner_choice_org_key"),
    )
    op.create_index("ix_artifact_owner_choices_org_id", _TABLE, ["org_id"])


def downgrade() -> None:
    if _has_table():
        op.drop_index("ix_artifact_owner_choices_org_id", table_name=_TABLE)
        op.drop_table(_TABLE)
