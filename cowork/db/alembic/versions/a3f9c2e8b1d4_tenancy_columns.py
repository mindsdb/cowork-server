"""tenancy ownership columns

Additive only, all nullable, no backfill: desktop rows keep NULL (local mode
never filters by org); cloud databases start empty and every cloud write is
stamped by the scoped session layer. Deterministic on purpose — Alembic
tracks applied revisions, so an unexpectedly different schema should fail
loudly here rather than be tolerated.

Revision ID: a3f9c2e8b1d4
Revises: f7d2b9e4a1c6
Create Date: 2026-07-16 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a3f9c2e8b1d4"
down_revision: Union[str, Sequence[str], None] = "f7d2b9e4a1c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _id() -> sa.String:
    # Canonical UUID text (36 chars) as emitted by the identity headers and
    # normalized by cowork.principal — the type the scoped layer compares.
    return sa.String(length=36)


# (table, column, type) in upgrade order.
_COLUMNS: list[tuple[str, str, sa.types.TypeEngine]] = [
    ("projects", "org_id", _id()),
    ("projects", "created_by", _id()),
    ("conversations", "org_id", _id()),
    ("conversations", "created_by", _id()),
    ("messages", "created_by", _id()),  # attribution only; tenancy via conversation
    ("files", "org_id", _id()),
    ("files", "created_by", _id()),
    ("schedules", "org_id", _id()),
    ("schedules", "created_by", _id()),
    ("channel_installations", "org_id", _id()),
    ("channel_bindings", "org_id", _id()),
    ("channel_bindings", "created_by", _id()),
    ("pins", "user_id", _id()),
    ("pins", "org_id", _id()),  # pins reference org-owned items; personal within an org
    ("settings", "scope", sa.String(length=16)),  # 'org' | 'user'; inert until week 6
    ("settings", "user_id", _id()),
    ("settings", "org_id", _id()),
]

# (index name, table, columns). org_id singles serve the scoped-select filter;
# pins get the composite for "this user's pins in this org" (leftmost prefix
# also covers user-only lookups). settings get none — the only live query path
# is by `key`, which is already unique-indexed.
_INDEXES: list[tuple[str, str, list[str]]] = [
    ("ix_projects_org_id", "projects", ["org_id"]),
    ("ix_conversations_org_id", "conversations", ["org_id"]),
    ("ix_files_org_id", "files", ["org_id"]),
    ("ix_schedules_org_id", "schedules", ["org_id"]),
    ("ix_channel_installations_org_id", "channel_installations", ["org_id"]),
    ("ix_channel_bindings_org_id", "channel_bindings", ["org_id"]),
    ("ix_pins_user_id_org_id", "pins", ["user_id", "org_id"]),
]


def upgrade() -> None:
    """Upgrade schema."""
    for table, column, type_ in _COLUMNS:
        op.add_column(table, sa.Column(column, type_, nullable=True))
    for name, table, columns in _INDEXES:
        op.create_index(name, table, columns)


def downgrade() -> None:
    """Downgrade schema (reverse order; indexes before their columns)."""
    for name, table, _columns in reversed(_INDEXES):
        op.drop_index(name, table_name=table)
    tables = list(dict.fromkeys(table for table, _c, _t in _COLUMNS))
    for table in reversed(tables):
        columns = [c for t, c, _ in _COLUMNS if t == table]
        # batch_alter_table: SQLite can't DROP COLUMN in place; the batch
        # rebuild preserves reflected constraints/indexes (verified by tests).
        with op.batch_alter_table(table) as batch_op:
            for column in reversed(columns):
                batch_op.drop_column(column)
