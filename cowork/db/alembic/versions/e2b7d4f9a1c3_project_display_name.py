"""add projects.display_name

Purely additive and deliberately NOT backfilled. `name` stays the internal
identifier — the on-disk directory, the URL segment and the lookup key — and
nothing about it moves. A NULL display_name means "created before this column
existed"; readers resolve it as `display_name or name`, so every existing
project renders exactly as it did (ENG-1676).

Revision ID: e2b7d4f9a1c3
Revises: cfbc79856e9e
Create Date: 2026-09-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e2b7d4f9a1c3"
down_revision: Union[str, Sequence[str], None] = "b7f4d2c9a3e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_column("projects", "display_name"):
        op.add_column("projects", sa.Column("display_name", sa.String(length=255), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Plain drop_column, NOT batch_alter_table. Batch mode rebuilds the table on
    # SQLite, which silently destroys `uq_projects_default_per_org` -- an
    # expression-based index SQLAlchemy cannot reflect (it warns and skips it),
    # so the rebuild omits it and every later downgrade in the chain then fails
    # on `DROP INDEX uq_projects_default_per_org`. SQLite has supported
    # ALTER TABLE DROP COLUMN since 3.35 (2021) and Postgres always has.
    if _has_column("projects", "display_name"):
        op.drop_column("projects", "display_name")
