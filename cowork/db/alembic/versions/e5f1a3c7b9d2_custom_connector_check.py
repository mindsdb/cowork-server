"""custom_connectors: keep the template's connection check

A custom connector template may declare how to test its credentials: one GET
of a path, with headers that place the member's field values. It is stored
beside the form, as anton's parser accepted it, and never names a host.

Revision ID: e5f1a3c7b9d2
Revises: d4b8e2f6a9c1
Create Date: 2026-11-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e5f1a3c7b9d2"
down_revision: Union[str, Sequence[str], None] = "d4b8e2f6a9c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TABLE = "custom_connectors"
_COLUMN = "connection_check"


def _has_column() -> bool:
    """Whether the column exists. A database created from the models before
    Alembic already has it."""
    return _COLUMN in {column["name"] for column in sa.inspect(op.get_bind()).get_columns(_TABLE)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_column():
        op.add_column(_TABLE, sa.Column(_COLUMN, sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema. Drops every stored check; the templates stay."""
    if _has_column():
        op.drop_column(_TABLE, _COLUMN)
