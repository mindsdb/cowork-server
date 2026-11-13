"""custom_connectors: keep the template's auth section

A custom connector template says how the member's values go on a request:
one auth type with `{field}` slots. It is stored beside the form and the
check, as anton's parser accepted it, and holds no value.

Revision ID: f6a2c4e8b1d3
Revises: e5f1a3c7b9d2
Create Date: 2026-11-11 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f6a2c4e8b1d3"
down_revision: Union[str, Sequence[str], None] = "e5f1a3c7b9d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TABLE = "custom_connectors"
_COLUMN = "connection_auth"


def _has_column() -> bool:
    """Whether the column exists. A database created from the models before
    Alembic already has it."""
    return _COLUMN in {column["name"] for column in sa.inspect(op.get_bind()).get_columns(_TABLE)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_column():
        op.add_column(_TABLE, sa.Column(_COLUMN, sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema. Drops every stored auth section; the templates stay."""
    if _has_column():
        op.drop_column(_TABLE, _COLUMN)
