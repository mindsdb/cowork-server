"""merge the held artifact roots and the push devices heads

``7a18a01d56a0`` (the held artifact roots' merge revision) and
``1d5fe417cc0a`` (staging's push devices table, after the schedule runs
revisions) both descend from ``e2a7c4f9b613``. Two heads make
``alembic upgrade head`` fail everywhere, server startup included, so this
empty revision rejoins them. It touches no schema and rewrites neither
branch's down_revision, so a database stamped at either head stays
consistent. The branches touch disjoint tables, so their order is
irrelevant.

Revision ID: f7a2ecdd2c79
Revises: 1d5fe417cc0a, 7a18a01d56a0
Create Date: 2026-11-13 07:48:46.066811

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "f7a2ecdd2c79"
down_revision: Union[str, Sequence[str], None] = ("1d5fe417cc0a", "7a18a01d56a0")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
