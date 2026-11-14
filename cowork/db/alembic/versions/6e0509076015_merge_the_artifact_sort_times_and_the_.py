"""merge the artifact sort times and the push devices heads

``33c9ba9927ad`` (the artifact_sort_times table) and ``f7a2ecdd2c79``
(the merge of the held artifact roots with staging's push devices table)
both descend from ``7a18a01d56a0``. Two heads make ``alembic upgrade head``
fail everywhere, server startup included, so this empty revision rejoins
them. It touches no schema and rewrites neither branch's down_revision, so
a database stamped at either head stays consistent. The branches touch
disjoint tables, so their order is irrelevant.

Revision ID: 6e0509076015
Revises: 33c9ba9927ad, f7a2ecdd2c79
Create Date: 2026-11-13 07:51:53.081432

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "6e0509076015"
down_revision: Union[str, Sequence[str], None] = ("33c9ba9927ad", "f7a2ecdd2c79")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
