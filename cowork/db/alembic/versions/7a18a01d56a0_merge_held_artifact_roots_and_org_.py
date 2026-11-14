"""merge the held artifact roots and the org appearance heads

``7e636946b5f8`` (the held_artifact_roots table) and
``e2a7c4f9b613`` (staging's merge of the drop-org-appearance and artifact
locations heads) both descend from ``f93151b5bad1``. Two heads make
``alembic upgrade head`` fail everywhere, server startup included, so this
empty revision rejoins them. It touches no schema and rewrites neither
branch's down_revision, so a database stamped at either head stays
consistent. The branches touch disjoint tables, so their order is
irrelevant.

Revision ID: 7a18a01d56a0
Revises: e2a7c4f9b613, 7e636946b5f8
Create Date: 2026-11-10 21:58:41.487200

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "7a18a01d56a0"
down_revision: Union[str, Sequence[str], None] = ("e2a7c4f9b613", "7e636946b5f8")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
