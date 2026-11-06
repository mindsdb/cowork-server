"""merge the org appearance and model comparisons heads

``3e78d31dc845`` (org_appearance, ENG-4020) and ``a7c3e5f9b2d4`` (model
comparisons) both descend from ``c69488788ac7``. Two heads make
``alembic upgrade head`` fail everywhere, server startup included, so this
empty revision rejoins them. It touches no schema and rewrites neither
branch's down_revision, so a database stamped at either head stays
consistent. The branches touch disjoint tables, so their order is
irrelevant.

Revision ID: 5b9e1c7d3a20
Revises: 3e78d31dc845, a7c3e5f9b2d4
Create Date: 2026-11-05 18:00:00.000000

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "5b9e1c7d3a20"
down_revision: Union[str, Sequence[str], None] = ("3e78d31dc845", "a7c3e5f9b2d4")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
