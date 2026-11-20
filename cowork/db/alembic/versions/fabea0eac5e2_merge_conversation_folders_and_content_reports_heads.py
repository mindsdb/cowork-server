"""merge the conversation folders and content reports heads

``7b2d4f6a8c1e`` (conversation folders) and ``a5c8721e09dd`` (content
reports) both branched off ``f6a2c4e8b1d3``, so ``alembic upgrade head`` errors
out. This empty revision rejoins them; each creates only its own table, so
apply order is irrelevant and neither branch's down_revision changes.

Revision ID: fabea0eac5e2
Revises: 7b2d4f6a8c1e, a5c8721e09dd
Create Date: 2026-11-20 00:00:00.000000

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = 'fabea0eac5e2'
down_revision: Union[str, Sequence[str], None] = ('7b2d4f6a8c1e', 'a5c8721e09dd')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
