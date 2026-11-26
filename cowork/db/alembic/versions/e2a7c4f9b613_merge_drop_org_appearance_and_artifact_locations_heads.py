"""merge the drop-org-appearance and artifact locations heads

``d16c08651013`` (drops org_appearance now that auth owns the org look,
ENG-4020) and ``f93151b5bad1`` (staging's merge of the conversations owner
index and the artifact location/queued repairs chain) both descend from
``5b9e1c7d3a20``'s ancestry. Two heads make ``alembic upgrade head`` fail
everywhere, server startup included, so this empty revision rejoins them.
It touches no schema and rewrites neither branch's down_revision, so a
database stamped at either head (this branch's PR env ran
``d16c08651013``; staging ran ``f93151b5bad1``) stays consistent. The
branches touch disjoint tables, so their order is irrelevant.

Revision ID: e2a7c4f9b613
Revises: d16c08651013, f93151b5bad1
Create Date: 2026-11-09 00:00:00.000000

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "e2a7c4f9b613"
down_revision: Union[str, Sequence[str], None] = ("d16c08651013", "f93151b5bad1")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
