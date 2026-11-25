"""merge the schedule-run slot claim and artifact owner choice heads

``c41f7a9e2b58`` (each cron slot claimed by its run row, with a lease) and
``c5a8e2f1d7b3`` (skill sharing tables, followed by ``d9c4e7a2b5f1``, the
artifact owner choices) both branched off ``a3734c9e5210``. Both are heads,
so ``alembic upgrade head`` errors out and every caller fails, including server
startup through ``cowork/db/migrations.py``. This is an empty merge revision
that rejoins them into a single head. It touches no schema, because each
branch's own upgrade already did its work, and it rewrites neither branch's
down_revision, so databases already stamped at either head stay consistent.
The branches touch disjoint tables (schedule_runs vs skill_access,
skill_user_setting and artifact_owner_choices), so their apply order is
irrelevant.

Revision ID: 2a366c1f4b9f
Revises: c41f7a9e2b58, d9c4e7a2b5f1
Create Date: 2026-11-02 00:00:00.000000

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = '2a366c1f4b9f'
down_revision: Union[str, Sequence[str], None] = ('c41f7a9e2b58', 'd9c4e7a2b5f1')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
