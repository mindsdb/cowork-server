"""merge the artifact-locations and schedule-slot heads

``b5c8e2f4a9d1`` (artifact_locations) and ``c41f7a9e2b58`` (each cron slot
claimed with its run row, after ``a3734c9e5210``'s live-app availability
waits) both descend from ``a6e3c9f2b4d8``. Both are heads, so
``alembic upgrade head`` errors out and every caller fails, including server
startup through ``cowork/db/migrations.py``. This is an empty merge revision
that rejoins them into a single head. It touches no schema, because each
branch's own upgrade already did its work, and it rewrites neither branch's
down_revision, so databases already stamped at either head stay consistent.
The branches touch disjoint tables (artifact_locations vs
live_app_availability_waits and schedule_runs), so their apply order is
irrelevant.

Revision ID: d4e9a7c2f1b6
Revises: b5c8e2f4a9d1, c41f7a9e2b58
Create Date: 2026-10-30 18:00:00.000000

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = 'd4e9a7c2f1b6'
down_revision: Union[str, Sequence[str], None] = ('b5c8e2f4a9d1', 'c41f7a9e2b58')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
