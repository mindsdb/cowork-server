"""merge the artifact-locations/schedule-slot head and the queued agent repairs head

``d4e9a7c2f1b6`` rejoins ``b5c8e2f4a9d1`` (artifact_locations) and
``c41f7a9e2b58`` (each cron slot claimed with its run row).
``c7d1e3f5a8b2`` (queued_agent_repairs) also descends from ``b5c8e2f4a9d1``,
so both are heads, and ``alembic upgrade head`` errors out and every caller
fails, including server startup through ``cowork/db/migrations.py``. This is
an empty merge revision that rejoins them into a single head. It touches no
schema, because each branch's own upgrade already did its work, and it
rewrites neither branch's down_revision, so databases already stamped at
either head stay consistent. The branches touch disjoint tables
(live_app_availability_waits and schedule_runs vs queued_agent_repairs), so
their apply order is irrelevant.

Revision ID: e3b9d5f7a2c4
Revises: d4e9a7c2f1b6, c7d1e3f5a8b2
Create Date: 2026-10-31 07:21:55.000000

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = 'e3b9d5f7a2c4'
down_revision: Union[str, Sequence[str], None] = ('d4e9a7c2f1b6', 'c7d1e3f5a8b2')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
