"""merge shared-resource attribution and code-control/channel heads

``e2a6c4f8b1d3`` (shared-resource attribution and mutation audit) branched off
``a4c8e1f6b3d9`` while ``cfbc79856e9e`` rejoined the code-control and channel
org-scoping branches off the same ancestor. Both are heads, so
``alembic upgrade head`` errors out and every caller fails, including server
startup through ``cowork/db/migrations.py``. This is an empty merge revision
that rejoins them into a single head. It touches no schema, because each
branch's own upgrade already did its work, and it rewrites neither branch's
down_revision, so databases already stamped at either head stay consistent.
The branches touch disjoint tables (shared_resource_attributions and
shared_resource_mutations vs code_control_records, channel_installations and
channel_events), so their apply order is irrelevant.

Revision ID: b7f4d2c9a3e1
Revises: cfbc79856e9e, e2a6c4f8b1d3
Create Date: 2026-09-03 00:00:00.000000

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = 'b7f4d2c9a3e1'
down_revision: Union[str, Sequence[str], None] = ('cfbc79856e9e', 'e2a6c4f8b1d3')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
