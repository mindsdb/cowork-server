"""merge main's conversations index head into staging's artifact repairs head

``599bf4bf0370`` (main's hotfix, off ``2a366c1f4b9f``) and ``dce94d4bafc0``
(the same change on staging, off ``6f1b8d3a5c27``, followed by
``b5c8e2f4a9d1`` and ``c7d1e3f5a8b2``, the artifact location rows and the
queued repairs) both create ``ix_conversations_org_id_created_by``. Syncing
main into staging leaves ``599bf4bf0370`` and ``c7d1e3f5a8b2`` as heads, so
``alembic upgrade head`` errors out and every caller fails, including server
startup through ``cowork/db/migrations.py``. This is an empty merge revision
that rejoins them into a single head. It touches no schema, and it rewrites
neither branch's down_revision, so a database stamped at either head stays
consistent: one that ran main's hotfix applies staging's chain, and staging
applies ``599bf4bf0370``. Both create the index only when it is missing, so
whichever runs second leaves it as it is.

Revision ID: f93151b5bad1
Revises: c7d1e3f5a8b2, 599bf4bf0370
Create Date: 2026-11-08 00:00:00.000000

"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = 'f93151b5bad1'
down_revision: Union[str, Sequence[str], None] = ('c7d1e3f5a8b2', '599bf4bf0370')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
