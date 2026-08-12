"""One default project per org

`projects` has no unique constraint, so the "insert the org's default unless it
already has one" guard was only safe on SQLite, which serialises writes. On
Postgres two replicas can both pass the NOT EXISTS at READ COMMITTED and both
insert, leaving an org with two default projects (review finding).

A partial unique index scoped to `name = 'general'` makes the database the
arbiter: the loser of the race gets an IntegrityError and adopts the winner's
row. Deliberately narrow — projects are not unique by name in general (each org
may keep its own naming, and `_unique_name` only dedupes per scope), so a full
`(org_id, name)` constraint could reject legitimate existing rows.

Revision ID: c3f8a2b6d1e4
Revises: b7e1d4c9a2f6
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c3f8a2b6d1e4"
down_revision: Union[str, Sequence[str], None] = "b7e1d4c9a2f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX = "uq_projects_default_per_org"
# org_id is NULL on desktop/legacy rows; coalesce so those collapse to one key
# too (a local install has exactly one seeded `general`).
_WHERE = sa.text("name = 'general'")


def upgrade() -> None:
    # Collapse any duplicates a pre-index deployment already created, keeping the
    # oldest row so existing conversations keep resolving to it.
    op.execute(
        sa.text(
            """
            DELETE FROM projects
            WHERE name = 'general'
              AND id NOT IN (
                SELECT id FROM (
                  SELECT id,
                         ROW_NUMBER() OVER (
                           PARTITION BY coalesce(org_id, '') ORDER BY created_at, id
                         ) AS rn
                  FROM projects WHERE name = 'general'
                ) ranked
                WHERE rn = 1
              )
            """
        )
    )
    op.create_index(
        INDEX,
        "projects",
        [sa.text("coalesce(org_id, '')")],
        unique=True,
        sqlite_where=_WHERE,
        postgresql_where=_WHERE,
    )


def downgrade() -> None:
    op.drop_index(INDEX, table_name="projects")
