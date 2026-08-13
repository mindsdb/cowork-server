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


# Tables whose rows reference a project id. None declare ON DELETE, so a loser
# `general` row can't be deleted while children point at it (FK violation aborts
# the whole upgrade on Postgres; SQLite doesn't enforce FKs, so tests miss it).
# Repoint children to the surviving row first. anton_project_id is nullable, the
# rest are NOT NULL — the UPDATE never introduces a NULL either way.
_CHILD_FKS = (
    ("conversations", "project_id"),
    ("schedules", "project_id"),
    ("task_objects", "project_id"),
    ("channel_bindings", "anton_project_id"),
)


def upgrade() -> None:
    conn = op.get_bind()
    # Collapse any duplicate `general` rows a pre-index deployment created,
    # keeping the OLDEST per org (same winner the index would have kept) so
    # existing conversations keep resolving to it. Done in Python so the
    # repoint-then-delete is identical on Postgres and SQLite.
    rows = conn.execute(
        sa.text(
            "SELECT id, coalesce(org_id, '') AS okey FROM projects "
            "WHERE name = 'general' ORDER BY okey, created_at, id"
        )
    ).fetchall()
    winner_by_org: dict[str, object] = {}
    remap: dict[object, object] = {}  # loser id -> surviving id
    for row in rows:
        okey = row.okey
        if okey not in winner_by_org:
            winner_by_org[okey] = row.id
        elif row.id != winner_by_org[okey]:
            remap[row.id] = winner_by_org[okey]

    for loser, winner in remap.items():
        for table, col in _CHILD_FKS:
            conn.execute(
                sa.text(f"UPDATE {table} SET {col} = :w WHERE {col} = :l"),
                {"w": winner, "l": loser},
            )
        conn.execute(sa.text("DELETE FROM projects WHERE id = :l"), {"l": loser})

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
