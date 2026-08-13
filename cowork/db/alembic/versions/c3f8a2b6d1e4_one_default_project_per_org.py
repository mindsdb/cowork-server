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


def _child_fks(conn) -> list[tuple[str, str]]:
    """Every (table, column) in the live schema referencing projects.id.

    Reflected rather than hardcoded so a column added by an earlier revision
    can't be missed. No FK declares ON DELETE, so children must be repointed
    before a duplicate `general` row can be deleted (Postgres enforces this;
    SQLite doesn't, so tests won't catch a miss).
    """
    inspector = sa.inspect(conn)
    fks = []
    for table in inspector.get_table_names():
        for fk in inspector.get_foreign_keys(table):
            if fk["referred_table"] == "projects" and fk["referred_columns"] == ["id"]:
                fks.append((table, fk["constrained_columns"][0]))
    return fks


def upgrade() -> None:
    conn = op.get_bind()
    # Collapse duplicate `general` rows from pre-index deployments, keeping
    # the oldest per org so existing children keep resolving to it.
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

    child_fks = _child_fks(conn) if remap else []
    for loser, winner in remap.items():
        for table, col in child_fks:
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
