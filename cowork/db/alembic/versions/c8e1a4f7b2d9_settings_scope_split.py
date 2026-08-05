"""settings scope split

Replace the global UNIQUE(settings.key) with per-scope uniqueness: one row per
key for the deployment (scope NULL), per org, and per (org, user). A user is
identified by (org_id, user_id) — the same person in two orgs must not share a
row — so the user index carries org_id.

init (93375a6617f4) created BOTH a table-level UNIQUE(key) constraint AND a
unique ix_settings_key index; both must go. SQLite has the constraint inline in
the table, so it's removed via a batch rebuild (which also stamps the row-shape
CHECK); Postgres drops the reflected constraint directly.

A CHECK pins the three valid row shapes so a NULL owner or a bogus scope string
can't slip past the partial (filtered) indexes. Existing rows are all NULL-scope
with NULL owners, so they satisfy the global shape unchanged.

Downgrade restores UNIQUE(key); it PREFLIGHTS for duplicate keys and aborts
before touching anything if a real un-split is impossible, so a rejected
downgrade leaves the schema intact.

Revision ID: c8e1a4f7b2d9
Revises: b3d7f1a9c5e2
Create Date: 2026-08-05 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c8e1a4f7b2d9"
down_revision: Union[str, Sequence[str], None] = "b3d7f1a9c5e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The three valid row shapes. Kept identical to Setting.__table_args__.
_SHAPE_CHECK = (
    "(scope IS NULL AND org_id IS NULL AND user_id IS NULL) OR "
    "(scope = 'org' AND org_id IS NOT NULL AND user_id IS NULL) OR "
    "(scope = 'user' AND org_id IS NOT NULL AND user_id IS NOT NULL)"
)


def _key_unique_constraint(bind) -> tuple[bool, str | None]:
    """(exists, reflected_name) for the table-level UNIQUE(key) init created.
    On SQLite an unnamed constraint reflects with name=None but still exists."""
    for uc in sa.inspect(bind).get_unique_constraints("settings"):
        if uc["column_names"] == ["key"]:
            return True, uc["name"]
    return False, None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    # Old unique index goes first (both dialects).
    op.drop_index("ix_settings_key", table_name="settings")

    # Drop the table-level UNIQUE(key) constraint (only if present — a DB built
    # from the current model via create_all won't have it) + add the CHECK.
    has_key_uc, uq_name = _key_unique_constraint(bind)
    if bind.dialect.name == "sqlite":
        # init created it UNNAMED; SQLite reflects name=None, which can't be
        # dropped by name. Give batch a naming convention that reproduces
        # alembic's default `uq_<table>_<col>` name, then drop by that (the
        # documented pattern for unnamed constraints).
        naming = {"uq": "uq_%(table_name)s_%(column_0_name)s"}
        with op.batch_alter_table("settings", schema=None, naming_convention=naming) as batch:
            if has_key_uc:
                batch.drop_constraint("uq_settings_key", type_="unique")
            batch.create_check_constraint("ck_settings_scope_shape", _SHAPE_CHECK)
    else:
        if has_key_uc:
            op.drop_constraint(uq_name or "settings_key_key", "settings", type_="unique")
        op.create_check_constraint("ck_settings_scope_shape", "settings", _SHAPE_CHECK)

    # Plain lookup index (reads resolve by key) + the three partial uniques.
    op.create_index("ix_settings_key", "settings", ["key"], unique=False)
    op.create_index(
        "uq_settings_key_global",
        "settings",
        ["key"],
        unique=True,
        sqlite_where=sa.text("scope IS NULL"),
        postgresql_where=sa.text("scope IS NULL"),
    )
    op.create_index(
        "uq_settings_key_org",
        "settings",
        ["key", "org_id"],
        unique=True,
        sqlite_where=sa.text("scope = 'org'"),
        postgresql_where=sa.text("scope = 'org'"),
    )
    op.create_index(
        "uq_settings_key_user",
        "settings",
        ["key", "org_id", "user_id"],
        unique=True,
        sqlite_where=sa.text("scope = 'user'"),
        postgresql_where=sa.text("scope = 'user'"),
    )


def downgrade() -> None:
    """Downgrade schema. Preflights duplicate keys and aborts (schema intact)
    before dropping anything if restoring UNIQUE(key) is impossible."""
    bind = op.get_bind()
    dupes = bind.exec_driver_sql(
        "SELECT key FROM settings GROUP BY key HAVING COUNT(*) > 1"
    ).fetchall()
    if dupes:
        raise RuntimeError(
            f"cannot downgrade settings scope split: {len(dupes)} key(s) have "
            "per-scope rows that would violate UNIQUE(key); consolidate first. "
            "Nothing was changed."
        )

    op.drop_index("uq_settings_key_user", table_name="settings")
    op.drop_index("uq_settings_key_org", table_name="settings")
    op.drop_index("uq_settings_key_global", table_name="settings")
    op.drop_index("ix_settings_key", table_name="settings")

    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("settings", schema=None) as batch:
            batch.drop_constraint("ck_settings_scope_shape", type_="check")
            batch.create_unique_constraint("uq_settings_key", ["key"])
    else:
        op.drop_constraint("ck_settings_scope_shape", "settings", type_="check")
        op.create_unique_constraint("settings_key_key", "settings", ["key"])

    op.create_index("ix_settings_key", "settings", ["key"], unique=True)
