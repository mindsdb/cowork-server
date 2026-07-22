"""pins: per-user, per-org uniqueness

The init schema's UNIQUE(item_type, item_id) is table-global — the second org
member pinning the same item hits an IntegrityError. Replace it with a unique
index on (item_type, item_id, COALESCE(user_id, ''), COALESCE(org_id, '')).
org_id is part of the boundary because neither identifier is globally unique:
users belong to many orgs, and item_id is client-supplied (project pins use
the name, unique only per org). Desktop rows (both NULL) collapse to '' and
keep the pre-tenancy one-pin-per-item guarantee.

Revision ID: d2e8f1a4c7b9
Revises: a1c3e5f7b9d2
Create Date: 2026-07-22 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d2e8f1a4c7b9"
down_revision: Union[str, Sequence[str], None] = "a1c3e5f7b9d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

UNIQUE_INDEX = "uq_pins_item_user"
_UNIQUE_EXPR = [
    "item_type",
    "item_id",
    sa.text("coalesce(user_id, '')"),
    sa.text("coalesce(org_id, '')"),
]


def _pins_table(*, with_item_unique: bool) -> sa.Table:
    """The pins table as it stands at this revision, for SQLite rebuilds.

    The init migration's UNIQUE(item_type, item_id) is an unnamed inline
    constraint, which SQLite can only shed (or restore) via a table rebuild —
    batch mode needs an explicit ``copy_from`` definition to do that.
    """
    args = [
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("item_type", sa.String(64), nullable=False),
        sa.Column("item_id", sa.String(64), nullable=False),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.Column(
            "modified_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.Column("user_id", sa.String(36), nullable=True),
        sa.Column("org_id", sa.String(36), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.Index("ix_pins_user_id_org_id", "user_id", "org_id"),
    ]
    if with_item_unique:
        args.append(sa.UniqueConstraint("item_type", "item_id"))
    return sa.Table("pins", sa.MetaData(), *args)


def _item_unique_name(bind) -> str:
    """Reflected name of UNIQUE(item_type, item_id) — backends auto-name it."""
    for uc in sa.inspect(bind).get_unique_constraints("pins"):
        if sorted(uc["column_names"]) == ["item_id", "item_type"]:
            return uc["name"]
    raise RuntimeError("pins UNIQUE(item_type, item_id) not found; unexpected schema")


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table(
            "pins", copy_from=_pins_table(with_item_unique=False), recreate="always"
        ):
            pass
    else:
        op.drop_constraint(_item_unique_name(bind), "pins", type_="unique")
    op.create_index(UNIQUE_INDEX, "pins", _UNIQUE_EXPR, unique=True)


def downgrade() -> None:
    """Downgrade schema. Fails on data only a multi-user org can produce
    (the same item pinned by two users) — single-user databases are clean."""
    op.drop_index(UNIQUE_INDEX, table_name="pins")
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table(
            "pins", copy_from=_pins_table(with_item_unique=True), recreate="always"
        ):
            pass
    else:
        op.create_unique_constraint("pins_item_type_item_id_key", "pins", ["item_type", "item_id"])
