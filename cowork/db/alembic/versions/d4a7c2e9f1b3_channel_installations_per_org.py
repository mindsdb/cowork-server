"""channel installations per org

Replace the global UniqueConstraint(channel_installations.channel_type) with
per-scope uniqueness: one row per channel_type for the deployment (org_id
NULL, desktop/local — today's behavior verbatim) or per (channel_type,
org_id) — org-wide installations, not per-member, matching how provider
credentials are already scoped. Mirrors c8e1a4f7b2d9 (settings scope split).

channel_events gains org_id as its installation anchor (the module docstring
in cowork/services/channel_events.py already flagged this as the follow-up
this migration needed to land in) and its inbound-dedupe index splits the
same way, so one org's redelivered event id can never dedupe against
another's.

Revision ID: d4a7c2e9f1b3
Revises: c3f8a2b6d1e4
Create Date: 2026-08-17 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4a7c2e9f1b3"
down_revision: Union[str, Sequence[str], None] = "c3f8a2b6d1e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()

    # channel_installations: drop the flat unique constraint (SQLite needs a
    # batch rebuild to drop a named constraint; Postgres drops it directly),
    # add the local/org partial-index split.
    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("channel_installations", schema=None) as batch:
            batch.drop_constraint("uq_channel_installations_type", type_="unique")
    else:
        op.drop_constraint("uq_channel_installations_type", "channel_installations", type_="unique")

    op.create_index(
        "uq_channel_installations_type_global",
        "channel_installations",
        ["channel_type"],
        unique=True,
        sqlite_where=sa.text("org_id IS NULL"),
        postgresql_where=sa.text("org_id IS NULL"),
    )
    op.create_index(
        "uq_channel_installations_type_org",
        "channel_installations",
        ["channel_type", "org_id"],
        unique=True,
        sqlite_where=sa.text("org_id IS NOT NULL"),
        postgresql_where=sa.text("org_id IS NOT NULL"),
    )

    # channel_events: add the installation anchor + split dedupe the same way.
    op.add_column("channel_events", sa.Column("org_id", sa.String(length=36), nullable=True))
    op.create_index(op.f("ix_channel_events_org_id"), "channel_events", ["org_id"], unique=False)

    op.drop_index("uq_channel_events_inbound_dedupe", table_name="channel_events")
    op.create_index(
        "uq_channel_events_inbound_dedupe_global",
        "channel_events",
        ["channel_type", "dedupe_key"],
        unique=True,
        sqlite_where=sa.text("direction = 'inbound' AND dedupe_key IS NOT NULL AND org_id IS NULL"),
        postgresql_where=sa.text("direction = 'inbound' AND dedupe_key IS NOT NULL AND org_id IS NULL"),
    )
    op.create_index(
        "uq_channel_events_inbound_dedupe_org",
        "channel_events",
        ["channel_type", "org_id", "dedupe_key"],
        unique=True,
        sqlite_where=sa.text("direction = 'inbound' AND dedupe_key IS NOT NULL AND org_id IS NOT NULL"),
        postgresql_where=sa.text("direction = 'inbound' AND dedupe_key IS NOT NULL AND org_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Downgrade schema. Preflights rows that would violate the old
    single-installation-per-channel_type constraint and aborts (schema
    intact) if un-splitting is impossible."""
    bind = op.get_bind()
    dupes = bind.exec_driver_sql(
        "SELECT channel_type FROM channel_installations GROUP BY channel_type HAVING COUNT(*) > 1"
    ).fetchall()
    if dupes:
        raise RuntimeError(
            f"cannot downgrade channel installations per org: {len(dupes)} channel_type(s) "
            "have more than one installation, which would violate the single global "
            "UniqueConstraint(channel_type); consolidate first. Nothing was changed."
        )

    op.drop_index("uq_channel_events_inbound_dedupe_org", table_name="channel_events")
    op.drop_index("uq_channel_events_inbound_dedupe_global", table_name="channel_events")
    op.create_index(
        "uq_channel_events_inbound_dedupe",
        "channel_events",
        ["channel_type", "dedupe_key"],
        unique=True,
        sqlite_where=sa.text("direction = 'inbound' AND dedupe_key IS NOT NULL"),
        postgresql_where=sa.text("direction = 'inbound' AND dedupe_key IS NOT NULL"),
    )
    op.drop_index(op.f("ix_channel_events_org_id"), table_name="channel_events")
    op.drop_column("channel_events", "org_id")

    op.drop_index("uq_channel_installations_type_org", table_name="channel_installations")
    op.drop_index("uq_channel_installations_type_global", table_name="channel_installations")

    if bind.dialect.name == "sqlite":
        with op.batch_alter_table("channel_installations", schema=None) as batch:
            batch.create_unique_constraint("uq_channel_installations_type", ["channel_type"])
    else:
        op.create_unique_constraint(
            "uq_channel_installations_type", "channel_installations", ["channel_type"]
        )
