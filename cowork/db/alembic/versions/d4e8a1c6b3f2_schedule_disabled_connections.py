"""schedules: disabled_connections

A scheduled task can leave out connections it doesn't need, so an unattended
run can't use them: the same {engine, name} list a chat turn mutes them with.

Revision ID: d4e8a1c6b3f2
Revises: c271568dc904
Create Date: 2026-11-23 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "d4e8a1c6b3f2"
down_revision: Union[str, Sequence[str], None] = "c271568dc904"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Guarded: a database first built from the models already has it.
    """
    if "disabled_connections" not in {c["name"] for c in sa.inspect(op.get_bind()).get_columns("schedules")}:
        op.add_column("schedules", sa.Column("disabled_connections", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Pause tasks whose connector exclusions the older server cannot enforce."""
    connection = op.get_bind()
    if "disabled_connections" not in {c["name"] for c in sa.inspect(connection).get_columns("schedules")}:
        return
    schedules = sa.table(
        "schedules",
        sa.column("disabled_connections", sa.JSON()),
        sa.column("enabled", sa.Boolean()),
        sa.column("paused_at", sa.DateTime(timezone=True)),
    )
    disabled = schedules.c.disabled_connections
    json_type = sa.func.json_type if connection.dialect.name == "sqlite" else sa.func.json_typeof
    has_exclusions = sa.case(
        (json_type(disabled) == "array", sa.func.json_array_length(disabled) > 0),
        else_=False,
    )
    op.execute(
        schedules.update()
        .where(has_exclusions)
        .values(enabled=False, paused_at=sa.func.coalesce(schedules.c.paused_at, sa.func.current_timestamp()))
    )
    op.drop_column("schedules", "disabled_connections")
