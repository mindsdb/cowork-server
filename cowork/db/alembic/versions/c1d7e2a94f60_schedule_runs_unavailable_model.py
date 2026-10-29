"""schedule_runs: unavailable_model

A task can pin a model. When the pin can't be served, the run uses the
account default instead of failing, and records the pinned id here so the
app can say which model it couldn't use.

Revision ID: c1d7e2a94f60
Revises: 8ffc4296fcbe
Create Date: 2026-10-29 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "c1d7e2a94f60"
down_revision: Union[str, Sequence[str], None] = "8ffc4296fcbe"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Guarded: a database first built from the models (pre-Alembic desktops)
    already has it.
    """
    if "unavailable_model" not in {c["name"] for c in sa.inspect(op.get_bind()).get_columns("schedule_runs")}:
        op.add_column("schedule_runs", sa.Column("unavailable_model", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("schedule_runs") as batch_op:
        batch_op.drop_column("unavailable_model")
