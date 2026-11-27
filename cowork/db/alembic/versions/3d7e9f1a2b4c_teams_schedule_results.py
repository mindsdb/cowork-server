"""Teams: post a scheduled run's result to the owner's Teams chat

schedules.deliver_to_teams turns it on per schedule, schedule_runs.teams_delivery
says how posting went, and teams_contacts keeps where to reach each person in
Teams, as the self-hosted recording proxy pairs them. Existing schedules don't
post; existing runs have no delivery.

Revision ID: 3d7e9f1a2b4c
Revises: a7c3e9f1b5d2
Create Date: 2026-11-24 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3d7e9f1a2b4c"
down_revision: Union[str, Sequence[str], None] = "a7c3e9f1b5d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    """Upgrade schema. Each step is skipped where a database created from the models has it."""
    if "deliver_to_teams" not in _columns("schedules"):
        with op.batch_alter_table("schedules") as batch:
            batch.add_column(sa.Column("deliver_to_teams", sa.Boolean(), nullable=False, server_default=sa.false()))
    if "teams_delivery" not in _columns("schedule_runs"):
        with op.batch_alter_table("schedule_runs") as batch:
            batch.add_column(sa.Column("teams_delivery", sa.String(length=16), nullable=True))
    if "teams_contacts" not in sa.inspect(op.get_bind()).get_table_names():
        op.create_table(
            "teams_contacts",
            sa.Column("user_id", sa.String(length=36), primary_key=True),
            sa.Column("tenant_id", sa.String(length=64), nullable=False),
            sa.Column("aad_object_id", sa.String(length=64), nullable=False),
            sa.Column("service_url", sa.String(length=512), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )


def downgrade() -> None:
    """Downgrade schema. Forgets every Teams contact and delivery state; schedules stop posting."""
    if "teams_contacts" in sa.inspect(op.get_bind()).get_table_names():
        op.drop_table("teams_contacts")
    # Plain drops, not batch copies: SQLite drops a column in place, so a downgrade that fails in an
    # older revision leaves no copied table behind to break the next attempt.
    if "teams_delivery" in _columns("schedule_runs"):
        op.drop_column("schedule_runs", "teams_delivery")
    if "deliver_to_teams" in _columns("schedules"):
        op.drop_column("schedules", "deliver_to_teams")
