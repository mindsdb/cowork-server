"""channel installation external_account_id

Adds channel_installations.external_account_id — the pre-scope webhook-
routing key (Slack team_id, Discord application_id, WhatsApp phone_number_id, ...).
An inbound webhook has no org scope yet, so this has to be looked up before
any org context exists, and unique on its own (not per-org): two orgs must
never claim the same platform account.

Revision ID: e5b8d3f0a2c7
Revises: d4a7c2e9f1b3
Create Date: 2026-08-17 19:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e5b8d3f0a2c7"
down_revision: Union[str, Sequence[str], None] = "d4a7c2e9f1b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "channel_installations", sa.Column("external_account_id", sa.String(length=255), nullable=True)
    )
    op.create_index(
        "uq_channel_installations_external_account",
        "channel_installations",
        ["channel_type", "external_account_id"],
        unique=True,
        sqlite_where=sa.text("external_account_id IS NOT NULL"),
        postgresql_where=sa.text("external_account_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("uq_channel_installations_external_account", table_name="channel_installations")
    op.drop_column("channel_installations", "external_account_id")
