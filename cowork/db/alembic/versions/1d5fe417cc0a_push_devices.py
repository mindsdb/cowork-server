"""push_devices: the phones the mobile app registered for push

One row per device token (APNs or FCM), owned by the person who registered it
in their organization. The token is unique: a phone that another person signs
in on moves to them. Org mode only; the desktop's SQLite gets the empty table
and never writes it.

Revision ID: 1d5fe417cc0a
Revises: c1d7e2a94f60
Create Date: 2026-11-06 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "1d5fe417cc0a"
down_revision: Union[str, Sequence[str], None] = "c1d7e2a94f60"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    # Guarded: a database first built from the models already has it.
    if _has_table("push_devices"):
        return
    op.create_table(
        "push_devices",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("org_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("platform", sa.String(length=16), nullable=False),
        sa.Column("token", sa.String(length=512), nullable=False),
        sa.Column("environment", sa.String(length=16), nullable=True),
        sa.Column("app_version", sa.String(length=64), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token", name="uq_push_devices_token"),
    )
    op.create_index("ix_push_devices_org_id_user_id", "push_devices", ["org_id", "user_id"])


def downgrade() -> None:
    if _has_table("push_devices"):
        op.drop_index("ix_push_devices_org_id_user_id", table_name="push_devices")
        op.drop_table("push_devices")
