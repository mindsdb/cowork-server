"""live apps: launch records, shares and copies

Revision ID: f3b9d6a2c8e4
Revises: 3e4b5f7586d3
Create Date: 2026-10-21 05:00:00.000000

The three tables cowork.models.live_app declares. Only a deployment with live
apps on writes them; elsewhere they stay empty.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "f3b9d6a2c8e4"
down_revision: Union[str, Sequence[str], None] = "3e4b5f7586d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _base_columns() -> list[sa.Column]:
    """BaseSQLModel's id and timestamps."""
    return [
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    ]


def upgrade() -> None:
    # Bootstrap can create SQLModel metadata before Alembic adopts the schema.
    if not _has_table("live_app_launches"):
        op.create_table(
            "live_app_launches",
            *_base_columns(),
            sa.Column("slot", sa.Integer(), nullable=False),
            sa.Column("launcher_user_id", sa.String(length=36), nullable=False),
            sa.Column("launcher_partition_id", sa.String(length=36), nullable=False),
            sa.Column("launcher_email", sa.String(length=254), nullable=False),
            sa.Column("project_id", sa.Uuid(), nullable=False),
            sa.Column("slug", sa.String(length=255), nullable=False),
            sa.Column("artifact_dir", sa.String(length=4096), nullable=False),
            sa.Column("address", sa.String(length=512), nullable=False),
            sa.Column("boot_id", sa.String(length=64), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("state", sa.String(length=16), nullable=False),
            sa.Column("stopped_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("stop_reason", sa.String(length=16), nullable=True),
            sa.Column("limit_hours", sa.Float(), nullable=True),
            sa.Column("stop_noticed", sa.Boolean(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "uq_live_app_launches_running_slot",
            "live_app_launches",
            ["slot"],
            unique=True,
            sqlite_where=sa.text("state = 'running'"),
            postgresql_where=sa.text("state = 'running'"),
        )
        op.create_index(
            "ix_live_app_launches_launcher",
            "live_app_launches",
            ["launcher_partition_id", "launcher_user_id"],
        )
    if not _has_table("live_app_shares"):
        op.create_table(
            "live_app_shares",
            *_base_columns(),
            sa.Column("owner_partition_id", sa.String(length=36), nullable=False),
            sa.Column("owner_user_id", sa.String(length=36), nullable=False),
            sa.Column("owner_email", sa.String(length=254), nullable=False),
            sa.Column("project_id", sa.Uuid(), nullable=False),
            sa.Column("slug", sa.String(length=255), nullable=False),
            sa.Column("recipient_email", sa.String(length=254), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "owner_partition_id",
                "project_id",
                "slug",
                "recipient_email",
                name="uq_live_app_shares_app_recipient",
            ),
        )
        op.create_index(
            "uq_live_app_shares_app_everyone",
            "live_app_shares",
            ["owner_partition_id", "project_id", "slug"],
            unique=True,
            sqlite_where=sa.text("recipient_email IS NULL"),
            postgresql_where=sa.text("recipient_email IS NULL"),
        )
        op.create_index("ix_live_app_shares_recipient_email", "live_app_shares", ["recipient_email"])
    if not _has_table("live_app_copies"):
        op.create_table(
            "live_app_copies",
            *_base_columns(),
            sa.Column("partition_id", sa.String(length=36), nullable=False),
            sa.Column("project_id", sa.Uuid(), nullable=False),
            sa.Column("slug", sa.String(length=255), nullable=False),
            sa.Column("owner_email", sa.String(length=254), nullable=False),
            sa.Column("share_id", sa.Uuid(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("partition_id", "project_id", "slug", name="uq_live_app_copies_artifact"),
        )


def downgrade() -> None:
    for table_name in ("live_app_copies", "live_app_shares", "live_app_launches"):
        if _has_table(table_name):
            op.drop_table(table_name)
