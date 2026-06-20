"""project invitations

Revision ID: f1c2d3e4a5b6
Revises: e8b2c4d6a9f0
Create Date: 2026-06-20 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f1c2d3e4a5b6"
down_revision: Union[str, Sequence[str], None] = "e8b2c4d6a9f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _has_index(table_name: str, index_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return index_name in {index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_table("project_invitations"):
        op.create_table(
            "project_invitations",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("project_id", sa.Uuid(), nullable=False),
            sa.Column("email", sa.String(length=255), nullable=False),
            sa.Column("display_name", sa.String(length=255), nullable=True),
            sa.Column("role", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=64), nullable=False),
            sa.Column("token_hash", sa.String(length=64), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_sent_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("send_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("invited_by_subject", sa.String(length=255), nullable=True),
            sa.Column("invited_by_email", sa.String(length=255), nullable=True),
            sa.Column("invited_by_name", sa.String(length=255), nullable=True),
            sa.Column("accepted_by_subject", sa.String(length=255), nullable=True),
            sa.Column("accepted_by_email", sa.String(length=255), nullable=True),
            sa.Column("accepted_by_name", sa.String(length=255), nullable=True),
            sa.Column("notification_state", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_project_invitations_project_id"), "project_invitations", ["project_id"], unique=False)
        op.create_index(op.f("ix_project_invitations_email"), "project_invitations", ["email"], unique=False)
        op.create_index(op.f("ix_project_invitations_status"), "project_invitations", ["status"], unique=False)
        op.create_index(op.f("ix_project_invitations_token_hash"), "project_invitations", ["token_hash"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    if _has_table("project_invitations"):
        for index_name in (
            op.f("ix_project_invitations_token_hash"),
            op.f("ix_project_invitations_status"),
            op.f("ix_project_invitations_email"),
            op.f("ix_project_invitations_project_id"),
        ):
            if _has_index("project_invitations", index_name):
                op.drop_index(index_name, table_name="project_invitations")
        op.drop_table("project_invitations")
