"""identity and artifact shares

Revision ID: a1b2c3d4e5f6
Revises: f1c2d3e4a5b6
Create Date: 2026-06-21 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "f1c2d3e4a5b6"
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
    if not _has_table("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("email", sa.String(length=255), nullable=False),
            sa.Column("display_name", sa.String(length=255), nullable=True),
            sa.Column("sso_subject", sa.String(length=255), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("email", name="uq_users_email"),
        )
        op.create_index(op.f("ix_users_email"), "users", ["email"], unique=False)
        op.create_index(op.f("ix_users_sso_subject"), "users", ["sso_subject"], unique=False)

    if not _has_table("artifact_shares"):
        op.create_table(
            "artifact_shares",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("artifact_id", sa.Uuid(), nullable=False),
            sa.Column("grantee_email", sa.String(length=255), nullable=False),
            sa.Column("role", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=64), nullable=False),
            sa.Column("token_hash", sa.String(length=64), nullable=False),
            sa.Column("created_by", sa.String(length=255), nullable=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("accepted_user_id", sa.Uuid(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=True),
            sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
            sa.ForeignKeyConstraint(["accepted_user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_artifact_shares_artifact_id"), "artifact_shares", ["artifact_id"], unique=False)
        op.create_index(op.f("ix_artifact_shares_grantee_email"), "artifact_shares", ["grantee_email"], unique=False)
        op.create_index(op.f("ix_artifact_shares_status"), "artifact_shares", ["status"], unique=False)
        op.create_index(op.f("ix_artifact_shares_token_hash"), "artifact_shares", ["token_hash"], unique=False)
        op.create_index(op.f("ix_artifact_shares_accepted_user_id"), "artifact_shares", ["accepted_user_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    if _has_table("artifact_shares"):
        for index_name in (
            op.f("ix_artifact_shares_accepted_user_id"),
            op.f("ix_artifact_shares_token_hash"),
            op.f("ix_artifact_shares_status"),
            op.f("ix_artifact_shares_grantee_email"),
            op.f("ix_artifact_shares_artifact_id"),
        ):
            if _has_index("artifact_shares", index_name):
                op.drop_index(index_name, table_name="artifact_shares")
        op.drop_table("artifact_shares")

    if _has_table("users"):
        for index_name in (
            op.f("ix_users_sso_subject"),
            op.f("ix_users_email"),
        ):
            if _has_index("users", index_name):
                op.drop_index(index_name, table_name="users")
        op.drop_table("users")
