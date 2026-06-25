"""direction a artifact metadata

Revision ID: 9b7c6d5e4f3a
Revises: f1c2d3e4a5b6
Create Date: 2026-06-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9b7c6d5e4f3a"
down_revision: Union[str, Sequence[str], None] = "f1c2d3e4a5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _columns(table_name: str) -> set[str]:
    if not _has_table(table_name):
        return set()
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def _has_index(table_name: str, index_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return index_name in {index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table_name)}


def _has_foreign_key(table_name: str, constraint_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return constraint_name in {
        foreign_key.get("name")
        for foreign_key in sa.inspect(op.get_bind()).get_foreign_keys(table_name)
    }


def upgrade() -> None:
    """Upgrade schema."""
    version_cols = _columns("artifact_versions")
    if version_cols:
        fk_name = "fk_artifact_versions_pre_snapshot_version_id"
        with op.batch_alter_table("artifact_versions") as batch_op:
            if "snapshot_role" not in version_cols:
                batch_op.add_column(sa.Column("snapshot_role", sa.String(length=64), nullable=True))
            if "pre_snapshot_version_id" not in version_cols:
                batch_op.add_column(sa.Column("pre_snapshot_version_id", sa.Uuid(), nullable=True))
            if not _has_foreign_key("artifact_versions", fk_name):
                batch_op.create_foreign_key(
                    fk_name,
                    "artifact_versions",
                    ["pre_snapshot_version_id"],
                    ["id"],
                )
        if not _has_index("artifact_versions", op.f("ix_artifact_versions_snapshot_role")):
            op.create_index(op.f("ix_artifact_versions_snapshot_role"), "artifact_versions", ["snapshot_role"], unique=False)
        if not _has_index("artifact_versions", op.f("ix_artifact_versions_pre_snapshot_version_id")):
            op.create_index(
                op.f("ix_artifact_versions_pre_snapshot_version_id"),
                "artifact_versions",
                ["pre_snapshot_version_id"],
                unique=False,
            )

    comment_cols = _columns("artifact_comments")
    if comment_cols:
        with op.batch_alter_table("artifact_comments") as batch_op:
            if "review_verdict" not in comment_cols:
                batch_op.add_column(sa.Column("review_verdict", sa.String(length=64), nullable=True))
        if not _has_index("artifact_comments", op.f("ix_artifact_comments_review_verdict")):
            op.create_index(op.f("ix_artifact_comments_review_verdict"), "artifact_comments", ["review_verdict"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    if _has_table("artifact_comments"):
        if _has_index("artifact_comments", op.f("ix_artifact_comments_review_verdict")):
            op.drop_index(op.f("ix_artifact_comments_review_verdict"), table_name="artifact_comments")
        if "review_verdict" in _columns("artifact_comments"):
            with op.batch_alter_table("artifact_comments") as batch_op:
                batch_op.drop_column("review_verdict")

    if _has_table("artifact_versions"):
        for index_name in (
            op.f("ix_artifact_versions_pre_snapshot_version_id"),
            op.f("ix_artifact_versions_snapshot_role"),
        ):
            if _has_index("artifact_versions", index_name):
                op.drop_index(index_name, table_name="artifact_versions")
        version_cols = _columns("artifact_versions")
        with op.batch_alter_table("artifact_versions") as batch_op:
            if "pre_snapshot_version_id" in version_cols:
                if _has_foreign_key("artifact_versions", "fk_artifact_versions_pre_snapshot_version_id"):
                    batch_op.drop_constraint("fk_artifact_versions_pre_snapshot_version_id", type_="foreignkey")
                batch_op.drop_column("pre_snapshot_version_id")
            if "snapshot_role" in version_cols:
                batch_op.drop_column("snapshot_role")
