"""task_objects: index of artifacts/files a task owns

Lets a task (conversation) be moved to another project together with the
artifacts and files it produced. Each row maps an object (artifact slug or
file id) to its owning conversation and current project.

Revision ID: d5f3a8c1e6b2
Revises: c4e7a1b9d2f0
Create Date: 2026-06-18 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d5f3a8c1e6b2"
down_revision: Union[str, Sequence[str], None] = "c4e7a1b9d2f0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(table_name: str) -> bool:
    return table_name in sa.inspect(op.get_bind()).get_table_names()


def _has_index(table_name: str, index_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return index_name in {index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table_name)}


def upgrade() -> None:
    """Upgrade schema.

    Guarded with existence checks so re-running over a database that already
    has the table (e.g. a pre-Alembic schema baselined via
    ``SQLModel.metadata.create_all``) is a no-op — matching the idempotent
    pattern the other cowork migrations use.
    """
    if not _has_table("task_objects"):
        op.create_table(
            "task_objects",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("conversation_id", sa.Uuid(), nullable=False),
            sa.Column("project_id", sa.Uuid(), nullable=False),
            sa.Column("kind", sa.String(length=16), nullable=False),
            sa.Column("ref", sa.String(length=255), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            sa.Column(
                "modified_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=True,
            ),
            sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
            sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
    if not _has_index("task_objects", "ix_task_objects_conversation_id"):
        op.create_index("ix_task_objects_conversation_id", "task_objects", ["conversation_id"])
    if not _has_index("task_objects", "ix_task_objects_project_id"):
        op.create_index("ix_task_objects_project_id", "task_objects", ["project_id"])
    if not _has_index("task_objects", "ix_task_objects_ref"):
        op.create_index("ix_task_objects_ref", "task_objects", ["ref"])


def downgrade() -> None:
    """Downgrade schema."""
    if _has_index("task_objects", "ix_task_objects_ref"):
        op.drop_index("ix_task_objects_ref", table_name="task_objects")
    if _has_index("task_objects", "ix_task_objects_project_id"):
        op.drop_index("ix_task_objects_project_id", table_name="task_objects")
    if _has_index("task_objects", "ix_task_objects_conversation_id"):
        op.drop_index("ix_task_objects_conversation_id", table_name="task_objects")
    if _has_table("task_objects"):
        op.drop_table("task_objects")
