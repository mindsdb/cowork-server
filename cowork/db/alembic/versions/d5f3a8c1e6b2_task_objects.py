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


def upgrade() -> None:
    """Upgrade schema."""
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
    op.create_index("ix_task_objects_conversation_id", "task_objects", ["conversation_id"])
    op.create_index("ix_task_objects_project_id", "task_objects", ["project_id"])
    op.create_index("ix_task_objects_ref", "task_objects", ["ref"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_task_objects_ref", table_name="task_objects")
    op.drop_index("ix_task_objects_project_id", table_name="task_objects")
    op.drop_index("ix_task_objects_conversation_id", table_name="task_objects")
    op.drop_table("task_objects")
