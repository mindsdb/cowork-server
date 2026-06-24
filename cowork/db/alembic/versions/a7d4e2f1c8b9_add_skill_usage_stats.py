"""add skill usage stats

Adds `used` and `confidence` counters to the skills table. `used` is bumped
on each `recall_skill` invocation (see SkillService.record_use); `confidence`
is reserved for the recall classifier signal Anton tracks per skill.

Revision ID: a7d4e2f1c8b9
Revises: fbe3964c2030
Create Date: 2026-06-24 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a7d4e2f1c8b9"
down_revision: Union[str, Sequence[str], None] = "fbe3964c2030"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    """Upgrade schema."""
    if not _has_column("skills", "used"):
        op.add_column(
            "skills",
            sa.Column("used", sa.Integer(), nullable=False, server_default="0"),
        )
    if not _has_column("skills", "confidence"):
        op.add_column(
            "skills",
            sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("skills") as batch_op:
        if _has_column("skills", "confidence"):
            batch_op.drop_column("confidence")
        if _has_column("skills", "used"):
            batch_op.drop_column("used")
