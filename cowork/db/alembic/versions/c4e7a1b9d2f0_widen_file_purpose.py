"""widen files.purpose to 255

The attachment purpose tag is "attachment:{project}:{session}". The session
UUID is 36 chars, so the old String(64) left only ~16 chars for the project
name — a longer name (e.g. "Catana-Outbound-email") overflowed and crashed
the attachment upload with a 500. Widen the column to 255.

Revision ID: c4e7a1b9d2f0
Revises: b7c1d2e3f4a5
Create Date: 2026-06-16 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c4e7a1b9d2f0"
down_revision: Union[str, Sequence[str], None] = "b7c1d2e3f4a5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # batch mode so the ALTER works on SQLite (no native ALTER COLUMN TYPE).
    with op.batch_alter_table("files") as batch_op:
        batch_op.alter_column(
            "purpose",
            existing_type=sa.String(64),
            type_=sa.String(255),
            existing_nullable=False,
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("files") as batch_op:
        batch_op.alter_column(
            "purpose",
            existing_type=sa.String(255),
            type_=sa.String(64),
            existing_nullable=False,
        )
