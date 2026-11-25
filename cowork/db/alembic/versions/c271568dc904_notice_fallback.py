"""Store fallback summaries and destinations for unfamiliar notice kinds."""

import sqlalchemy as sa
from alembic import op

revision = "c271568dc904"
down_revision = "7e93b26a4c18"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "fallback" not in {column["name"] for column in sa.inspect(op.get_bind()).get_columns("notices")}:
        op.add_column("notices", sa.Column("fallback", sa.JSON(), nullable=True))


def downgrade() -> None:
    if "fallback" in {column["name"] for column in sa.inspect(op.get_bind()).get_columns("notices")}:
        op.drop_column("notices", "fallback")
