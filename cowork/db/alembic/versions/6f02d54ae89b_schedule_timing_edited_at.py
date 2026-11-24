"""Record explicit edits to schedule timing."""

from alembic import op
import sqlalchemy as sa

revision = "6f02d54ae89b"
down_revision = "fabea0eac5e2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A database created from current models already has the column.
    columns = sa.inspect(op.get_bind()).get_columns("schedules")
    if not any(column["name"] == "timing_edited_at" for column in columns):
        op.add_column("schedules", sa.Column("timing_edited_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    connection = op.get_bind()
    # SQLite can autocommit the first DDL in a downgrade. Keep the edit
    # timestamps if a later revision refuses the downgrade and rolls back.
    if connection.dialect.name == "sqlite" and not connection.connection.driver_connection.in_transaction:
        connection.exec_driver_sql("BEGIN")
    op.drop_column("schedules", "timing_edited_at")
