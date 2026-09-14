"""Add tenant-namespaced Code control-plane records.

Revision ID: e2c4a6f8b1d3
Revises: a4c8e1f6b3d9
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e2c4a6f8b1d3"
down_revision: Union[str, Sequence[str], None] = "a4c8e1f6b3d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "code_control_records" not in inspector.get_table_names():
        op.create_table(
            "code_control_records",
            sa.Column("namespace_id", sa.String(length=128), nullable=False),
            sa.Column("collection", sa.String(length=32), nullable=False),
            sa.Column("document_id", sa.String(length=160), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("assigned_computer_id", sa.String(length=128), nullable=True),
            sa.Column("lifecycle_status", sa.String(length=32), nullable=True),
            sa.Column("parent_id", sa.String(length=160), nullable=True),
            sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.PrimaryKeyConstraint("namespace_id", "collection", "document_id"),
        )
        inspector = sa.inspect(bind)
    existing_columns = {item["name"] for item in inspector.get_columns("code_control_records")}
    if "parent_id" not in existing_columns:
        op.add_column(
            "code_control_records",
            sa.Column("parent_id", sa.String(length=160), nullable=True),
        )
        inspector = sa.inspect(bind)
    # Prototype databases may already contain durable records from the first
    # version of this migration. Backfill the new ownership projection before
    # creating its index; otherwise those existing runs remain invisible to
    # scoped polling and cascade deletion until individually rewritten.
    records = sa.table(
        "code_control_records",
        sa.column("namespace_id", sa.String()),
        sa.column("collection", sa.String()),
        sa.column("document_id", sa.String()),
        sa.column("payload", sa.JSON()),
        sa.column("parent_id", sa.String()),
    )
    owned_collections = {"runs", "workspaces", "grants", "commands", "run_credentials", "audit"}
    for namespace_id, collection, document_id, payload in bind.execute(
        sa.select(
            records.c.namespace_id,
            records.c.collection,
            records.c.document_id,
            records.c.payload,
        ).where(records.c.parent_id.is_(None))
    ):
        if collection not in owned_collections or not isinstance(payload, dict):
            continue
        parent_key = "task_id" if collection == "runs" else "run_id"
        parent_id = payload.get(parent_key)
        if parent_id:
            bind.execute(
                records.update()
                .where(records.c.namespace_id == namespace_id)
                .where(records.c.collection == collection)
                .where(records.c.document_id == document_id)
                .values(parent_id=str(parent_id))
            )
    existing_indexes = {item["name"] for item in inspector.get_indexes("code_control_records")}
    if "ix_code_control_records_namespace_collection" not in existing_indexes:
        op.create_index(
            "ix_code_control_records_namespace_collection",
            "code_control_records",
            ["namespace_id", "collection", "updated_at"],
        )
    if "ix_code_control_records_run_queue" not in existing_indexes:
        op.create_index(
            "ix_code_control_records_run_queue",
            "code_control_records",
            ["namespace_id", "collection", "assigned_computer_id", "lifecycle_status", "created_at"],
        )
    if "ix_code_control_records_parent" not in existing_indexes:
        op.create_index(
            "ix_code_control_records_parent",
            "code_control_records",
            ["namespace_id", "collection", "parent_id", "updated_at"],
        )


def downgrade() -> None:
    if "code_control_records" in sa.inspect(op.get_bind()).get_table_names():
        op.drop_table("code_control_records")
