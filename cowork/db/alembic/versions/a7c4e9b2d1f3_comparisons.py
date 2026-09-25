"""Model comparisons: two sides of one task, and the user's verdicts.

Revision ID: a7c4e9b2d1f3
Revises: 3e4b5f7586d3
"""

from alembic import op
import sqlalchemy as sa

revision = "a7c4e9b2d1f3"
down_revision = "3e4b5f7586d3"
branch_labels = None
depends_on = None


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("modified_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
    ]


def _ownership() -> list[sa.Column]:
    return [
        sa.Column("org_id", sa.String(36), nullable=True),
        sa.Column("created_by", sa.String(36), nullable=True),
    ]


def upgrade() -> None:
    # Bootstrap can create SQLModel metadata before Alembic adopts the schema,
    # so each table is created only when absent.
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    if "comparisons" not in tables:
        op.create_table(
            "comparisons",
            sa.Column("id", sa.Uuid(), nullable=False),
            *_timestamps(),
            sa.Column("title", sa.String(255), nullable=False),
            sa.Column("source_project_id", sa.Uuid(), nullable=True),
            sa.Column("source_project_label", sa.String(255), nullable=True),
            *_ownership(),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_comparisons_org_id", "comparisons", ["org_id"])
    if "comparison_sides" not in tables:
        op.create_table(
            "comparison_sides",
            sa.Column("id", sa.Uuid(), nullable=False),
            *_timestamps(),
            sa.Column("comparison_id", sa.Uuid(), nullable=False),
            sa.Column("label", sa.String(1), nullable=False),
            sa.Column("model", sa.String(255), nullable=False),
            sa.Column("reasoning_effort", sa.String(32), nullable=True),
            sa.Column("project_id", sa.Uuid(), nullable=False),
            sa.Column("conversation_id", sa.Uuid(), nullable=False),
            sa.Column("continued_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("continued_turn_count", sa.Integer(), nullable=True),
            *_ownership(),
            sa.ForeignKeyConstraint(["comparison_id"], ["comparisons.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("comparison_id", "label", name="uq_comparison_sides_label"),
        )
        op.create_index("ix_comparison_sides_conversation_id", "comparison_sides", ["conversation_id"])
    if "comparison_verdicts" not in tables:
        op.create_table(
            "comparison_verdicts",
            sa.Column("id", sa.Uuid(), nullable=False),
            *_timestamps(),
            sa.Column("comparison_id", sa.Uuid(), nullable=False),
            sa.Column("turn_index", sa.Integer(), nullable=False),
            sa.Column("winner", sa.String(16), nullable=False),
            *_ownership(),
            sa.ForeignKeyConstraint(["comparison_id"], ["comparisons.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("comparison_id", "turn_index", name="uq_comparison_verdicts_turn"),
        )
        op.create_index("ix_comparison_verdicts_comparison_id", "comparison_verdicts", ["comparison_id"])


def downgrade() -> None:
    tables = set(sa.inspect(op.get_bind()).get_table_names())
    for name in ("comparison_verdicts", "comparison_sides", "comparisons"):
        if name in tables:
            op.drop_table(name)
