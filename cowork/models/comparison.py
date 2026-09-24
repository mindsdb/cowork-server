from datetime import datetime
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import Index, UniqueConstraint
from sqlmodel import Field, Relationship

from cowork.models.base import BaseSQLModel


class Comparison(BaseSQLModel, table=True):
    """One task given to two models side by side.

    Each side is an ordinary conversation in its own hidden project (see
    `ProjectService.create_comparison_sandbox`), so everything a task already
    does -- streaming, history, artifacts, cancel -- works per side unchanged.
    This row only groups the two and remembers what the user set up.
    """

    __tablename__ = "comparisons"

    title: str = Field(max_length=255, description="The first prompt, shortened, for the history list")
    # No FK: the source project may be deleted later, and the comparison is a
    # record of what was run, which must outlive it.
    source_project_id: UUID | None = Field(
        default=None, description="Project whose files both sides started from; NULL for an empty start"
    )
    source_project_label: str | None = Field(
        default=None, max_length=255, description="That project's name when the comparison started"
    )
    org_id: str | None = Field(default=None, index=True, max_length=36, description="Owning organization; NULL on local/desktop rows")
    created_by: str | None = Field(default=None, max_length=36, description="User who created the row; NULL on local/desktop rows")

    sides: list["ComparisonSide"] = Relationship(
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "order_by": "ComparisonSide.label"},
    )
    verdicts: list["ComparisonVerdict"] = Relationship(
        sa_relationship_kwargs={"cascade": "all, delete-orphan", "order_by": "ComparisonVerdict.turn_index"},
    )


class ComparisonSide(BaseSQLModel, table=True):
    __tablename__ = "comparison_sides"

    comparison_id: UUID = Field(foreign_key="comparisons.id", description="Comparison this side belongs to")
    label: str = Field(max_length=1, description="'a' or 'b'")
    model: str = Field(max_length=255, description="Model alias every turn of this side runs on")
    reasoning_effort: str | None = Field(
        default=None, max_length=32, description="Effort every turn of this side runs at; NULL for the model's default"
    )
    # Plain UUIDs, no FK: once a side is continued its conversation belongs to
    # a real project and the user can delete it like any task. An FK here
    # would make that delete fail on Postgres (SQLite never enforces it).
    project_id: UUID = Field(description="The side's sandbox project")
    conversation_id: UUID = Field(index=True, description="The side's conversation")
    continued_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description="When the user continued with this side; its conversation left the sandbox then",
    )
    continued_turn_count: int | None = Field(
        default=None, description="Turns the conversation had when it was continued; the comparison shows those"
    )
    org_id: str | None = Field(default=None, max_length=36, description="Owning organization; NULL on local/desktop rows")
    created_by: str | None = Field(default=None, max_length=36, description="User who created the row; NULL on local/desktop rows")

    __table_args__ = (UniqueConstraint("comparison_id", "label", name="uq_comparison_sides_label"),)


class ComparisonVerdict(BaseSQLModel, table=True):
    """Which side the user judged better, recorded against a turn.

    One row per judged turn, so a changing opinion across a long comparison is
    kept rather than overwritten; the latest row is the comparison's verdict.
    """

    __tablename__ = "comparison_verdicts"

    comparison_id: UUID = Field(foreign_key="comparisons.id", description="Comparison judged")
    turn_index: int = Field(description="Zero-based index of the shared turn the verdict was given on")
    winner: str = Field(max_length=16, description="'a', 'b', 'tie' or 'neither'")
    org_id: str | None = Field(default=None, max_length=36, description="Owning organization; NULL on local/desktop rows")
    created_by: str | None = Field(default=None, max_length=36, description="User who created the row; NULL on local/desktop rows")

    __table_args__ = (
        UniqueConstraint("comparison_id", "turn_index", name="uq_comparison_verdicts_turn"),
        Index("ix_comparison_verdicts_comparison_id", "comparison_id"),
    )
