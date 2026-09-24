from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from cowork.schemas.base import CamelRequest, CamelResponse


class ComparisonSideRequest(CamelRequest):
    model: str = Field(min_length=1, max_length=255)
    reasoning_effort: str | None = Field(default=None, max_length=32)


class ComparisonCreateRequest(CamelRequest):
    title: str = Field(default="", max_length=2000, description="The first prompt; shortened for the history list")
    sides: list[ComparisonSideRequest] = Field(min_length=2, max_length=2)
    source_project_id: UUID | None = Field(
        default=None, description="Copy this project's files into both sides; omit for an empty start"
    )


class ComparisonVerdictRequest(CamelRequest):
    winner: Literal["a", "b", "tie", "neither"]


class ComparisonContinueRequest(CamelRequest):
    project_id: UUID


class ComparisonSideResponse(CamelResponse):
    label: str
    model: str
    reasoning_effort: str | None = None
    project_id: UUID
    conversation_id: UUID
    message_count: int
    continued_at: datetime | None = None
    continued_turn_count: int | None = None


class ComparisonVerdictResponse(CamelResponse):
    turn_index: int
    winner: str
    modified_at: datetime | None = None


class ComparisonResponse(CamelResponse):
    id: UUID
    title: str
    created_at: datetime | None
    source_project_id: UUID | None = None
    source_project_label: str | None = None
    sides: list[ComparisonSideResponse]
    verdicts: list[ComparisonVerdictResponse]
    #: The latest judged turn's winner, or None before the first verdict.
    verdict: str | None = None
