"""Private moderation queue. No client endpoint lists reports or their evidence."""
from uuid import UUID
from sqlalchemy import UniqueConstraint, Index
from sqlmodel import Field
from cowork.models.base import BaseSQLModel

class ContentReport(BaseSQLModel, table=True):
    __tablename__ = "content_reports"
    __table_args__ = (
        UniqueConstraint("org_id", "user_id", "target_key", name="uq_content_report_target"),
        Index("ix_content_reports_queue", "status", "created_at"),
        Index("ix_content_reports_reporter", "org_id", "user_id", "created_at"),
    )
    org_id: str = Field(max_length=36)
    user_id: str = Field(max_length=36)
    target_key: str = Field(max_length=128)
    conversation_id: UUID | None = None
    message_id: UUID | None = None
    project_id: UUID | None = None
    artifact_id: UUID | None = None
    # Copy from the authorized database row, never client-supplied evidence.
    evidence: str = Field(max_length=65536)
    status: str = Field(default="pending", max_length=24)
    reviewed_by: str | None = Field(default=None, max_length=128)
    resolution: str | None = Field(default=None, max_length=2000)
