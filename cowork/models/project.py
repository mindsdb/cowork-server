from datetime import datetime

import sqlalchemy as sa
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class Project(BaseSQLModel, table=True):
    __tablename__ = "projects"

    name: str = Field(description="Name of the project", max_length=255)
    path: str = Field(
        description="Path to the project directory on the server",
        max_length=1024,
    )

    # ── Organization metadata (server-side, follows the user across devices) ──
    pinned: bool = Field(
        default=False,
        description="Whether the project is pinned/favorited in the list.",
    )
    sort_order: int = Field(
        default=0,
        description="Manual ordering position within the list (ascending).",
    )
    archived: bool = Field(
        default=False,
        description="Whether the project is archived (hidden from the active list).",
    )
    last_selected_at: datetime | None = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),  # type: ignore
        description=(
            "The last time this project was selected/opened by the user. This is "
            "the single server-side notion of the 'active' project: interactive "
            "requests carry an explicit project, so the server only consults this "
            "as the fallback for headless/scheduled runs that omit one."
        ),
    )

