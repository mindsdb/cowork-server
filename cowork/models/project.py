from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class Project(BaseSQLModel, table=True):
    __tablename__ = "projects"

    name: str = Field(description="Name of the project", max_length=255)
    display_name: str | None = Field(
        default=None,
        max_length=255,
        description=(
            "What the user typed, preserved verbatim. NULL means the row predates "
            "this column; readers resolve it as `display_name or name` via "
            "`projects.display_label`. Never an identifier - `name` remains the "
            "directory, the URL segment and the lookup key (ENG-1676)."
        ),
    )
    path: str = Field(
        description="Path to the project directory on the server",
        max_length=1024,
    )
    is_active: bool = Field(default=True, description="Whether the project is active")
    org_id: str | None = Field(default=None, index=True, max_length=36, description="Owning organization; NULL on local/desktop rows")
    created_by: str | None = Field(default=None, max_length=36, description="User who created the row; NULL on local/desktop rows")

