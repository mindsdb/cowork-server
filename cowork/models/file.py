from sqlalchemy import Text
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class File(BaseSQLModel, table=True):
    __tablename__ = "files"

    filename: str = Field(max_length=255)
    content_type: str = Field(max_length=127)
    size: int
    # Attachment purpose tags are "attachment:{session_id}" — keyed by the
    # stable conversation id only (ENG-338: embedding the mutable project
    # name stranded attachments on rename; ENG-333: it also let long names
    # overflow this column when it had a fixed width). Kept as unbounded
    # TEXT: other purposes remain free-form, and width must never be the
    # thing that 500s an upload again.
    purpose: str = Field(sa_type=Text)
    path: str = Field(max_length=1024)
    org_id: str | None = Field(default=None, index=True, max_length=36, description="Owning organization; NULL on local/desktop rows")
    created_by: str | None = Field(default=None, max_length=36, description="User who created the row; NULL on local/desktop rows")
