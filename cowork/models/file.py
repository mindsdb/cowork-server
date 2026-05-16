from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class File(BaseSQLModel, table=True):
    __tablename__ = "files"

    filename: str = Field(max_length=255)
    content_type: str = Field(max_length=127)
    size: int
    purpose: str = Field(max_length=64)
    path: str = Field(max_length=1024)
