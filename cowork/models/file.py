from sqlalchemy import Text
from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class File(BaseSQLModel, table=True):
    __tablename__ = "files"

    filename: str = Field(max_length=255)
    content_type: str = Field(max_length=127)
    size: int
    # Attachment purpose tags embed the project name + conversation id
    # ("attachment:{project}:{session}"). The session UUID alone is 36 chars,
    # and a project name can be up to 255 (Project.name), so this string can
    # reach ~303 chars — any fixed width couples the column to Project.name's
    # cap and can silently re-overflow (the original bug was a 64-char cap
    # → 500). Use unbounded TEXT so a long project name can never crash the
    # upload, regardless of how Project.name's limit changes later.
    purpose: str = Field(sa_type=Text)
    path: str = Field(max_length=1024)
