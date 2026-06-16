from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class File(BaseSQLModel, table=True):
    __tablename__ = "files"

    filename: str = Field(max_length=255)
    content_type: str = Field(max_length=127)
    size: int
    # Attachment purpose tags embed the project name + conversation id
    # ("attachment:{project}:{session}"). The session UUID alone is 36 chars,
    # so 64 only left ~16 for the project name — a longer project name (e.g.
    # "Catana-Outbound-email") overflowed, failed model validation, and
    # crashed the upload with a 500. Widened to 255 to fit any project name.
    purpose: str = Field(max_length=255)
    path: str = Field(max_length=1024)
