from sqlmodel import Field

from cowork.models.base import BaseSQLModel


class Setting(BaseSQLModel, table=True):
    __tablename__ = "settings"

    key: str = Field(max_length=128, unique=True, index=True)
    value: str
