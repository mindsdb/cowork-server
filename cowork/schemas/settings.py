from pydantic import BaseModel


class SettingUpsertRequest(BaseModel):
    value: str


class SettingsBulkUpsertRequest(BaseModel):
    values: dict[str, str]


class SettingResponse(BaseModel):
    key: str
    label: str
    description: str
    is_sensitive: bool
    is_set: bool
    value: str | None
    options: list[str] | None = None
