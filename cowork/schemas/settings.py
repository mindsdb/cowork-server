from pydantic import BaseModel


class SettingUpsertRequest(BaseModel):
    value: str


class SettingsBulkUpsertRequest(BaseModel):
    # `None` joins "***" as a skip sentinel — the client's write-diff sends it
    # for untouched fields, and the service already skips both.
    values: dict[str, str | None]


class SettingResponse(BaseModel):
    key: str
    label: str
    description: str
    is_sensitive: bool
    is_set: bool
    value: str | None
    options: list[str] | None = None
