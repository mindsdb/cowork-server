from pydantic import BaseModel


class PinRequest(BaseModel):
    item_type: str
    item_id: str
    title: str | None = None
