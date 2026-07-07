
from cowork.schemas.base import CamelRequest


class PinRequest(CamelRequest):
    item_type: str
    item_id: str
    title: str | None = None
