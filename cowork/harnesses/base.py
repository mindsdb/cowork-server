from typing import AsyncIterator, Protocol

from cowork.schemas.responses import Message


class HarnessProvider(Protocol):
    id: str
    label: str
    
    async def stream_response(
        self,
        *,
        messages: list[Message],
        model: str,
    ) -> AsyncIterator[str]:
        ...