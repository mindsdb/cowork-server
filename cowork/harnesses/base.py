from typing import AsyncIterator, Protocol


class HarnessProvider(Protocol):
    id: str
    label: str
    
    async def stream_response(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
    ) -> AsyncIterator[str]:
        ...