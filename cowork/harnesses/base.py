from typing import AsyncIterator, Protocol

from cowork.runtime.schemas import (
    CoworkEvent,
    HarnessCapabilities,
    HarnessReadiness,
    HarnessTurnRequest,
)


class HarnessProvider(Protocol):
    id: str
    label: str

    def capabilities(self) -> HarnessCapabilities:
        ...

    async def health(self) -> dict:
        ...

    def validate_request(self, request: HarnessTurnRequest) -> HarnessReadiness:
        ...

    async def start_turn(self, request: HarnessTurnRequest) -> AsyncIterator[CoworkEvent]:
        ...

    async def cancel_turn(self, turn_id: str) -> None:
        ...
