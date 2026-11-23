"""Drive a response the way Starlette does, so it releases what it holds."""
from __future__ import annotations

import anyio
from starlette.responses import Response


async def served_body(response: Response) -> bytes:
    """The body ``response`` sends through its ASGI call.

    A PinnedStreamingResponse releases its descriptors only in that call, so
    a test that reads ``body_iterator`` instead leaves them to the garbage
    collector (the ``pinned_responses_released`` fixture in tests/conftest.py).
    """
    body: list[bytes] = []

    async def receive():
        # No disconnect: the client stays until the body is sent.
        await anyio.sleep_forever()

    async def send(message):
        if message["type"] == "http.response.body":
            body.append(message.get("body", b""))

    await response({"type": "http", "method": "GET", "path": "/", "headers": []}, receive, send)
    return b"".join(body)
