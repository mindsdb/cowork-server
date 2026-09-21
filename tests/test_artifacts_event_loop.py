"""ENG-2436: `GET /api/v1/artifacts/` must not block the event loop.

On 2026-09-09 the whole-EFS scan ran synchronously on the request handler,
freezing both prod replicas until the health probe timed out and Kubernetes
killed the pods. A slow filesystem scan must degrade the one request, not the
process: everything else on the loop (including the liveness probe) has to
keep making progress while it runs.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from unittest.mock import patch

import httpx

from cowork.api.v1.endpoints import artifacts as artifacts_ep
from cowork.server import create_app


def test_list_artifacts_does_not_block_the_event_loop():
    """Discriminates on tick count, not wall-clock gaps: a GC pause or a loaded
    CI runner can widen a single gap past any fixed millisecond threshold with
    entirely correct code. A blocked loop gives 0-1 ticks across the 0.5s
    stub; a healthy one gives roughly 50 — the same discrimination without a
    timing threshold.
    """
    def slow_all_artifact_cards(session):
        time.sleep(0.5)
        return []

    async def flow():
        app = create_app()
        transport = httpx.ASGITransport(app=app)
        ticks_during_request = 0

        async def heartbeat():
            nonlocal ticks_during_request
            while True:
                await asyncio.sleep(0.01)
                ticks_during_request += 1

        hb_task = asyncio.create_task(heartbeat())
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                with patch.object(artifacts_ep, "_all_artifact_cards", slow_all_artifact_cards):
                    res = await client.get("/api/v1/artifacts/")
            assert res.status_code == 200, res.text
        finally:
            hb_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb_task

        assert ticks_during_request >= 5, (
            f"heartbeat ticked only {ticks_during_request} times during the "
            "request — the loop stalled"
        )

    asyncio.run(flow())
