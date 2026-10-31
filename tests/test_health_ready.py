"""Readiness answers from the event loop alone.

The kubelet pulls a pod that fails readiness out of the service. Readiness
used to build settings and open a database session, so a full pool or a slow
file store failed it on every replica at once and nginx answered 503 for the
whole service. /api/v1/health/ready reads neither.
"""
from __future__ import annotations

import asyncio

import httpx

from cowork.api.v1.endpoints import health
from cowork.server import create_app

BUDGET_SECONDS = 0.5


async def test_ready_answers_with_the_pool_held_full_and_settings_unreachable(
    one_connection_pool, monkeypatch,
):
    def unreachable(*_args, **_kwargs):
        raise AssertionError("readiness read settings")

    monkeypatch.setattr(health, "get_user_settings", unreachable)
    monkeypatch.setattr(health, "get_app_settings", unreachable)
    held = one_connection_pool.connect()
    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            loop = asyncio.get_running_loop()
            started = loop.time()
            response = await client.get("/api/v1/health/ready")
            elapsed = loop.time() - started
    finally:
        held.close()

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert elapsed < BUDGET_SECONDS
