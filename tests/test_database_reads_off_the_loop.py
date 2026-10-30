"""Search, settings, the MindsHub workspaces and the channel ingress reconcile
read the database in worker threads.

With the pool's only connection held elsewhere, each one waits for it without
stopping the event loop: a heartbeat on the test's loop keeps ticking and
/api/v1/health/live keeps answering within 0.5 s. Each one is still refused
once POOL_TIMEOUT has passed, a request with the 503 the app answers a full
pool with. The heartbeat runs through httpx.ASGITransport, so a query that ran
on the loop shows up as missing ticks rather than as a slow request.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx
import pytest

import cowork.api.v1.endpoints.hub_workspaces as hub_workspaces_ep
from cowork.channels import ingress
from cowork.db.units import DatabaseBusy
from cowork.server import create_app
from cowork.services.hub_workspaces import HubWorkspaceListing

# POOL_TIMEOUT in these tests, in seconds; one_connection_pool sets it.
WAIT = 2
LIVE_BUDGET = 0.5


@dataclass
class _Measured:
    outcome: object
    elapsed: float
    ticks: int
    live_answers: int
    slowest_live: float


async def _while_others_are_served(
    client: httpx.AsyncClient, work: Callable[[], Awaitable[object]]
) -> _Measured:
    """Run ``work`` while a heartbeat ticks every 10 ms and
    /api/v1/health/live is asked every 50 ms."""
    ticks = 0
    live_answers = 0
    slowest_live = 0.0
    loop = asyncio.get_running_loop()

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    async def probe_liveness():
        nonlocal live_answers, slowest_live
        while True:
            asked = loop.time()
            if (await client.get("/api/v1/health/live")).status_code == 200:
                live_answers += 1
            slowest_live = max(slowest_live, loop.time() - asked)
            await asyncio.sleep(0.05)

    beat = asyncio.create_task(heartbeat())
    prober = asyncio.create_task(probe_liveness())
    started = loop.time()
    try:
        try:
            outcome = await asyncio.wait_for(work(), timeout=WAIT * 5)
        except DatabaseBusy as refused:
            outcome = refused
        elapsed = loop.time() - started
    finally:
        for task in (beat, prober):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    return _Measured(outcome, elapsed, ticks, live_answers, slowest_live)


def _assert_the_loop_kept_serving(measured: _Measured) -> None:
    # Refused once POOL_TIMEOUT has passed, and not long after.
    assert 0.8 * WAIT <= measured.elapsed <= 1.5 * WAIT, f"answered after {measured.elapsed:.2f}s"
    # A free loop ticks about 100 times a second here; a blocked one stalls.
    assert measured.ticks >= 20, f"event loop stalled: {measured.ticks} ticks in {measured.elapsed:.2f}s"
    assert measured.live_answers >= 5, f"/api/v1/health/live answered {measured.live_answers} times"
    assert measured.slowest_live < LIVE_BUDGET, f"/api/v1/health/live took {measured.slowest_live:.2f}s"


@pytest.fixture
def held_pool(one_connection_pool):
    """The pool's only connection, out with another caller for the test."""
    held = one_connection_pool.connect()
    try:
        yield one_connection_pool
    finally:
        held.close()


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=create_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def one_workspace(monkeypatch):
    """MindsHub's listing, answered without a network call."""

    async def listing(*, bearer_token, org_id, user_id):
        return HubWorkspaceListing(workspaces=[], reachable=True)

    monkeypatch.setattr(hub_workspaces_ep, "fetch_hub_workspaces", listing)


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/search?q=quarterly",
        "/api/v1/settings/",
        "/api/v1/settings/recommended-models",
        "/api/v1/hub/workspaces/",
    ],
)
async def test_a_read_that_finds_the_pool_held_waits_off_the_loop(held_pool, client, one_workspace, path):
    measured = await _while_others_are_served(client, lambda: client.get(path))

    response = measured.outcome
    assert response.status_code == 503, response.text[:300]
    assert response.json()["code"] == "server_busy"
    assert response.headers["retry-after"] == str(WAIT)
    _assert_the_loop_kept_serving(measured)


async def test_the_ingress_reconcile_waits_off_the_loop(held_pool, client):
    """The reconcile runs every 30 s on every replica. Its read of every org's
    installations waits for the pool in a worker thread, and the pass is
    refused at POOL_TIMEOUT; the loop's caller logs it and tries again."""

    class _Manager:
        def running_keys(self):
            return []

    measured = await _while_others_are_served(
        client, lambda: ingress.reconcile_once(_Manager(), adapters=None)
    )

    assert isinstance(measured.outcome, DatabaseBusy)
    _assert_the_loop_kept_serving(measured)
