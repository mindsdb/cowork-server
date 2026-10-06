"""A question that finds no free database connection is refused within
POOL_TIMEOUT, with a 503 the web UI can show, while the event loop keeps
serving everyone else.

Waits are measured with a heartbeat on the test's own loop, through
httpx.ASGITransport, so a checkout that blocked the loop shows up as missing
ticks rather than as a slow request.
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import httpx
import pytest

import cowork.db.session as db_session
import cowork.handlers.responses as responses_mod
from cowork.common.settings.app_settings import DatabaseSettings
from cowork.handlers.response_routing import DELEGATED_AGENTIC, RouteDecision
from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.units import run_db
from cowork.server import create_app
from cowork.streaming import registry

from _fakes import PausedHarness

# POOL_TIMEOUT in these tests, in seconds; one_connection_pool uses the same.
WAIT = 2

BUSY = {"detail": f"Cowork is busy. Try again in about {WAIT} seconds.", "code": "server_busy"}


@pytest.fixture(autouse=True)
def _forget_turns():
    yield
    registry.reset()


@dataclass
class _Measured:
    response: httpx.Response
    elapsed: float
    ticks: int
    live_answers: int


async def _ask_while_others_are_served() -> _Measured:
    """POST a question while a heartbeat ticks every 10 ms and
    /api/v1/health/live is asked every 50 ms."""
    ticks = 0
    live_answers = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    transport = httpx.ASGITransport(app=create_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def probe_liveness():
            nonlocal live_answers
            while True:
                if (await client.get("/api/v1/health/live")).status_code == 200:
                    live_answers += 1
                await asyncio.sleep(0.05)

        beat = asyncio.create_task(heartbeat())
        prober = asyncio.create_task(probe_liveness())
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            response = await asyncio.wait_for(
                client.post("/api/v1/responses/", json={"input": "hi", "stream": True}),
                timeout=WAIT * 5,
            )
            elapsed = loop.time() - started
        finally:
            for task in (beat, prober):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
    return _Measured(response=response, elapsed=elapsed, ticks=ticks, live_answers=live_answers)


def _assert_refused_as_busy(measured: _Measured) -> None:
    assert measured.response.status_code == 503, measured.response.text[:300]
    assert measured.response.json() == BUSY
    assert measured.response.headers["retry-after"] == str(WAIT)
    # Refused once POOL_TIMEOUT has passed, and not long after.
    assert 0.8 * WAIT <= measured.elapsed <= 1.2 * WAIT, f"refused after {measured.elapsed:.2f}s"
    # A free loop ticks about 100 times a second here; a blocked one once.
    assert measured.ticks >= 20, f"event loop stalled: {measured.ticks} ticks in {measured.elapsed:.2f}s"
    assert measured.live_answers >= 5, f"/api/v1/health/live answered {measured.live_answers} times"


async def test_a_question_that_finds_the_pool_held_is_refused_while_the_loop_keeps_serving(
    one_connection_pool,
):
    """The pool's only connection is out with another caller, so the request's
    unit waits for it in a worker thread and gives up at POOL_TIMEOUT."""
    held_elsewhere = one_connection_pool.connect()
    try:
        measured = await _ask_while_others_are_served()
    finally:
        held_elsewhere.close()

    _assert_refused_as_busy(measured)


async def test_a_connection_that_frees_within_pool_timeout_is_used(one_connection_pool, monkeypatch):
    """The pool's only connection comes back half way through POOL_TIMEOUT,
    so the question waiting for it is answered, not refused."""
    gate = PausedHarness()
    gate.release.set()
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: gate)

    async def decide(**_kwargs):
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses_mod, "decide_route", decide)
    held_elsewhere = one_connection_pool.connect()
    asyncio.get_running_loop().call_later(WAIT / 2, held_elsewhere.close)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app()), base_url="http://test",
        ) as client:
            response = await asyncio.wait_for(
                client.post("/api/v1/responses/", json={"input": "hi", "stream": True}),
                timeout=WAIT * 5,
            )
    finally:
        held_elsewhere.close()

    assert response.status_code == 200, response.text[:300]
    assert "event: response.completed" in response.text


async def test_a_question_that_finds_every_unit_slot_taken_is_refused_while_the_loop_keeps_serving(
    monkeypatch,
):
    """Units queue for a slot, one per connection the pool can lend (one on
    SQLite), before they ask the pool. That wait is bounded by POOL_TIMEOUT
    too, and ends in the same refusal."""
    monkeypatch.setattr(db_session.settings.database, "pool_timeout", WAIT)
    entered = threading.Event()
    release = threading.Event()

    def hold_the_slot(session):
        entered.set()
        release.wait(timeout=30)

    holder = asyncio.create_task(run_db(hold_the_slot, scope=LOCAL_SCOPE))
    try:
        while not entered.is_set():
            await asyncio.sleep(0.01)
        measured = await _ask_while_others_are_served()
    finally:
        release.set()
        await holder

    _assert_refused_as_busy(measured)


def test_a_postgres_pool_waits_five_seconds_unless_pool_timeout_says_otherwise(monkeypatch):
    """The 5 s default is what keeps a refusal inside 10 s on a deployment that
    sets no POOL_TIMEOUT. create_engine connects lazily: no database needed."""
    uri = "postgresql+psycopg://cowork:secret@db.invalid:5432/cowork"

    def pool_wait() -> float:
        monkeypatch.setattr(
            db_session, "settings", SimpleNamespace(database=DatabaseSettings(_env_file=None)),
        )
        engine = db_session._create_engine(uri)
        try:
            return engine.pool.timeout()
        finally:
            engine.dispose()

    monkeypatch.delenv("POOL_TIMEOUT", raising=False)
    assert pool_wait() == 5

    monkeypatch.setenv("POOL_TIMEOUT", "7")
    assert pool_wait() == 7
