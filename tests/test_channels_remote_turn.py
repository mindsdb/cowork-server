"""_turn_stream: the one place a channel turn picks in-process vs remote
execution. Real Session, not a fake: org_mode ScopedSession needs session.info and a listener only a real Session gives."""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

import cowork.channels.runtime as runtime_mod
import cowork.handlers.responses as responses_mod
from cowork.channels.runtime import AntonChannelRuntime
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_open_session
from _fakes import inline_run_db


class _FakeHarnessLocal:
    def stream_response(self, **kwargs):
        async def gen():
            yield "local-sentinel"
        return gen()


def test_turn_stream_uses_the_harness_when_backend_is_not_remote(monkeypatch):
    monkeypatch.delenv("COWORK_TURN_BACKEND", raising=False)
    runtime = AntonChannelRuntime(runtime_mod.LiveAdapterRegistry())
    harness = _FakeHarnessLocal()

    called_remote = []
    monkeypatch.setattr(
        runtime_mod, "remote_turn_events",
        lambda **kwargs: called_remote.append(kwargs) or _empty_async_gen(),
    )

    async def scenario():
        session = get_open_session()
        try:
            scoped = ScopedSession(session, TenantScope(org_mode=False))
            conversation = type("C", (), {"id": uuid4()})()
            stream = await runtime._turn_stream(
                harness, "anton", scoped, conversation, [{"type": "text", "text": "hi"}],
                "hi", None, [],
            )
            chunk = await stream.__anext__()
            assert chunk == "local-sentinel"
        finally:
            session.close()

    asyncio.run(scenario())
    assert called_remote == []


async def _empty_async_gen():
    return
    yield  # pragma: no cover — unreachable, makes this an async generator


def test_turn_stream_uses_the_remote_worker_when_backend_is_remote(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    runtime = AntonChannelRuntime(runtime_mod.LiveAdapterRegistry())
    harness = _FakeHarnessLocal()

    # _turn_stream stages through stage_remote_workspace, which calls
    # ResponsesHandler._stage_remote_workspace_files: stub the class
    # attribute, not a name in runtime_mod.
    staged = []
    from cowork.handlers.responses import ResponsesHandler
    monkeypatch.setattr(
        ResponsesHandler, "_stage_remote_workspace_files",
        staticmethod(lambda session, conv_id: staged.append(conv_id)),
    )

    captured = {}

    async def fake_remote_turn_events(**kwargs):
        captured.update(kwargs)
        return
        yield  # pragma: no cover

    monkeypatch.setattr(runtime_mod, "remote_turn_events", fake_remote_turn_events)

    # Declared outside scenario() so the post-await assertions below can see
    # them too — a nested function's locals don't leak to its enclosing scope.
    conv_id = uuid4()
    turn_rows = []

    # Staging is a unit, run by cowork.handlers.responses; it runs inline
    # here, on a session nothing reads.
    monkeypatch.setattr(responses_mod, "run_db", inline_run_db(None))
    scope = TenantScope(org_mode=True, org_id="org-a")

    async def scenario():
        conversation = type("C", (), {"id": conv_id})()
        stream = await runtime._turn_stream(
            harness, "anton", scope, conversation, [{"type": "text", "text": "hi"}],
            "hi", None, turn_rows,
        )
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()

    asyncio.run(scenario())

    assert staged == [conv_id]
    assert captured["scope"] == scope
    assert captured["conv_id"] == conv_id
    assert captured["org_id"] == "org-a"
    assert captured["input_text"] == "hi"
    assert captured["turn_rows"] is turn_rows


def _remote_turn_whose_staging_unit(monkeypatch, *, unit_session) -> list:
    """A remote channel turn whose staging unit opens its session through
    ``unit_session``, with the staging itself a no-op. Returns the ids of the
    conversations whose remote turn started."""
    from cowork.db import units
    from cowork.handlers.responses import ResponsesHandler

    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    monkeypatch.setattr(
        ResponsesHandler, "_stage_remote_workspace_files", staticmethod(lambda session, conv_id: None),
    )
    monkeypatch.setattr(units, "unit_session", unit_session)
    started: list = []

    async def fake_remote_turn_events(**kwargs):
        started.append(kwargs["conv_id"])
        return
        yield  # pragma: no cover

    monkeypatch.setattr(runtime_mod, "remote_turn_events", fake_remote_turn_events)
    return started


async def _stream_one_turn(conv_id) -> None:
    runtime = AntonChannelRuntime(runtime_mod.LiveAdapterRegistry())
    stream = await runtime._turn_stream(
        _FakeHarnessLocal(), "anton", TenantScope(org_mode=True, org_id="org-a"),
        type("C", (), {"id": conv_id})(), [{"type": "text", "text": "hi"}], "hi", None, [],
    )
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


def test_a_channel_turn_whose_staging_unit_fails_to_commit_still_runs(monkeypatch, caplog):
    """Staging degrades on its own (_stage_remote_workspace_files), so a
    staging unit that fails to commit runs the turn without the staged
    files, as a browser turn does, and logs the failure."""
    import contextlib
    import logging
    from types import SimpleNamespace

    @contextlib.contextmanager
    def unit_whose_commit_fails(*, scope):
        yield SimpleNamespace(scope=scope)
        raise RuntimeError("the unit's commit failed")

    started = _remote_turn_whose_staging_unit(monkeypatch, unit_session=unit_whose_commit_fails)
    conv_id = uuid4()

    with caplog.at_level(logging.ERROR, logger=responses_mod.logger.name):
        asyncio.run(_stream_one_turn(conv_id))

    assert started == [conv_id]
    [record] = [r for r in caplog.records if r.name == responses_mod.logger.name]
    assert (record.levelname, record.getMessage()) == ("ERROR", "[responses] failed to stage workspace files")
    assert record.conversation_id == str(conv_id)


def test_a_channel_turn_whose_staging_unit_finds_no_connection_is_refused(monkeypatch):
    """A full pool is not a missing file: the turn is refused rather than
    run without what the user sent."""
    from cowork.db.units import DatabaseBusy

    def refuse(*, scope):
        raise DatabaseBusy("no database connection freed within POOL_TIMEOUT")

    started = _remote_turn_whose_staging_unit(monkeypatch, unit_session=refuse)

    with pytest.raises(DatabaseBusy):
        asyncio.run(_stream_one_turn(uuid4()))

    assert started == []
