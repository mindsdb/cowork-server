"""_turn_stream: the one place a channel turn picks in-process vs remote
execution. Real Session, not a fake: org_mode ScopedSession needs session.info and a listener only a real Session gives."""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

import cowork.channels.runtime as runtime_mod
from cowork.channels.runtime import AntonChannelRuntime
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.db.session import get_open_session
from cowork.turnqueue.remote_turn import RemoteTurnFailed


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

    # _turn_stream calls ResponsesHandler._stage_remote_workspace_files
    # directly — stub the class attribute, not a name in runtime_mod.
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

    async def scenario():
        session = get_open_session()
        try:
            scoped = ScopedSession(session, TenantScope(org_mode=True, org_id="org-a"))
            conversation = type("C", (), {"id": conv_id})()
            stream = await runtime._turn_stream(
                harness, "anton", scoped, conversation, [{"type": "text", "text": "hi"}],
                "hi", None, turn_rows,
            )
            with pytest.raises(StopAsyncIteration):
                await stream.__anext__()
        finally:
            session.close()

    asyncio.run(scenario())

    assert staged == [conv_id]
    assert captured["conv_id"] == conv_id
    assert captured["org_id"] == "org-a"
    assert captured["input_text"] == "hi"
    assert captured["turn_rows"] is turn_rows
