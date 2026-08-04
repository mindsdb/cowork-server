"""Streaming producer selection: COWORK_TURN_BACKEND=remote routes the turn
through the Redis-backed remote producer instead of the in-process detached
run; unset/"inprocess" must stay byte-identical to today.

Built without __init__ (same pattern as tests/test_turn_errors.py) so no
DB/harness setup is needed - the DB-touching pieces (_remote_history and
the persistence layer inside _produce_remote) are stubbed.
"""
from __future__ import annotations

from uuid import uuid4

import pytest

import cowork.handlers.responses as responses_mod
from cowork.handlers.responses import ResponsesHandler


class _FakeScope:
    org_id = "org-123"
    user_id = "user-456"


class _FakeScoped:
    scope = _FakeScope()


def _handler() -> ResponsesHandler:
    handler = object.__new__(ResponsesHandler)
    handler.scoped = _FakeScoped()
    return handler


def _kwargs(**overrides) -> dict:
    base = dict(
        conv_id=uuid4(),
        harness_input=[{"type": "text", "text": "hello there"}],
        original_content="hello there",
        model="anton",
        disabled=None,
        harness_name="anton",
        harness_id="anton",
        buffer=object(),
        trace_tags=None,
        trace_metadata=None,
    )
    base.update(overrides)
    return base


def test_remote_backend_selected(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")

    called = {}

    handler = _handler()

    def _boom(**kwargs):
        raise AssertionError("in-process _produce must not run when backend=remote")

    handler._produce = _boom

    def fake_produce_remote(**kwargs):
        called.update(kwargs)
        return "remote-sentinel"

    handler._produce_remote = fake_produce_remote

    kwargs = _kwargs()
    result = handler._select_producer(**kwargs)

    assert result == "remote-sentinel"
    assert called["conv_id"] == kwargs["conv_id"]
    assert called["input_text"] == "hello there"
    assert called["original_content"] == "hello there"
    assert called["model"] == "anton"
    assert called["harness_id"] == "anton"
    assert called["buffer"] is kwargs["buffer"]


def _remote_handler(monkeypatch, saved):
    """Handler wired for _produce_remote with the DB layer faked out."""
    handler = _handler()
    handler.principal = object()
    handler._remote_history = lambda session, conv_id: []

    class FakeConversationService:
        def __init__(self, session):
            pass

        def get_conversation(self, conv_id):
            return object()

        def save_user_message(self, conv_id, content, *, pending=False):
            saved["user"] = content

        def finalize_pending(self, conv_id):
            saved["finalized"] = True

        def save_assistant_turn(self, conv_id, text, events, harness=None, tool_rows=None):
            saved["assistant"] = text
            saved["events"] = events
            saved["harness"] = harness

    class FakeSession:
        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(responses_mod, "ConversationService", FakeConversationService)
    monkeypatch.setattr(responses_mod, "ScopedSession", lambda s, scope: FakeSession())
    monkeypatch.setattr(responses_mod, "get_open_session", lambda: None)
    monkeypatch.setattr(responses_mod, "scope_from_principal", lambda p: None)
    return handler


@pytest.mark.asyncio
async def test_produce_remote_finalizes_user_and_persists_assistant(monkeypatch):
    # ENG-1231: the user message is persisted (pending) upstream in handle(); on
    # terminal the producer finalizes it (rejoining LLM history) and persists the
    # assistant turn. It no longer writes the user row itself.
    saved = {}
    handler = _remote_handler(monkeypatch, saved)

    async def fake_produce_remote_turn(*, on_event=None, **kwargs):
        on_event("turn_delta", {"text": "he"})
        on_event("turn_delta", {"text": "llo"})
        on_event("turn_completed", {})

    monkeypatch.setattr(responses_mod, "produce_remote_turn", fake_produce_remote_turn)

    await handler._produce_remote(
        conv_id=uuid4(), input_text="hi", original_content="hi",
        model="anton", harness_id="anton", buffer=object(),
    )

    assert saved["finalized"] is True
    assert "user" not in saved  # the producer no longer writes the user row
    assert saved["assistant"] == "hello"
    assert saved["harness"] == "anton"
    assert saved["events"][-1] == {"type": "response.completed"}


@pytest.mark.asyncio
async def test_produce_remote_persists_on_failure(monkeypatch):
    saved = {}
    handler = _remote_handler(monkeypatch, saved)

    async def fake_produce_remote_turn(*, on_event=None, **kwargs):
        on_event("turn_delta", {"text": "partial"})
        # The real producer attaches the classified (code, message) it streamed.
        on_event("turn_failed", {"error": "RuntimeError: boom",
                                 "code": "anton_error", "message": "An unexpected error occurred."})

    monkeypatch.setattr(responses_mod, "produce_remote_turn", fake_produce_remote_turn)

    await handler._produce_remote(
        conv_id=uuid4(), input_text="hi", original_content="hi",
        model="anton", harness_id="anton", buffer=object(),
    )

    assert saved["finalized"] is True
    assert saved["assistant"] == "partial"
    # Persisted events mirror the streamed response.failed frame.
    assert saved["events"][-1] == {"type": "response.failed", "code": "anton_error",
                                   "error": "An unexpected error occurred."}


@pytest.mark.parametrize("backend_env", [None, "inprocess"])
def test_inprocess_backend_selected_by_default(monkeypatch, backend_env):
    if backend_env is None:
        monkeypatch.delenv("COWORK_TURN_BACKEND", raising=False)
    else:
        monkeypatch.setenv("COWORK_TURN_BACKEND", backend_env)

    called = {}

    def fake_produce_remote_turn(**kwargs):
        raise AssertionError("remote producer must not run for the inprocess backend")

    monkeypatch.setattr(responses_mod, "produce_remote_turn", fake_produce_remote_turn)

    handler = _handler()

    def fake_produce(**kwargs):
        called.update(kwargs)
        return "inprocess-sentinel"

    handler._produce = fake_produce

    kwargs = _kwargs()
    result = handler._select_producer(**kwargs)

    assert result == "inprocess-sentinel"
    # Verbatim pass-through - none of _produce's existing args are dropped
    # or reordered by the new selection branch.
    assert called == kwargs
