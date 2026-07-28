"""Streaming producer selection: COWORK_TURN_BACKEND=remote routes the turn
through the Redis-backed remote producer instead of the in-process detached
run; unset/"inprocess" must stay byte-identical to today.

Built without __init__ (same pattern as tests/test_turn_errors.py) so no
DB/harness setup is needed - _select_producer only touches self.scoped and
self._produce.
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

    def fake_produce_remote_turn(**kwargs):
        called.update(kwargs)
        return "remote-sentinel"

    monkeypatch.setattr(responses_mod, "produce_remote_turn", fake_produce_remote_turn)

    handler = _handler()

    def _boom(**kwargs):
        raise AssertionError("in-process _produce must not run when backend=remote")

    handler._produce = _boom

    kwargs = _kwargs()
    result = handler._select_producer(**kwargs)

    assert result == "remote-sentinel"
    assert called["conversation_id"] == str(kwargs["conv_id"])
    assert called["org_id"] == "org-123"
    assert called["user_id"] == "user-456"
    assert called["input_text"] == "hello there"
    assert called["model"] == "anton"
    assert called["buffer"] is kwargs["buffer"]


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
