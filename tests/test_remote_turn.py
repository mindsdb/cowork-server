"""remote_turn_events: the channel-turn counterpart of _produce_remote's own
inner generator (cowork/handlers/responses.py) — same stream_remote_replies
kind-dispatch, built separately rather than shared (see the plan's
correction note). Every test replaces ResponsesHandler wholesale with a
fake stand-in (all 4 methods remote_turn_events actually calls), so these
tests exercise remote_turn_events's OWN dispatch logic in isolation — the
real ResponsesHandler helpers already have their own coverage in
tests/test_responses_remote_backend.py, no need to re-prove them here."""
from __future__ import annotations

from uuid import uuid4

import pytest

import cowork.turnqueue.remote_turn as remote_turn_mod
from cowork.turnqueue.remote_turn import RemoteTurnFailed, remote_turn_events


class _FakeScope:
    org_id = "org-123"
    user_id = None


class _FakeSession:
    scope = _FakeScope()


def _fake_handler(monkeypatch, *, persist_turn_memory=None, remote_artifacts_context=None):
    """Replace ResponsesHandler in remote_turn_mod's own namespace with a
    stand-in exposing only the 4 staticmethods remote_turn_events calls —
    monkeypatching the bare name it was imported under, resolved at call
    time, exactly like every other name this plan's tests stub out."""
    monkeypatch.setattr(remote_turn_mod, "ResponsesHandler", type(
        "FakeResponsesHandler", (), {
            "_remote_artifacts_context": staticmethod(remote_artifacts_context or (lambda s, c: None)),
            "_remote_history": staticmethod(lambda s, c: []),
            "_remote_workspace": staticmethod(lambda s, c: {}),
            "_persist_turn_memory": staticmethod(persist_turn_memory or (lambda s, c, e: None)),
        },
    ))


async def _drain(gen):
    events = []
    try:
        async for event in gen:
            events.append(event)
    except RemoteTurnFailed as exc:
        return events, exc
    return events, None


@pytest.mark.asyncio
async def test_turn_delta_becomes_a_stream_text_delta(monkeypatch):
    _fake_handler(monkeypatch)

    async def fake_replies(**kwargs):
        yield "turn_delta", {"text": "hi"}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    turn_rows = []
    events, failure = await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=turn_rows,
    ))

    assert failure is None
    assert len(events) == 1
    assert events[0].text == "hi"


@pytest.mark.asyncio
async def test_turn_failed_raises_with_the_workers_code_and_message(monkeypatch):
    _fake_handler(monkeypatch)

    async def fake_replies(**kwargs):
        yield "turn_failed", {"code": "anton_error", "message": "boom"}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    events, failure = await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=[],
    ))

    assert failure is not None
    assert failure.code == "anton_error"
    assert failure.message == "boom"


@pytest.mark.asyncio
async def test_turn_failed_falls_back_to_the_generic_message_and_code(monkeypatch):
    _fake_handler(monkeypatch)

    async def fake_replies(**kwargs):
        yield "turn_failed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    from cowork.handlers.turn_errors import GENERIC_TURN_ERROR_CODE, GENERIC_TURN_ERROR_MESSAGE

    _, failure = await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=[],
    ))

    assert failure.code == GENERIC_TURN_ERROR_CODE
    assert failure.message == GENERIC_TURN_ERROR_MESSAGE


@pytest.mark.asyncio
async def test_turn_history_mutates_the_callers_turn_rows_list(monkeypatch):
    _fake_handler(monkeypatch)

    rows = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "scratchpad", "input": {"code": "1"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "1"}]},
    ]

    async def fake_replies(**kwargs):
        yield "turn_history", {"rows": rows}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    turn_rows = []
    await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=turn_rows,
    ))

    assert turn_rows == rows


@pytest.mark.asyncio
async def test_turn_memory_calls_persist_turn_memory(monkeypatch):
    calls = []
    _fake_handler(monkeypatch, persist_turn_memory=lambda s, c, e: calls.append(e))

    async def fake_replies(**kwargs):
        yield "turn_memory", {"entries": [{"key": "likes", "value": "pizza"}]}
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=[],
    ))

    assert calls == [[{"key": "likes", "value": "pizza"}]]


@pytest.mark.asyncio
async def test_no_artifacts_context_skips_indexing_without_error(monkeypatch):
    _fake_handler(monkeypatch)

    async def fake_replies(**kwargs):
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    events, failure = await _drain(remote_turn_events(
        session=_FakeSession(), conv_id=uuid4(), org_id="org-123", user_id=None,
        input_text="hi", model="anton", turn_rows=[],
    ))

    assert failure is None
    assert events == []
