"""The in-process producers (_run_turn, _produce_direct) and the
data-vault probe (ProbeHandler) hand the browser the persisted assistant
message's id on the completion frame, so a client can rekey delete/sidecar/
usage-notice logic off it instead of a positional index that doesn't survive
a lazily-paginated conversation. See test_responses_remote_backend.py for
the remote-backend producer's equivalent coverage.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from cowork.handlers.responses import ResponsesHandler


def _payload(frame: str) -> dict:
    return json.loads(frame.split("data: ", 1)[1].strip())


def _handler_with_formatter(turn_formatter) -> ResponsesHandler:
    """Same construction as test_turn_errors.py's _handler_with_raising_formatter
    (built without __init__, no DB/harness setup needed)."""
    handler = object.__new__(ResponsesHandler)
    handler.principal = None

    async def _stream_response(*, conversation, input, model=None, reasoning_effort=None,
                                disabled_connections=None, trace_tags=None, trace_metadata=None):
        if False:
            yield

    class _Harness:
        id = "anton"
        formatter = staticmethod(turn_formatter)
        stream_response = staticmethod(_stream_response)

    handler.harness = _Harness()
    return handler


class _Buffer:
    def __init__(self):
        self.frames: list[str] = []

    async def append(self, _kind, data):
        self.frames.append(data["sse"])

    async def close(self, _status):
        self.frames.append(f"CLOSE:{_status}")


async def _run(handler, *, assistant_message_id, user_message_id=None) -> _Buffer:
    buffer = _Buffer()
    conv_id = uuid4()
    with (
        patch("cowork.handlers.responses.get_open_session", return_value=MagicMock()),
        patch("cowork.handlers.responses.ConversationService") as conv_svc,
        patch("cowork.handlers.responses.get_harness", return_value=handler.harness),
    ):
        conv_svc.return_value.get_conversation.return_value = MagicMock()
        conv_svc.return_value.save_user_message.return_value = SimpleNamespace(
            id=user_message_id or uuid4()
        )
        conv_svc.return_value.save_assistant_turn.return_value = (
            SimpleNamespace(id=assistant_message_id) if assistant_message_id else None
        )
        await handler._produce(
            conv_id=conv_id,
            harness_input=[{"type": "text", "text": "hi"}],
            original_content="hi",
            model="anton",
            disabled=None,
            harness_name="anton",
            harness_id="anton",
            buffer=buffer,
        )
    return buffer


@pytest.mark.asyncio
async def test_run_turn_completed_frame_carries_the_assistant_message_id():
    real_id = uuid4()

    async def formatter(stream, model, event_sink):
        yield "event: response.created\ndata: {}\n\n"
        yield 'event: response.output_text.delta\ndata: {"delta": "hi"}\n\n'
        yield ('event: response.completed\ndata: {"type": "response.completed", '
               '"response": {"output": [{"content": [{"type": "output_text", "text": "hi"}]}]}}\n\n')

    buffer = await _run(_handler_with_formatter(formatter), assistant_message_id=real_id)
    completed = [f for f in buffer.frames if f.startswith("event: response.completed")]
    assert len(completed) == 1
    assert _payload(completed[0])["assistant_message_id"] == str(real_id)


@pytest.mark.asyncio
async def test_run_turn_completed_frame_omits_the_id_when_nothing_persisted():
    async def formatter(stream, model, event_sink):
        yield ('event: response.completed\ndata: {"type": "response.completed", '
               '"response": {"output": []}}\n\n')

    buffer = await _run(_handler_with_formatter(formatter), assistant_message_id=None)
    completed = [f for f in buffer.frames if f.startswith("event: response.completed")]
    assert len(completed) == 1
    assert "assistant_message_id" not in _payload(completed[0])


@pytest.mark.asyncio
async def test_run_turn_failed_frame_carries_the_id_when_something_persisted():
    real_id = uuid4()

    async def formatter(stream, model, event_sink):
        yield "event: response.created\ndata: {}\n\n"
        raise RuntimeError("boom")

    buffer = await _run(_handler_with_formatter(formatter), assistant_message_id=real_id)
    failed = [f for f in buffer.frames if f.startswith("event: response.failed")]
    assert len(failed) == 1
    assert _payload(failed[0])["assistant_message_id"] == str(real_id)


# ── A delta frame that quotes a frame-type name must not be mistaken for
# the real thing (model output is untrusted input) ─────────────────────────

@pytest.mark.asyncio
async def test_run_turn_ignores_a_delta_frame_whose_text_quotes_a_frame_type_name():
    real_id = uuid4()
    # The delta's own text contains the literal substring "response.completed"
    # — e.g. the model discussing the API, or a prompt-injected instruction —
    # inside a frame that is NOT the real completion. A naive substring match
    # over the whole SSE text (rather than the frame's own `event:` line)
    # would mistake this for the terminal frame: persist a partial turn early
    # and rewrite this delta's event line to `response.completed`.
    async def formatter(stream, model, event_sink):
        yield "event: response.created\ndata: {}\n\n"
        yield (
            'event: response.output_text.delta\ndata: '
            '{"type": "response.output_text.delta", '
            '"delta": "note: the stream ends with event: response.completed"}\n\n'
        )
        yield ('event: response.completed\ndata: {"type": "response.completed", '
               '"response": {"output": [{"content": [{"type": "output_text", "text": "hi"}]}]}}\n\n')

    handler = _handler_with_formatter(formatter)
    buffer = _Buffer()
    conv_id = uuid4()
    with (
        patch("cowork.handlers.responses.get_open_session", return_value=MagicMock()),
        patch("cowork.handlers.responses.ConversationService") as conv_svc,
        patch("cowork.handlers.responses.get_harness", return_value=handler.harness),
    ):
        conv_svc.return_value.get_conversation.return_value = MagicMock()
        conv_svc.return_value.save_user_message.return_value = SimpleNamespace(id=uuid4())
        conv_svc.return_value.save_assistant_turn.return_value = SimpleNamespace(id=real_id)
        await handler._produce(
            conv_id=conv_id, harness_input=[{"type": "text", "text": "hi"}],
            original_content="hi", model="anton", disabled=None,
            harness_name="anton", harness_id="anton", buffer=buffer,
        )

    delta_frames = [f for f in buffer.frames if f.startswith("event: response.output_text.delta")]
    completed_frames = [f for f in buffer.frames if f.startswith("event: response.completed")]
    assert len(delta_frames) == 1, "the delta frame must pass through unchanged, not be consumed/rewritten"
    assert "response.completed" in _payload(delta_frames[0])["delta"], "delta text itself is untouched"
    assert len(completed_frames) == 1, "only the real terminal frame counts as a completion"
    assert _payload(completed_frames[0])["assistant_message_id"] == str(real_id)
    # The early-persist path must only fire on the real completion,
    # not once per frame that happens to mention it in its own text.
    assert conv_svc.return_value.save_assistant_turn.call_count == 1


@pytest.mark.asyncio
async def test_run_turn_failed_frame_omits_the_id_when_nothing_persisted():
    async def formatter(stream, model, event_sink):
        if False:
            yield  # forces this to be a real async generator, not a bare coroutine
        raise RuntimeError("boom")

    buffer = await _run(_handler_with_formatter(formatter), assistant_message_id=None)
    failed = [f for f in buffer.frames if f.startswith("event: response.failed")]
    assert len(failed) == 1
    assert "assistant_message_id" not in _payload(failed[0])


# ── _produce_direct: both rows already persisted before any frame is built ──

async def _run_direct(*, assistant_message_id, route_text="ok", user_message_id=None):
    from cowork.streaming.registry import TurnLifecycle
    from cowork.handlers.response_routing import RouteDecision

    handler = object.__new__(ResponsesHandler)
    handler.principal = None
    buffer = _Buffer()
    route = RouteDecision(route="direct_context", text=route_text, model="anton", reason="test")

    with (
        patch("cowork.handlers.responses.get_open_session", return_value=MagicMock()),
        patch("cowork.handlers.responses.ConversationService") as conv_svc,
    ):
        conv_svc.return_value.save_user_message.return_value = SimpleNamespace(
            id=user_message_id or uuid4()
        )
        conv_svc.return_value.save_assistant_turn.return_value = (
            SimpleNamespace(id=assistant_message_id) if assistant_message_id else None
        )
        await handler._produce_direct(
            lifecycle=TurnLifecycle(),
            conv_id=uuid4(),
            original_content="hi",
            route=route,
            buffer=buffer,
        )
    return buffer


@pytest.mark.asyncio
async def test_produce_direct_completed_frame_carries_the_assistant_message_id():
    real_id = uuid4()
    buffer = await _run_direct(assistant_message_id=real_id)
    completed = [f for f in buffer.frames if f.startswith("event: response.completed")]
    assert len(completed) == 1
    assert _payload(completed[0])["assistant_message_id"] == str(real_id)


@pytest.mark.asyncio
async def test_produce_direct_completed_frame_omits_the_id_when_nothing_persisted():
    buffer = await _run_direct(assistant_message_id=None)
    completed = [f for f in buffer.frames if f.startswith("event: response.completed")]
    assert len(completed) == 1
    assert "assistant_message_id" not in _payload(completed[0])


# ── ProbeHandler: the data-vault/connector-credential-probe stream ──────────
#
# Full end-to-end coverage of ProbeHandler.run() needs a workspace, LLM
# client, and submission store (test_probe_user_label.py notes there's no
# existing harness for it). The "submission expired" branch is the one path
# that needs none of that — just store.get(...) returning falsy — so it's
# enough to exercise the real _persist_once/_completed closures without
# building the full probe harness.

async def _run_probe_expired_submission(*, conversation_id, assistant_message_id):
    from cowork.handlers.probe import ProbeHandler

    with (
        patch("cowork.handlers.probe.store") as fake_store,
        patch("cowork.handlers.probe.ConversationService") as conv_svc,
    ):
        fake_store.get.return_value = None
        conv_svc.return_value.get_conversation.return_value = SimpleNamespace(
            id=conversation_id, project=SimpleNamespace(path="/tmp/proj")
        )
        conv_svc.return_value.save_assistant_turn.return_value = (
            SimpleNamespace(id=assistant_message_id) if assistant_message_id else None
        )
        handler = ProbeHandler(session=MagicMock())
        frames = [chunk async for chunk in handler.run(
            submission_id="sub-1", connector_id="postgres", method=None,
            name="my-db", conversation_id=str(conversation_id),
        )]
    return frames


@pytest.mark.asyncio
async def test_probe_completed_frame_carries_the_assistant_message_id():
    real_id = uuid4()
    frames = await _run_probe_expired_submission(conversation_id=uuid4(), assistant_message_id=real_id)
    completed = [f for f in frames if f.startswith("event: response.completed")]
    assert len(completed) == 1
    assert _payload(completed[0])["assistant_message_id"] == str(real_id)


@pytest.mark.asyncio
async def test_probe_completed_frame_omits_the_id_when_nothing_persisted():
    # The common case: probe turns are usually short/no-text-yet failures, so
    # save_assistant_turn's own early-return (no text) means no id at all —
    # expected, not a bug (see ProbeHandler._save_assistant_turn's removal
    # in favor of the inline _persist_once/_completed closures).
    frames = await _run_probe_expired_submission(conversation_id=uuid4(), assistant_message_id=None)
    completed = [f for f in frames if f.startswith("event: response.completed")]
    assert len(completed) == 1
    assert "assistant_message_id" not in _payload(completed[0])


# ── The user row's id rides response.created ────────────────────────────────
#
# The client appends the user's own message optimistically on send, so
# without this it holds a row with no id for the whole live turn — and every
# consumer keyed on message id (turn delete, the step sidecar, the
# usage-notice anchor) silently degrades for exactly the turn the user is
# looking at.

@pytest.mark.asyncio
async def test_run_turn_created_frame_carries_the_user_message_id():
    user_id = uuid4()

    async def formatter(stream, model, event_sink):
        yield "event: response.created\ndata: {}\n\n"
        yield ('event: response.completed\ndata: {"type": "response.completed", '
               '"response": {"output": [{"content": [{"type": "output_text", "text": "hi"}]}]}}\n\n')

    buffer = await _run(
        _handler_with_formatter(formatter), assistant_message_id=uuid4(), user_message_id=user_id,
    )
    created = [f for f in buffer.frames if f.startswith("event: response.created")]
    assert len(created) == 1
    assert _payload(created[0])["user_message_id"] == str(user_id)


@pytest.mark.asyncio
async def test_produce_direct_created_frame_carries_the_user_message_id():
    user_id = uuid4()
    buffer = await _run_direct(assistant_message_id=uuid4(), user_message_id=user_id)
    created = [f for f in buffer.frames if f.startswith("event: response.created")]
    assert len(created) == 1
    assert _payload(created[0])["user_message_id"] == str(user_id)


@pytest.mark.asyncio
async def test_created_frame_omits_the_user_message_id_when_no_user_row_was_persisted():
    """Absent, not null — the same convention assistant_message_id uses.

    The probe producer persists no user message at all, so its created frame
    must carry no `user_message_id` key rather than an explicit null the
    client would have to special-case.
    """
    frame = ResponsesHandler._inject_created(
        "event: response.created\ndata: {}\n\n", uuid4(), "anton", None,
    )
    assert "user_message_id" not in _payload(frame)
