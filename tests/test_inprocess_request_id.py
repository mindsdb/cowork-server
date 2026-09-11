"""A desktop turn that fails must hand the user something to quote.

The remote/cloud producer already attaches its pod correlation id to every
failure frame. The in-process producer — the one the desktop sidecar runs — had
no id at all, so a local user saw the generic message and their only move was
to open the log by hand and guess which traceback was theirs.

The id also has to reach the log RECORD, not just the message text: the
formatter renders `%(request_context)s` from a `request_id` attribute, and
nothing set one, so that placeholder was dead.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

import cowork.handlers.responses as responses_mod
from cowork.common.logger import CustomFormatter, setup_console_handler
from cowork.handlers.responses import ResponsesHandler


class _RecBuffer:
    def __init__(self) -> None:
        self.frames: list[str] = []
        self.closed: str | None = None
        # The seal reads this to decide whether the turn already terminated;
        # without it the guard takes its getattr default and never fires.
        self.is_closed = False

    @property
    def latest_seq(self) -> int:
        return len(self.frames)

    async def append(self, type_, data):
        self.frames.append(data.get("sse", ""))
        return len(self.frames)

    async def close(self, reason, extra=None):
        self.closed = reason
        self.is_closed = True


def _failing_handler(monkeypatch, saved: dict, exc: Exception):
    """A handler whose harness raises, so _run_turn takes its failure branch."""
    handler = object.__new__(ResponsesHandler)
    handler.principal = object()

    class FakeConversationService:
        def __init__(self, session):
            pass

        def get_conversation(self, conv_id):
            return object()

        def save_user_message(self, conv_id, content, *, created_at=None, pending=False):
            saved["user"] = content
            return SimpleNamespace(id=uuid4())

        def finalize_pending(self, conv_id, message_id=None):
            saved["finalized"] = True

        def save_assistant_turn(self, conv_id, text, events, harness=None, tool_rows=None):
            saved["events"] = events

        def repair_image_content(self, conv_id):
            # Without this the content-recovery branch takes its repair-FAILED
            # path, which logs a different line than the one under test.
            saved["repaired"] = True
            return [uuid4()]

    class FakeSession:
        def close(self):
            pass

    async def formatter(stream, model, event_sink):
        raise exc
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(responses_mod, "ConversationService", FakeConversationService)
    monkeypatch.setattr(responses_mod, "ScopedSession", lambda s, scope: FakeSession())
    monkeypatch.setattr(responses_mod, "get_open_session", lambda: None)
    monkeypatch.setattr(responses_mod, "scope_from_principal", lambda p: None)
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: SimpleNamespace(
        stream_response=lambda **kwargs: None, formatter=formatter,
    ))
    return handler


def _run(handler, buffer):
    return asyncio.run(handler._run_turn(
        conv_id=uuid4(), harness_input=[], original_content="hi", model="anton",
        disabled=None, harness_name="anton", harness_id="anton", buffer=buffer,
    ))


def _failed_payload(saved: dict) -> dict:
    return [e for e in saved["events"] if e.get("type") == "response.failed"][-1]


def test_inprocess_failure_carries_a_request_id(monkeypatch):
    saved: dict = {}
    buffer = _RecBuffer()
    handler = _failing_handler(monkeypatch, saved, RuntimeError("boom"))

    _run(handler, buffer)

    request_id = _failed_payload(saved)["request_id"]
    assert isinstance(request_id, str) and request_id
    # The streamed frame and the persisted row must agree, or a reopened
    # conversation shows a different Reference than the live one did.
    failed_frames = [f for f in buffer.frames if "response.failed" in f]
    assert failed_frames
    streamed = json.loads(failed_frames[-1].split("data: ", 1)[1])
    assert streamed["request_id"] == request_id


def _deployed_records(caplog):
    """The records staging and prod would actually emit, which run at WARNING."""
    return [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_inprocess_failure_puts_the_id_on_the_log_record(monkeypatch, caplog):
    # Not just in the message text: the formatter builds %(request_context)s
    # from this attribute, so an id only in the text leaves that placeholder
    # empty and support has nothing structured to filter on.
    saved: dict = {}
    handler = _failing_handler(monkeypatch, saved, RuntimeError("boom"))

    with caplog.at_level(logging.WARNING, logger="cowork.handlers.responses"):
        _run(handler, _RecBuffer())

    request_id = _failed_payload(saved)["request_id"]
    deployed = _deployed_records(caplog)
    assert deployed
    assert all(getattr(record, "request_id", None) == request_id for record in deployed)


def test_a_content_recovery_failure_tags_its_deployed_log_lines(monkeypatch, caplog):
    # Content recovery is the curated failure that logs at WARNING, so unlike
    # the other curated codes it IS emitted in staging and prod — and its line
    # is the one that explains the turn. An untagged line there is a reference
    # the user quotes that matches nothing.
    saved: dict = {}
    handler = _failing_handler(
        monkeypatch, saved,
        RuntimeError("Invalid value: 'image_url'. Supported values are: 'input_image'"),
    )

    with caplog.at_level(logging.WARNING, logger="cowork.handlers.responses"):
        _run(handler, _RecBuffer())

    payload = _failed_payload(saved)
    assert payload["code"] == "content_recovery"
    assert saved.get("repaired"), "the repair ran, so the WARNING under test fired"
    deployed = _deployed_records(caplog)
    assert deployed
    assert all(
        getattr(record, "request_id", None) == payload["request_id"]
        for record in deployed
    )


def test_a_curated_inprocess_failure_carries_the_id_too(monkeypatch):
    # Every failure carries the id so the payload shape stays uniform, and the
    # client renders it on the generic card alone. The log line is a separate
    # matter: this branch logs below the floor the deployed environments run at.
    from anton.core.llm.provider import ProviderAuthError

    saved: dict = {}
    handler = _failing_handler(monkeypatch, saved, ProviderAuthError("nope"))

    _run(handler, _RecBuffer())

    payload = _failed_payload(saved)
    assert payload["code"] != "anton_error"
    assert payload["request_id"]
    # The id is set after five branches that each REASSIGN extra, so assert the
    # curated affordances survive beside it: a reorder in either direction then
    # fails here instead of silently dropping one of the two. Keys, not values —
    # both depend on which provider the settings resolve to.
    assert "reconnectable" in payload
    assert "provider_label" in payload


@pytest.mark.parametrize("attrs, expected", [
    ({"request_id": "corr-abc"}, "[Req:corr-abc]"),
    ({}, ""),
    ({"request_id": None}, ""),
])
def test_formatter_renders_the_request_context(attrs, expected):
    # The placeholder has to survive a record that carries no request_id,
    # which is nearly all of them — a plain logging.Formatter would raise.
    # An explicit None is one of those: the seal passes it for a producer
    # that has no correlation id to offer.
    formatter = CustomFormatter("%(name)s%(request_context)s %(message)s")
    record = logging.LogRecord(
        name="cowork.test", level=logging.ERROR, pathname=__file__, lineno=1,
        msg="turn failed", args=(), exc_info=None,
    )
    for key, value in attrs.items():
        setattr(record, key, value)

    assert formatter.format(record) == f"cowork.test{expected} turn failed"


def test_a_turn_that_escapes_every_except_seals_with_the_same_id(monkeypatch):
    # A BaseException misses both except clauses, so the buffer is never
    # closed and the finally-seal is the only thing that terminates the
    # stream. That frame is the user's sole reference for the worst failure
    # the path has.
    saved: dict = {}
    handler = _failing_handler(monkeypatch, saved, KeyboardInterrupt())
    buffer = _RecBuffer()

    with pytest.raises(KeyboardInterrupt):
        _run(handler, buffer)

    assert buffer.closed == "error"
    sealed = json.loads(buffer.frames[-1].split("data: ", 1)[1])
    assert sealed["code"] == "anton_error"
    UUID(sealed["request_id"])


@pytest.mark.parametrize("rich_logging", ["false", "true"])
def test_the_console_formatter_renders_the_request_context(monkeypatch, rich_logging):
    # The console stream is the one the desktop captures into the log tail its
    # help modal offers to copy, so an id that renders only on a file handler
    # is an id the person reporting the failure never gets to quote. Both
    # branches of setup_console_handler have to carry it; rich is installed,
    # so RICH_LOGGING is one env var from being the live one.
    monkeypatch.setenv("RICH_LOGGING", rich_logging)
    formatter = setup_console_handler().formatter

    def record(**attrs):
        rec = logging.LogRecord(
            name="cowork.test", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="turn failed", args=(), exc_info=None,
        )
        for key, value in attrs.items():
            setattr(rec, key, value)
        return rec

    assert "[Req:corr-abc]" in formatter.format(record(request_id="corr-abc"))
    # The same formatter still has to render the records that carry no id,
    # which is nearly all of them.
    assert "[Req:" not in formatter.format(record())


# --- the direct-answer producer -------------------------------------------
#
# The gate can answer without delegating to the agent. That producer failed
# generically with no id at all, so a user whose direct answer broke had the
# same nothing-to-quote problem the delegated paths had fixed.
#
# The id is NOT minted inside the producer: the caller already mints one for
# `record_turn`, and the turn index exists so a replica can find the turn by
# that id. Two ids would give the user a Reference that matches the log line
# and nothing in the index.


class _RaisingOnceBuffer(_RecBuffer):
    """Fails the first append so the producer's except body raises.

    That is what makes the `finally` seal the branch actually under test: the
    normal failure path closes the buffer, and the seal's `is_closed` guard
    then makes it a no-op.
    """

    def __init__(self, fail_on: int) -> None:
        super().__init__()
        self.calls = 0
        self.fail_on = fail_on

    async def append(self, type_, data):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("buffer append failed")
        return await super().append(type_, data)


def _direct_handler(monkeypatch, *, on_save_assistant):
    handler = object.__new__(ResponsesHandler)
    handler.principal = object()
    handler.scoped = SimpleNamespace(scope=SimpleNamespace(org_id="org-1", user_id="user-1"))

    monkeypatch.setattr(responses_mod, "get_open_session", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        responses_mod, "ScopedSession", lambda session, scope: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(responses_mod, "scope_from_principal", lambda principal: object())
    monkeypatch.setattr(
        responses_mod,
        "ConversationService",
        lambda scoped: SimpleNamespace(
            save_user_message=lambda cid, content, pending=False: SimpleNamespace(id=uuid4()),
            save_assistant_turn=on_save_assistant,
        ),
    )
    return handler


def _failed_frames(buffer) -> list[dict]:
    out = []
    for frame in buffer.frames:
        payload = json.loads(frame.split("data: ", 1)[1])
        if payload.get("type") == "response.failed":
            out.append(payload)
    return out


@pytest.mark.asyncio
async def test_direct_turn_failure_quotes_the_id_the_turn_index_holds(monkeypatch, caplog):
    """The frame, the log record and `record_turn` all carry ONE id."""
    from cowork.handlers.response_routing import DIRECT_CONTEXT, RouteDecision

    def _boom(conv_id, text, events, harness=None):
        raise RuntimeError("direct answer could not be persisted")

    handler = _direct_handler(monkeypatch, on_save_assistant=_boom)
    buffer = _RecBuffer()
    recorded: dict = {}

    async def fake_start(**kwargs):
        # Run the producer for real instead of discarding it; the failure
        # branch is the whole point.
        await kwargs["producer_coro"]
        return SimpleNamespace(buffer=buffer)

    async def fake_record(conversation_id, **kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(responses_mod, "new_buffer", lambda _cid, _turn_id: buffer)
    monkeypatch.setattr(responses_mod.registry, "start", fake_start)
    monkeypatch.setattr(responses_mod, "get_backend", lambda: "redis")
    monkeypatch.setattr(responses_mod, "record_turn", fake_record)

    conv_id = UUID("d27d3533-2e4e-4021-bb5a-6e238245974c")
    with caplog.at_level(logging.WARNING, logger="cowork.handlers.responses"):
        await handler._handle_direct_response(
            request=SimpleNamespace(stream=True),
            conversation_id=conv_id,
            turn_id=3,
            original_content="Hello",
            route=RouteDecision(
                route=DIRECT_CONTEXT, reason="router_direct_response", model="m", text="Hi.",
            ),
        )

    failed = _failed_frames(buffer)
    assert len(failed) == 1
    quoted = failed[0]["request_id"]
    assert quoted, "the generic direct failure must carry an id to quote"
    # The id the user reads is the id the turn index holds, not a second one.
    assert quoted == recorded["correlation_id"]

    tagged = [r for r in caplog.records if getattr(r, "request_id", None) == quoted]
    assert tagged, "the failure must reach the log as a record attribute, not just message text"


@pytest.mark.asyncio
async def test_direct_turn_seal_quotes_the_same_id(monkeypatch):
    """A failure that escapes the except body still seals with the turn's id."""
    from cowork.handlers.response_routing import DIRECT_CONTEXT, RouteDecision

    def _boom(conv_id, text, events, harness=None):
        raise RuntimeError("direct answer could not be persisted")

    handler = _direct_handler(monkeypatch, on_save_assistant=_boom)
    # First append is the except body's failure frame; the seal's own append
    # is the second and succeeds.
    buffer = _RaisingOnceBuffer(fail_on=1)
    corr = "direct-11111111-2222-3333-4444-555555555555"

    with pytest.raises(RuntimeError):
        await handler._produce_direct(
            lifecycle=SimpleNamespace(discarded=False),
            conv_id=UUID("d27d3533-2e4e-4021-bb5a-6e238245974c"),
            original_content="Hello",
            route=RouteDecision(
                route=DIRECT_CONTEXT, reason="router_direct_response", model="m", text="Hi.",
            ),
            buffer=buffer,
            request_id=corr,
        )

    sealed = _failed_frames(buffer)
    assert len(sealed) == 1
    assert sealed[0]["request_id"] == corr
    assert buffer.closed == "error"


@pytest.mark.asyncio
async def test_non_streaming_direct_failure_answers_with_a_quotable_500(monkeypatch):
    """The other half of the direct producer. A client that omits `stream`
    lands here, and an escape used to reach the user as a bare 500 with no
    body at all — nothing to read, let alone quote."""
    from fastapi import HTTPException

    from cowork.handlers.response_routing import DIRECT_CONTEXT, RouteDecision

    handler = object.__new__(ResponsesHandler)
    handler.scoped = object()

    def _boom(conv_id, text, events, harness=None):
        raise RuntimeError("direct answer could not be persisted")

    monkeypatch.setattr(
        responses_mod,
        "ConversationService",
        lambda scoped: SimpleNamespace(
            save_user_message=lambda cid, content: SimpleNamespace(id=uuid4()),
            save_assistant_turn=_boom,
        ),
    )

    with pytest.raises(HTTPException) as caught:
        await handler._handle_direct_response(
            request=SimpleNamespace(stream=False),
            conversation_id=UUID("d27d3533-2e4e-4021-bb5a-6e238245974c"),
            turn_id=1,
            original_content="Hello",
            route=RouteDecision(
                route=DIRECT_CONTEXT, reason="router_direct_response", model="m", text="Hi.",
            ),
        )

    assert caught.value.status_code == 500
    # Same body shape the delegated non-streaming path emits, so a caller
    # never has to branch on which producer answered.
    assert caught.value.detail["code"] == "anton_error"
    assert caught.value.detail["request_id"]
