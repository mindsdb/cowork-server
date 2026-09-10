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
from uuid import uuid4

import pytest

import cowork.handlers.responses as responses_mod
from cowork.common.logger import CustomFormatter, setup_console_handler
from cowork.handlers.responses import ResponsesHandler


class _RecBuffer:
    def __init__(self) -> None:
        self.frames: list[str] = []
        self.closed: str | None = None

    @property
    def latest_seq(self) -> int:
        return len(self.frames)

    async def append(self, type_, data):
        self.frames.append(data.get("sse", ""))
        return len(self.frames)

    async def close(self, reason, extra=None):
        self.closed = reason


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


def test_inprocess_failure_puts_the_id_on_the_log_record(monkeypatch, caplog):
    # Not just in the message text: the formatter builds %(request_context)s
    # from this attribute, so an id only in the text leaves that placeholder
    # empty and support has nothing structured to filter on.
    saved: dict = {}
    handler = _failing_handler(monkeypatch, saved, RuntimeError("boom"))

    with caplog.at_level(logging.WARNING, logger="cowork.handlers.responses"):
        _run(handler, _RecBuffer())

    request_id = _failed_payload(saved)["request_id"]
    assert any(
        getattr(record, "request_id", None) == request_id
        for record in caplog.records
        if record.levelno >= logging.WARNING
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


def test_the_console_formatter_renders_the_request_context(monkeypatch):
    # The console stream is the one the desktop captures into the log tail its
    # help modal offers to copy, so an id that renders only on a file handler
    # is an id the person reporting the failure never gets to quote.
    monkeypatch.setenv("RICH_LOGGING", "false")
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
