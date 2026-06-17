"""User-facing turn-error handling (ported from cowork PR #156).

When a turn dies on a cryptic provider 400 — most notably an image
arriving as an OpenAI-style ``image_url`` block instead of Anthropic's
``image`` block — the handler must surface a clean ``response.failed``
event (streaming) / 400 (non-streaming) with curated copy, while any
unmapped failure stays generic so provider internals never leak.

These tests pin the detection/mapping policy and the handler emission on
both the streaming and non-streaming paths.
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import pytest
from fastapi import HTTPException

from cowork.handlers import turn_errors as te
from cowork.handlers.responses import ResponsesHandler


# ── Detection / mapping policy ────────────────────────────────────

def test_detects_anthropic_image_url_rejection():
    exc = Exception(
        "Input tag 'image_url' found using 'type' does not match "
        "any of the expected tags: 'image'"
    )
    assert te.is_image_format_error(exc) is True


def test_detects_unsupported_image_phrasing():
    assert te.is_image_format_error(Exception("Unsupported image media type")) is True


def test_ignores_unrelated_errors():
    assert te.is_image_format_error(Exception("Internal server error")) is False
    # A tool_use 400 must NOT be misread as an image failure.
    assert te.is_image_format_error(
        Exception("tool_use ids were found without tool_result blocks")
    ) is False


def test_maps_image_error_to_curated_copy():
    result = te.friendly_turn_error(
        Exception("'image_url' does not match expected tags: 'image'")
    )
    assert result is not None
    code, message = result
    assert code == "image_format"
    assert "PNG or JPEG" in message


def test_returns_none_for_unmapped_error():
    assert te.friendly_turn_error(Exception("boom")) is None


def test_response_failed_sse_shape():
    frame = te.response_failed_sse("oops", "image_format")
    assert frame.startswith("event: response.failed\ndata: ")
    assert frame.endswith("\n\n")
    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload == {"type": "response.failed", "code": "image_format", "error": "oops"}


# ── Handler emission ──────────────────────────────────────────────

def _handler_with_raising_formatter(exc: Exception) -> ResponsesHandler:
    """A ResponsesHandler whose formatter yields one frame then raises —
    built without __init__ so no DB/harness setup is needed."""
    handler = object.__new__(ResponsesHandler)

    async def _formatter(stream, model, event_sink):
        yield "event: response.created\ndata: {}\n\n"
        raise exc

    class _Harness:
        id = "anton"
        formatter = staticmethod(_formatter)

    handler.harness = _Harness()
    return handler


async def _collect_stream(handler: ResponsesHandler) -> list[str]:
    return [frame async for frame in handler._stream(stream=None, conversation_id=uuid4(), model="anton")]


def test_stream_emits_friendly_failed_event_for_image_error():
    exc = Exception("Input tag 'image_url' ... does not match the expected tags: 'image'")
    frames = asyncio.run(_collect_stream(_handler_with_raising_formatter(exc)))
    # created frame still came through, then a clean failure — no raise.
    assert any("response.created" in f for f in frames)
    failed = [f for f in frames if "response.failed" in f]
    assert len(failed) == 1
    payload = json.loads(failed[0].split("data: ", 1)[1].strip())
    assert payload["code"] == "image_format"
    assert "PNG or JPEG" in payload["error"]


def test_stream_redacts_generic_error():
    frames = asyncio.run(_collect_stream(_handler_with_raising_formatter(Exception("psycopg2: password authentication failed for user 'admin'"))))
    failed = [f for f in frames if "response.failed" in f]
    assert len(failed) == 1
    payload = json.loads(failed[0].split("data: ", 1)[1].strip())
    assert payload["code"] == te.GENERIC_TURN_ERROR_CODE
    assert payload["error"] == te.GENERIC_TURN_ERROR_MESSAGE
    # The raw provider/internal detail must not leak.
    assert "password" not in failed[0]


def test_collect_raises_400_with_curated_message_for_image_error():
    handler = _handler_with_raising_formatter(
        Exception("'image_url' does not match the expected tags: 'image'")
    )
    with pytest.raises(HTTPException) as err:
        asyncio.run(handler._collect(stream=None, conversation_id=uuid4(), model="anton", output_item_id="msg-1"))
    assert err.value.status_code == 400
    assert "PNG or JPEG" in err.value.detail


def test_collect_raises_500_generic_for_unmapped_error():
    handler = _handler_with_raising_formatter(Exception("kaboom: secret-token-xyz"))
    with pytest.raises(HTTPException) as err:
        asyncio.run(handler._collect(stream=None, conversation_id=uuid4(), model="anton", output_item_id="msg-1"))
    assert err.value.status_code == 500
    assert err.value.detail == te.GENERIC_TURN_ERROR_MESSAGE
    assert "secret-token" not in err.value.detail
