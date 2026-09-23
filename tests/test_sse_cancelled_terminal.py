"""A cancelled turn ends its SSE stream with a response.cancelled frame.

Without it a Stop and a dropped connection look the same on the wire: both
close with no response.completed/failed, and the client can't tell a turn the
user stopped from one that broke.
"""
from __future__ import annotations

import json

import pytest

from cowork.handlers.responses import sse_from_buffer
from cowork.streaming.buffer import FileStreamBuffer, turn_buffer_path

_CREATED = "event: response.created\ndata: {}\n\n"


async def _sealed_buffer(tmp_path, reason: str) -> FileStreamBuffer:
    buf = FileStreamBuffer(turn_buffer_path(tmp_path, "conv-1", 0))
    await buf.append("sse", {"sse": _CREATED})
    await buf.close(reason)
    return buf


def _event_types(frames: list[str]) -> list[str]:
    types = []
    for frame in frames:
        data_line = next(line for line in frame.split("\n") if line.startswith("data:"))
        types.append(json.loads(data_line[5:]).get("type"))
    return types


async def test_cancelled_turn_ends_with_a_cancelled_frame(tmp_path):
    buf = await _sealed_buffer(tmp_path, "cancelled")

    frames = [f async for f in sse_from_buffer(buf, 0)]

    assert frames[0] == _CREATED
    assert frames[-1].startswith("event: response.cancelled\n")
    assert _event_types(frames[1:]) == ["response.cancelled"]


async def test_reconnect_to_a_cancelled_turn_still_gets_the_frame(tmp_path):
    """A tab or device that reattaches after the Stop replays from its last
    seq, past every normal record, and must still learn it was a cancel."""
    buf = await _sealed_buffer(tmp_path, "cancelled")

    frames = [f async for f in sse_from_buffer(buf, 1)]

    assert _event_types(frames) == ["response.cancelled"]


@pytest.mark.parametrize("reason", ["completed", "error", "interrupted", "restart"])
async def test_other_terminals_add_no_frame(tmp_path, reason):
    buf = await _sealed_buffer(tmp_path, reason)

    frames = [f async for f in sse_from_buffer(buf, 0)]

    assert frames == [_CREATED]
