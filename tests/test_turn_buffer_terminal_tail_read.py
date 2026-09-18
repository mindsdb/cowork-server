"""latest_terminal_reason reads only the buffer file's tail.

The terminal record is always the buffer's last line, so a full-file
replay isn't needed — reading the tail is enough and doesn't get slower
as a turn's history grows.
"""
from __future__ import annotations

import asyncio

from cowork.streaming.buffer import FileStreamBuffer, turn_buffer_path
from cowork.streaming.recovery import latest_terminal_reason

# Comfortably larger than recovery._TAIL_READ_BYTES (8192) once written, so a
# tail-only read would miss the terminal if it fell back to a full replay.
_MANY_RECORDS = 2000


def _write(path, n_records, close_reason=None):
    buf = FileStreamBuffer(path)

    async def _go():
        for i in range(n_records):
            await buf.append("sse", {"sse": f"event: response.output_text.delta\ndata: {{\"delta\": \"{i}\"}}\n\n"})
        if close_reason is not None:
            await buf.close(close_reason)

    asyncio.run(_go())


def test_finds_the_terminal_in_a_file_far_larger_than_the_tail_window(tmp_path):
    path = turn_buffer_path(tmp_path, "conv-1", 0)
    _write(path, _MANY_RECORDS, close_reason="completed")

    assert path.stat().st_size > 8192
    assert latest_terminal_reason(path) == "completed"


def test_returns_none_for_a_large_unterminated_file(tmp_path):
    path = turn_buffer_path(tmp_path, "conv-1", 0)
    _write(path, _MANY_RECORDS, close_reason=None)

    assert path.stat().st_size > 8192
    assert latest_terminal_reason(path) is None


def test_returns_none_for_a_missing_file(tmp_path):
    assert latest_terminal_reason(tmp_path / "nope.jsonl") is None


def test_returns_none_for_an_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.touch()
    assert latest_terminal_reason(path) is None


def test_tolerates_a_half_written_last_line(tmp_path):
    # A crash mid-write of the NEXT record after a real one leaves a
    # truncated trailing line — must not raise, and must not misread it.
    path = turn_buffer_path(tmp_path, "conv-1", 0)
    _write(path, 5, close_reason=None)
    with path.open("a", encoding="utf-8") as f:
        f.write('{"seq": 5, "ts": "now", "type": "sse", "data": {"sse": "trun')  # cut off, no newline

    assert latest_terminal_reason(path) is None


def test_a_short_completed_file_still_resolves(tmp_path):
    # Sanity: the tail-window optimization must not break the common,
    # well-under-8KB case that dominates real usage.
    path = turn_buffer_path(tmp_path, "conv-1", 0)
    _write(path, 3, close_reason="cancelled")

    assert latest_terminal_reason(path) == "cancelled"
