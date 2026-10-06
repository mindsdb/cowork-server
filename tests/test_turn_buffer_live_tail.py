"""A live reader of a turn's file buffer parses each record once.

Every append wakes every reader of the turn. A reader that re-read the whole
file on each wake parsed a turn of N records about N * N / 2 times, on the
event loop that serves every request. A reader keeps its byte offset instead,
and holds back a last line that is not whole yet.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import cowork.streaming.buffer as buffer_mod
from cowork.streaming.buffer import FileStreamBuffer, turn_buffer_path


def _count_parses(monkeypatch) -> list[str | bytes]:
    parsed: list[str | bytes] = []

    def loads(line):
        parsed.append(line)
        return json.loads(line)

    monkeypatch.setattr(
        buffer_mod, "json",
        SimpleNamespace(loads=loads, dumps=json.dumps, JSONDecodeError=json.JSONDecodeError),
    )
    return parsed


async def test_a_live_reader_parses_each_record_once(tmp_path, monkeypatch):
    parsed = _count_parses(monkeypatch)
    buffer = FileStreamBuffer(turn_buffer_path(tmp_path, "conv-1", 0))
    received = []

    async def read():
        async for record in buffer.tail():
            received.append(record)

    reader = asyncio.create_task(read())
    records = 300
    for i in range(records):
        await buffer.append("sse", {"sse": f"frame {i}"})
        await asyncio.sleep(0)  # the reader wakes on every record, as a live stream's does
    await buffer.close("completed")
    await asyncio.wait_for(reader, timeout=5)

    assert [r.data.get("sse") for r in received[:-1]] == [f"frame {i}" for i in range(records)]
    assert received[-1].type == "Done"
    assert len(parsed) == records + 1, f"{len(parsed)} parses for {records + 1} records"


async def test_a_line_written_in_two_pieces_is_parsed_once_it_is_whole(tmp_path, monkeypatch):
    """A writer in another process (or one that crashed mid-write) can leave
    the file ending in part of a line. The reader waits for the rest instead
    of parsing the part, then reads the line once."""
    parsed = _count_parses(monkeypatch)
    path = turn_buffer_path(tmp_path, "conv-1", 0)
    buffer = FileStreamBuffer(path)
    await buffer.append("sse", {"sse": "whole"})
    split = json.dumps({"seq": 1, "ts": "", "type": "sse", "data": {"sse": "split"}}) + "\n"
    with path.open("a", encoding="utf-8") as other_writer:
        other_writer.write(split[:20])
    received = []

    async def read():
        async for record in buffer.tail():
            received.append(record.data.get("sse") or record.type)

    reader = asyncio.create_task(read())
    await asyncio.sleep(0.05)
    seen_before_the_rest = list(received)
    with path.open("a", encoding="utf-8") as other_writer:
        other_writer.write(split[20:])
    await buffer.close("completed")
    await asyncio.wait_for(reader, timeout=5)

    assert seen_before_the_rest == ["whole"]
    assert received == ["whole", "split", "Done"]
    assert len(parsed) == 3, parsed
