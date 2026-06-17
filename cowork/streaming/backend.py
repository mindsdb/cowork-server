"""Backend selection for turn-stream buffers.

`COWORK_STREAM_BACKEND` chooses the implementation:
  - ``file`` (default) — FileStreamBuffer under ``COWORK_STREAMS_DIR``
    (default ``~/.cowork/streams``). Desktop + single-instance cloud.
  - ``redis`` — WIP; RedisStreamBuffer for multi-instance cloud.

The rest of the app only calls ``new_buffer()`` / ``get_streams_dir()``,
so swapping the backend is a one-line env change with no call-site churn.
"""
from __future__ import annotations

import os
from pathlib import Path

from cowork.streaming.buffer import (
    FileStreamBuffer,
    RedisStreamBuffer,
    StreamBuffer,
    turn_buffer_path,
)


def get_backend() -> str:
    return (os.environ.get("COWORK_STREAM_BACKEND") or "file").strip().lower()


def get_streams_dir() -> Path:
    override = os.environ.get("COWORK_STREAMS_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cowork" / "streams"


def new_buffer(conversation_id: str, turn_id: int) -> StreamBuffer:
    """Construct the buffer for a new turn on the configured backend."""
    backend = get_backend()
    if backend == "redis":
        # WIP — raises NotImplementedError until the cloud move wires it.
        return RedisStreamBuffer(conversation_id=conversation_id, turn_id=turn_id)
    return FileStreamBuffer(turn_buffer_path(get_streams_dir(), conversation_id, turn_id))
