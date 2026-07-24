"""Backend selection for turn-stream buffers.

Configured via ``StreamSettings`` (common/settings/app_settings.py):
  - ``backend`` (env ``COWORK_STREAM_BACKEND``, default ``file``) —
    ``file`` = FileStreamBuffer (desktop + single-instance cloud);
    ``redis`` = RedisStreamBuffer (multi-instance cloud, WIP).
  - ``dir`` (env ``COWORK_STREAMS_DIR``, default ``~/.cowork/streams``) —
    root for file-backed buffers.

The rest of the app only calls ``new_buffer()`` / ``get_streams_dir()``,
so swapping the backend is a one-line settings change with no call-site churn.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from cowork.common.settings.app_settings import StreamSettings
from cowork.streaming.buffer import (
    FileStreamBuffer,
    RedisStreamBuffer,
    StreamBuffer,
    conversation_dir,
    turn_buffer_path,
)


def get_backend() -> str:
    return (StreamSettings().backend or "file").strip().lower()


def get_streams_dir() -> Path:
    return Path(StreamSettings().dir)


def new_buffer(conversation_id: str, turn_id: int) -> StreamBuffer:
    """Construct the buffer for a new turn on the configured backend."""
    backend = get_backend()
    if backend == "redis":
        # WIP — raises NotImplementedError until the cloud move wires it.
        return RedisStreamBuffer(conversation_id=conversation_id, turn_id=turn_id)
    return FileStreamBuffer(turn_buffer_path(get_streams_dir(), conversation_id, turn_id))


def remove_conversation_buffers(conversation_id: str) -> None:
    """Delete a conversation's on-disk turn buffers.

    ponytail: file backend only — the Redis backend (WIP) stores buffers as keys,
    not files, so this is a no-op there; add key deletion when Redis ships.
    """
    if get_backend() != "file":
        return
    # Allowlist to a single safe path segment before it reaches the filesystem —
    # a conversation id is a UUID; `_safe_segment` alone lets `..` through.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", conversation_id):
        return
    shutil.rmtree(conversation_dir(get_streams_dir(), conversation_id), ignore_errors=True)
