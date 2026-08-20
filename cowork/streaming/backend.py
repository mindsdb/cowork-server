"""Backend selection for turn-stream buffers.

Configured via ``StreamSettings`` (common/settings/app_settings.py):
  - ``backend`` (env ``COWORK_STREAM_BACKEND``, default ``file``) —
    ``file`` = FileStreamBuffer (desktop + single-instance cloud);
    ``redis`` = RedisStreamBuffer (multi-instance cloud).
  - ``dir`` (env ``COWORK_STREAMS_DIR``, default ``~/.cowork/streams``) —
    root for file-backed buffers.

The rest of the app only calls ``new_buffer()`` / ``get_streams_dir()``,
so swapping the backend is a one-line settings change with no call-site churn.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from cowork.common.settings.app_settings import StreamSettings
from cowork.turnqueue.redis_client import get_sync_redis
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
        return RedisStreamBuffer(conversation_id=conversation_id, turn_id=turn_id)
    return FileStreamBuffer(turn_buffer_path(get_streams_dir(), conversation_id, turn_id))


def _remove_redis_buffers(conversation_id: str) -> None:
    """Delete every turn buffer key for a conversation.

    Not tidiness: turn_id is len(messages), so truncating a conversation makes
    the next turn reuse a deleted turn's id. A surviving buffer would be
    replayed as that turn's answer.
    """
    r = get_sync_redis()
    keys = list(r.scan_iter(match=f"cowork:stream:{conversation_id}:*", count=100))
    if keys:
        r.delete(*keys)


def remove_conversation_buffers(conversation_id: str) -> None:
    """Delete a conversation's turn buffers on whichever backend is active.

    Sync on purpose: the only caller is conversation delete, which runs in a
    threadpool thread with no event loop to await on.
    """
    if get_backend() == "redis":
        _remove_redis_buffers(conversation_id)
        return
    if get_backend() != "file":
        return
    # Resolve, then require the target to stay inside the streams dir — `_safe_segment`
    # alone lets `..` through (`.` is in its allowed set), so contain it here.
    streams = os.path.realpath(get_streams_dir())
    target = os.path.realpath(conversation_dir(get_streams_dir(), conversation_id))
    if not target.startswith(streams + os.sep):
        return
    shutil.rmtree(target, ignore_errors=True)
