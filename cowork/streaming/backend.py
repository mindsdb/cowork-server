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
import time
from pathlib import Path

from redis import Redis
from redis.exceptions import TimeoutError as RedisTimeoutError

from cowork.common.settings.app_settings import StreamSettings
from cowork.turnqueue.redis_client import (
    SYNC_CALL_BUDGET_S,
    get_sync_redis,
    redis_deadline,
    sync_connections_opened,
)
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


async def new_buffer(conversation_id: str, turn_id: int) -> StreamBuffer:
    """Open the buffer for a new turn on the configured backend.

    It starts without the records of an earlier turn that ended at the same
    turn number, which happens when that turn saved no message (its question
    was refused for a full pool, say). Called by RunRegistry.start once no
    turn of the conversation is answering.
    """
    backend = get_backend()
    if backend == "redis":
        buffer = RedisStreamBuffer(conversation_id=conversation_id, turn_id=turn_id)
        await buffer.clear_if_ended()
        return buffer
    return FileStreamBuffer(turn_buffer_path(get_streams_dir(), conversation_id, turn_id))


def _remove_redis_buffers(conversation_id: str) -> None:
    """Delete every turn buffer key for a conversation.

    Not tidiness: turn_id is len(messages), so truncating a conversation makes
    the next turn reuse a deleted turn's id. A surviving buffer would be
    replayed as that turn's answer.
    """
    r = get_sync_redis()
    keys = _scan_keys(r=r, match=f"cowork:stream:{conversation_id}:*")
    if keys:
        r.delete(*keys)


def _scan_keys(*, r: Redis, match: str) -> set[str]:
    """Every key matching ``match``, walked with SCAN under one deadline.

    A SCAN cursor means something only to the server that issued it. A page
    that went out on a connection opened after its cursor was issued (a retry,
    or a reconnect after the server dropped the pooled one) may have reached a
    new primary after a failover, so the walk starts again from cursor 0
    rather than skip keys.
    """
    keys: set[str] = set()
    with redis_deadline(budget_s=SYNC_CALL_BUDGET_S) as deadline:
        cursor = 0
        while True:
            opened = sync_connections_opened()
            next_cursor, page = r.scan(cursor=cursor, match=match, count=100)
            keys.update(page)
            if cursor != 0 and sync_connections_opened() != opened:
                if time.monotonic() >= deadline:
                    raise RedisTimeoutError("SCAN kept reconnecting and did not finish before its deadline")
                cursor = 0
                continue
            if int(next_cursor) == 0:
                return keys
            cursor = int(next_cursor)


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
