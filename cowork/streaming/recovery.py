"""Boot recovery + GC for file-backed turn buffers.

These operate on the on-disk JSONL files directly (no registry), so they
work across a process restart. No-ops for the Redis backend (WIP) — that
buffer's lifetime is managed by Redis stream trimming / TTL.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

from cowork.common.logger import log_context
from cowork.streaming.buffer import read_records
from cowork.streaming.records import TerminalReason, TurnRecord, now_iso

logger = logging.getLogger(__name__)

# Read only the last _TAIL_READ_BYTES of a buffer file to find its terminal
# record, rather than replaying the whole thing. Terminal records fit inside
# this window; a normal SSE record may be larger and start outside it.
_TAIL_READ_BYTES = 8192


def latest_terminal_reason(path: Path) -> TerminalReason | None:
    """The buffer's terminal record, if it has one.

    A turn buffer's terminal record — if present — is always exactly its
    last line: `FileStreamBuffer.close()` appends it once, at the
    then-current end, and nothing ever appends after. So it is enough to
    read the file's tail rather than replay every record in it.
    """
    try:
        size = path.stat().st_size
        if size == 0:
            return None
        with path.open("rb") as f:
            f.seek(max(0, size - _TAIL_READ_BYTES))
            tail = f.read()
    except OSError:
        return None
    lines = [ln for ln in tail.split(b"\n") if ln.strip()]
    if not lines:
        return None
    try:
        obj = json.loads(lines[-1])
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None  # partial JSON or a tail starting inside a UTF-8 character
    rec = TurnRecord(
        seq=int(obj.get("seq", -1)), ts=str(obj.get("ts", "")),
        type=str(obj.get("type", "")), data=dict(obj.get("data") or {}),
    )
    if not rec.is_terminal:
        return None
    reason = rec.data.get("reason")
    return reason if isinstance(reason, str) else None


def seal_orphan_buffers(streams_root: Path) -> int:
    """Append a synthetic ``Interrupted`` to any buffer left open by a
    crash/restart so future tail readers get a clean end-of-stream rather
    than waiting forever. Idempotent; safe on a missing dir. Returns the
    count sealed."""
    if not streams_root.is_dir():
        return 0
    sealed = 0
    for conv_dir in streams_root.iterdir():
        if not conv_dir.is_dir():
            continue
        for path in conv_dir.glob("turn_*.jsonl"):
            try:
                if latest_terminal_reason(path) is not None:
                    continue  # already cleanly closed
            except Exception:
                logger.debug("Could not inspect %s for terminal", path, exc_info=True)
                continue
            next_seq = sum(1 for _ in read_records(path))
            try:
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "seq": next_seq, "ts": now_iso(),
                        "type": "Interrupted", "data": {"reason": "restart"},
                    }) + "\n")
                sealed += 1
            except OSError:
                logger.warning("Could not seal orphan buffer %s", path, exc_info=True)
    if sealed:
        logger.info("Sealed %d orphan turn buffer(s) on boot.", sealed)
    return sealed


def _read_buffer_events(path: Path) -> Iterator[tuple[str, dict]]:
    """Replay a turn buffer's stored SSE frames back into ``(event_type,
    data)`` pairs — the same shape the live producer's ``event_sink`` sees.
    Only ``sse``-typed records carry one (the terminal record's own type is
    not an event); anything that fails to parse is skipped, matching
    `read_records`'s tolerance for a half-written line from the crash this
    exists to recover from."""
    for rec in read_records(path):
        if rec.type != "sse":
            continue
        for block in str(rec.data.get("sse") or "").split("\n\n"):
            line = next((ln for ln in block.split("\n") if ln.startswith("data:")), None)
            if not line:
                continue
            try:
                payload = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            event_type = payload.get("type")
            if isinstance(event_type, str):
                yield event_type, payload


def seal_orphan_turns_in_history(session, streams_root: Path) -> int:
    """Write each crash-orphaned turn's streamed text into its conversation's
    history as an interrupted turn, so a reload shows it instead of nothing.

    Idempotent: an existing reply must have a durable terminal event. Retry
    restart/interrupted buffers too, so a failed DB write can recover on the
    next boot even after the separate buffer sweep closes the replay stream.
    """
    from cowork.handlers.turn_errors import (
        GENERIC_TURN_ERROR_CODE,
        INTERRUPTED_TURN_MESSAGE,
        response_failed_payload,
    )
    from cowork.services.conversations import ConversationService
    from cowork.streaming.answer_text import accumulate_answer_text

    if not streams_root.is_dir():
        return 0
    svc = ConversationService(session)
    sealed = 0
    for conv_dir in streams_root.iterdir():
        if not conv_dir.is_dir():
            continue
        try:
            conversation_id = UUID(conv_dir.name)
        except ValueError:
            logger.debug("Turn buffer dir %s is not a conversation id", conv_dir)
            continue
        for path in conv_dir.glob("turn_*.jsonl"):
            try:
                if latest_terminal_reason(path) not in {None, "restart", "interrupted"}:
                    continue  # completed, failed, or deliberately stopped
                turn_id = int(path.stem.removeprefix("turn_"))
            except Exception:
                logger.debug("Could not inspect %s for terminal", path, exc_info=True)
                continue
            try:
                collected: list[str] = []
                terminal_event = None
                for event_type, data in _read_buffer_events(path):
                    accumulate_answer_text(collected, event_type, data)
                    if event_type in {"response.completed", "response.failed"}:
                        terminal_event = data
                # A completed frame is evidence the entire answer was produced;
                # otherwise an orphan must retain a visible interruption marker.
                terminal_event = terminal_event or response_failed_payload(
                    INTERRUPTED_TURN_MESSAGE, GENERIC_TURN_ERROR_CODE,
                )
                if svc.recover_interrupted_turn(
                    conversation_id, turn_id, "".join(collected), terminal_event,
                ):
                    sealed += 1
            except ValueError:
                session.rollback()
                continue  # conversation gone — nothing to seal
            except Exception:
                session.rollback()
                logger.exception(
                    "Could not seal orphan turn %d into history",
                    turn_id,
                    extra=log_context(conversation_id=conversation_id),
                )
    if sealed:
        logger.info("Sealed %d orphan turn(s) into conversation history on boot.", sealed)
    return sealed


def gc_old_buffers(streams_root: Path, max_age_days: int) -> int:
    """Delete buffer files older than ``max_age_days`` (the buffer is a UI
    replay log; canonical history lives in the DB). Best-effort. Returns
    the count deleted."""
    if not streams_root.is_dir() or max_age_days <= 0:
        return 0
    cutoff = time.time() - max_age_days * 86400.0
    deleted = 0
    for conv_dir in list(streams_root.iterdir()):
        if not conv_dir.is_dir():
            continue
        for path in list(conv_dir.glob("turn_*.jsonl")):
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                path.unlink()
                deleted += 1
            except OSError:
                logger.debug("Could not GC buffer %s", path, exc_info=True)
        try:
            if not any(conv_dir.iterdir()):
                conv_dir.rmdir()
        except OSError:
            pass
    if deleted:
        logger.info("GC swept %d old turn buffer(s).", deleted)
    return deleted
