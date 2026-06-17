"""Boot recovery + GC for file-backed turn buffers.

These operate on the on-disk JSONL files directly (no registry), so they
work across a process restart. No-ops for the Redis backend (WIP) — that
buffer's lifetime is managed by Redis stream trimming / TTL.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from cowork.streaming.buffer import read_records
from cowork.streaming.records import TerminalReason, now_iso

logger = logging.getLogger(__name__)


def latest_terminal_reason(path: Path) -> TerminalReason | None:
    if not path.is_file():
        return None
    last: TerminalReason | None = None
    for rec in read_records(path):
        if rec.is_terminal:
            reason = rec.data.get("reason")
            if isinstance(reason, str):
                last = reason  # type: ignore[assignment]
    return last


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
