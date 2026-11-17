"""When the server pushes to the mobile app, and to whom.

- A task someone started in the app ends: its owner, unless someone was
  watching it live when it ended (cowork.streaming.watchers). A Stop, a
  server restart and a deleted conversation send nothing; an error, an
  interruption or a stall sends "didn't finish".
- A scheduled run ends: the schedule's owner, finished or not
  (cowork.scheduler decides which runs count).
- Someone shares an artifact and asks to notify people: each person they
  named by id, the same people MindsHub emails.

Every hook returns at once and does its work on a detached task
(dispatch.spawn), so a push never slows or fails the turn, run or share.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING
from uuid import UUID

from cowork.services.push import payloads
from cowork.services.push.dispatch import notify_user, push_enabled, spawn

if TYPE_CHECKING:
    from cowork.db.scoped import ScopedSession
    from cowork.streaming.registry import RunHandle

logger = logging.getLogger(__name__)

_TERMINAL_READ_TIMEOUT_SECONDS = 5.0


def turn_ended(handle: RunHandle) -> None:
    """Called once when a turn's task is done (RunRegistry.start)."""
    lifecycle = handle.lifecycle
    if not lifecycle.interactive or lifecycle.discarded or lifecycle.shutting_down:
        return
    if not handle.org_id or not handle.user_id or not push_enabled():
        return
    spawn(_task_finished(handle))


async def _terminal_reason(handle: RunHandle) -> str | None:
    buffer = handle.buffer
    try:
        async with asyncio.timeout(_TERMINAL_READ_TIMEOUT_SECONDS):
            async for record in buffer.tail(max(buffer.latest_seq - 1, 0)):
                if record.is_terminal:
                    return str(record.data.get("reason") or "") or None
    except Exception:
        logger.warning("Could not read how a turn ended", exc_info=True, extra={"event": "push_terminal_unreadable"})
    return None


def turn_succeeded(reason: str | None, *, timed_out: bool, stalled: bool, cancel_requested: bool) -> bool | None:
    """True for a finished turn, False for one that didn't finish, None for
    one that gets no push: a Stop, a restart, or an ending we can't read."""
    if timed_out or stalled:
        return False
    if cancel_requested or reason in (None, "cancelled", "restart"):
        return None
    if reason == "completed":
        return True
    return False


def _conversation_title(session: ScopedSession, *, conversation_id: str) -> str | None:
    from cowork.services.conversations import ConversationService

    try:
        return ConversationService(session).get_conversation(UUID(conversation_id)).topic
    except ValueError:
        return None


async def _task_finished(handle: RunHandle) -> None:
    from cowork.db.scoped import scope_for_identity
    from cowork.db.units import run_db
    from cowork.streaming.watchers import is_watched

    try:
        lifecycle = handle.lifecycle
        succeeded = turn_succeeded(
            await _terminal_reason(handle),
            timed_out=lifecycle.timed_out, stalled=lifecycle.stalled,
            cancel_requested=lifecycle.cancel_requested,
        )
        if succeeded is None or lifecycle.discarded:
            return
        if await is_watched(handle.conversation_id, handle.turn_id):
            return
        org_id, user_id = handle.org_id or "", handle.user_id or ""
        title = await run_db(
            partial(_conversation_title, conversation_id=handle.conversation_id),
            scope=scope_for_identity(org_id=org_id, user_id=user_id),
        )
        await notify_user(org_id, user_id, payloads.task_finished(
            conversation_id=handle.conversation_id, title=title, succeeded=succeeded,
        ))
    except Exception:
        logger.warning("Task push failed", exc_info=True, extra={"event": "push_task_failed"})


def scheduled_run_ended(
    *, org_id: str | None, user_id: str | None, schedule_name: str | None,
    conversation_id: str | None, succeeded: bool,
) -> None:
    """Push a scheduled run's ending to the schedule's owner."""
    if not org_id or not user_id or not push_enabled():
        return
    spawn(notify_user(org_id, user_id, payloads.scheduled_run_finished(
        conversation_id=conversation_id, schedule_name=schedule_name, succeeded=succeeded,
    )))


def artifact_shared(
    *, org_id: str | None, sharer_id: str | None, sharer_name: str | None,
    recipient_ids: Iterable[str], artifact_title: str | None, url: str | None,
) -> None:
    """Push a share to each person the sharer asked to notify, never to the
    sharer themself."""
    recipients = [user_id for user_id in dict.fromkeys(recipient_ids) if user_id and user_id != sharer_id]
    if not org_id or not recipients or not push_enabled():
        return
    payload = payloads.artifact_shared(sharer=sharer_name, artifact_title=artifact_title, url=url)
    for user_id in recipients:
        spawn(notify_user(org_id, user_id, payload))
