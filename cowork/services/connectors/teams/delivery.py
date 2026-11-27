"""A scheduled run's result, posted to its owner's personal chat with the Teams bot.

A schedule with ``deliver_to_teams`` posts each completed run's answer there once, after the run
(cowork.scheduler), with a link to the run's conversation when the web UI's address is set; a run
that failed posts a short notice instead. Where to reach the owner comes from ``teams_contacts``,
which the self-hosted recording proxy keeps as it pairs people. The run's ``teams_delivery`` says
how it went: ``sending`` while it posts, then ``delivered`` or ``failed``, or ``unknown`` when it may
have reached Teams (a timeout or a 5xx after posting, or a later part failing after an earlier one
went out); a ``sending`` the server stopped in becomes ``unknown`` at the next start. Nothing is ever
sent again on its own, so a person never gets one result twice.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID

import httpx

from cowork.common.logger import log_context

logger = logging.getLogger(__name__)

SENDING, DELIVERED, FAILED, UNKNOWN = "sending", "delivered", "failed", "unknown"
# Teams takes a bot message up to about 100 KB counted in UTF-16, and Microsoft advises 80 KB.
MAX_PART_UNITS = 38_000
# The run's answer is its last assistant message with text; tool-only steps can follow it.
_ANSWER_LOOKBACK = 20
# Recording how a delivery went is tried again for a while, so a run isn't left ``sending``.
_RECORD_ATTEMPTS = 5
_RECORD_PAUSE_SECONDS = 2.0


def _system_session():
    from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
    from cowork.db.session import get_open_session

    return ScopedSession(get_open_session(), SYSTEM_SCOPE)


def _set_delivery(run_id: UUID, state: str) -> None:
    from cowork.models.schedule import ScheduleRun

    session = _system_session()
    try:
        run = session.get(ScheduleRun, run_id)
        if run is not None:
            run.teams_delivery = state
            session.add(run)
            session.commit()
    finally:
        session.close()


async def _record(run_id: UUID, state: str, *, conversation_id: UUID | None) -> bool:
    """Records STATE on the run, trying again while the database refuses; whether it landed."""
    for attempt in range(_RECORD_ATTEMPTS):
        try:
            await asyncio.to_thread(_set_delivery, run_id, state)
            return True
        except Exception as error:
            if attempt == _RECORD_ATTEMPTS - 1:
                logger.warning("Couldn't record how a scheduled run's Teams post went",
                               extra={**log_context(conversation_id=conversation_id), "event": "teams_delivery_unrecorded",
                                      "reason": state, "error_type": type(error).__name__})
                return False
            await asyncio.sleep(_RECORD_PAUSE_SECONDS)
    return False


def save_contact(*, user_id: str, tenant_id: str, aad_object_id: str, service_url: str) -> None:
    """Keeps where to reach USER_ID in Teams, replacing what was kept."""
    from cowork.models.teams_contact import TeamsContact

    session = _system_session()
    try:
        # A Microsoft ID reaches one person: re-pairing it to someone else ends the earlier owner's
        # results there.
        for other in session.exec(session.select(TeamsContact).where(
            TeamsContact.tenant_id == tenant_id, TeamsContact.aad_object_id == aad_object_id,
            TeamsContact.user_id != user_id,
        )).all():
            session.delete(other)
        contact = session.get(TeamsContact, user_id) or TeamsContact(user_id=user_id)
        contact.tenant_id, contact.aad_object_id, contact.service_url = tenant_id, aad_object_id, service_url
        contact.updated_at = now()
        session.add(contact)
        session.commit()
    finally:
        session.close()


def forget_contact(user_id: str) -> None:
    """Forgets where to reach USER_ID in Teams, so no result goes there. The recording proxy calls it as
    they disconnect, and its operator script as they're unpaired."""
    from cowork.models.teams_contact import TeamsContact

    session = _system_session()
    try:
        contact = session.get(TeamsContact, user_id)
        if contact is not None:
            session.delete(contact)
            session.commit()
    finally:
        session.close()


def _contact_and_answer(user_id: str, conversation_id: UUID):
    """The owner's Teams contact, and the run's answer: its conversation's last assistant text."""
    from cowork.db.scoped import unsafe_unscoped_session
    from cowork.models.message import Message
    from cowork.models.teams_contact import TeamsContact

    session = _system_session()
    try:
        contact = session.get(TeamsContact, user_id)
        messages = session.exec(
            session.select(Message)
            .where(Message.conversation_id == conversation_id, Message.role == "assistant")
            .order_by(Message.seq.desc())
            .limit(_ANSWER_LOOKBACK)
        ).all()
        answer = next((text for text in map(_assistant_text, messages) if text), None)
        if contact is not None:
            unsafe_unscoped_session(session).expunge(contact)
        return contact, answer
    finally:
        session.close()


def _assistant_text(message) -> str | None:
    if str(getattr(message.role, "value", message.role)) != "assistant":
        return None
    content = message.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "output_text") and isinstance(part.get("text"), str)
        )
    else:
        text = ""
    return text.strip() or None


def result_text(*, title: str, answer: str | None, link: str | None) -> str:
    text = f"**{title}**\n\n{answer or 'The run finished without an answer.'}"
    return f"{text}\n\n[Open in MindsHub]({link})" if link else text


def failure_text(*, title: str, link: str | None) -> str:
    text = f"**{title}**\n\nThis scheduled run didn't complete."
    return f"{text}\n\n[Open in MindsHub]({link}) to see what happened." if link else text


def _status_code(error: BaseException | None) -> int | None:
    return error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None


def _may_have_posted(error: BaseException) -> bool:
    """Whether Teams may have posted the message the error came back for."""
    from cowork.services.connectors.teams.client import TeamsPartlySent

    if isinstance(error, TeamsPartlySent):
        cause = error.__cause__
        if error.posted:
            return True
        code = _status_code(cause)
        # The request was sent; Teams may have posted before the answer timed out or failed.
        return (code is not None and (code >= 500 or code == 408)) or isinstance(cause, (
            httpx.ReadTimeout, httpx.WriteTimeout, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError,
        ))
    return False


def _units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def split_text(text: str, limit: int = MAX_PART_UNITS) -> list[str]:
    """TEXT as parts of at most LIMIT UTF-16 units: at paragraphs, then lines, then characters."""
    if _units(text) <= limit:
        return [text]
    parts: list[str] = []
    current = ""
    for piece in (p for paragraph in text.split("\n\n") for p in _pieces(paragraph, limit)):
        joined = f"{current}\n\n{piece}" if current else piece
        if current and _units(joined) > limit:
            parts.append(current)
            current = piece
        else:
            current = joined
    if current:
        parts.append(current)
    return parts


def _cut_at(line: str, limit: int) -> int:
    """How many characters of LINE fit in LIMIT UTF-16 units, counted in one pass."""
    units = 0
    for index, char in enumerate(line):
        units += 2 if ord(char) > 0xFFFF else 1
        if units > limit:
            return index
    return len(line)


def _pieces(paragraph: str, limit: int) -> list[str]:
    if _units(paragraph) <= limit:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for line in paragraph.split("\n"):
        while _units(line) > limit:
            cut = _cut_at(line, limit)
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:cut])
            line = line[cut:]
        joined = f"{current}\n{line}" if current else line
        if current and _units(joined) > limit:
            pieces.append(current)
            current = line
        else:
            current = joined
    if current:
        pieces.append(current)
    return pieces


async def deliver_run_result(
    *, run_id: UUID, user_id: str | None, conversation_id: UUID | None, title: str, succeeded: bool = True,
) -> None:
    """Posts a finished run's result to its owner's Teams chat once, or for a failed run a short
    notice, and records how it went."""
    from cowork.services.connectors.teams.bot import running_bot

    context = log_context(conversation_id=conversation_id)

    def failed(**fields) -> None:
        logger.warning("Couldn't post a scheduled run's result to Teams",
                       extra={**context, "event": "teams_delivery_failed",
                              **{name: value for name, value in fields.items() if value is not None}})

    bot = running_bot()
    if not await _record(run_id, SENDING, conversation_id=conversation_id):
        return
    if bot is None or not user_id or (succeeded and conversation_id is None):
        await _record(run_id, FAILED, conversation_id=conversation_id)
        failed(reason="no bot, owner or conversation")
        return
    try:
        contact, answer = await asyncio.to_thread(_contact_and_answer, user_id, conversation_id)
        if contact is None:
            await _record(run_id, FAILED, conversation_id=conversation_id)
            failed(reason="the owner hasn't connected Teams")
            return
        bot.check_service_url(contact.service_url)
        link = f"{bot.web_ui_url}/c/{conversation_id}" if bot.web_ui_url and conversation_id else None
        text = result_text(title=title, answer=answer, link=link) if succeeded else failure_text(title=title, link=link)
        client = await bot.client()
        await client.message_person(contact.service_url, contact.tenant_id, contact.aad_object_id,
                                    split_text(text, MAX_PART_UNITS))
    except Exception as error:
        maybe = _may_have_posted(error)
        await _record(run_id, UNKNOWN if maybe else FAILED, conversation_id=conversation_id)
        cause = error.__cause__ if error.__cause__ is not None else error
        # Only the type and status: an error from the Teams service can echo the request.
        failed(error_type=type(cause).__name__, status_code=_status_code(cause), reason="may have reached Teams" if maybe else None)
        return
    await _record(run_id, DELIVERED, conversation_id=conversation_id)


def forget_interrupted_deliveries() -> int:
    """Marks every delivery a stopped server left ``sending`` as ``unknown``: it may have reached
    Teams, so it is never sent again. Returns how many."""
    import sqlalchemy as sa

    from cowork.db.scoped import unsafe_unscoped_session
    from cowork.models.schedule import ScheduleRun

    session = _system_session()
    try:
        result = unsafe_unscoped_session(session).exec(sa.update(ScheduleRun).where(ScheduleRun.teams_delivery == SENDING).values(teams_delivery=UNKNOWN))
        session.commit()
        return result.rowcount or 0
    finally:
        session.close()


def now() -> datetime:
    return datetime.now(timezone.utc)
