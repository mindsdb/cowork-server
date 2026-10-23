"""Agent tools for the user's scheduled tasks: list them, and create, change,
pause, resume or delete one from chat.

A task runs unattended, so no change is made until the user approves it on a
confirmation card that shows what will run, how often and when it starts.
The card is asked here, in the handler, not left to the model: whatever the
model says in chat, the write only happens after a picked Confirm.

Writes go through ScheduleService, so the tools get the same ownership and
project checks as the Scheduled Tasks page. The page's own form rules (a
prompt, a future start time) live in its client, so they are restated here.
Built only for a turn someone can answer a card in: never for a scheduled
run, a channel bot, or a turn without questions (see AntonHarness).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from functools import partial
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import ValidationError

from cowork.common.datetime_utils import ensure_utc
from cowork.common.logger import get_logger
from cowork.common.settings.user_settings import current_settings_scope
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.units import run_db
from cowork.schemas.schedules import (
    DEFAULT_MODEL_SENTINEL,
    Cadence,
    ScheduleCreateRequest,
    ScheduleUpdateRequest,
)
from cowork.services.schedules import ScheduleService

logger = get_logger(__name__)

LIST_TOOL_NAME = "list_scheduled_tasks"
MANAGE_TOOL_NAME = "manage_scheduled_task"

# The page titles an untitled task with the start of its prompt.
_TITLE_FROM_PROMPT_CHARS = 80

_ACTIONS = ("create", "update", "pause", "resume", "delete")

_CADENCE_DESCRIPTION = (
    "'once' runs a single time at next_run_at. 'hourly', 'daily', 'weekdays' (Mon-Fri) and "
    "'weekly' repeat from next_run_at: at that minute past each hour, at that time of day, "
    "or on that weekday at that time."
)

_MANAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(_ACTIONS)},
        "schedule_id": {
            "type": "string",
            "description": "The task to change, from list_scheduled_tasks. Required for every action but create.",
        },
        "title": {"type": "string", "description": "Short name shown on the Scheduled Tasks page."},
        "prompt": {
            "type": "string",
            "description": (
                "The full instructions each run starts from. Each run is a new conversation "
                "with none of this chat's context, so this must stand on its own."
            ),
        },
        "cadence": {"type": "string", "enum": [c.value for c in Cadence], "description": _CADENCE_DESCRIPTION},
        "next_run_at": {
            "type": "string",
            "description": "First (or next) run, ISO 8601 with a UTC offset, e.g. 2026-10-27T09:00:00+00:00. Must be in the future.",
        },
        "timezone": {
            "type": "string",
            "description": "IANA timezone the cadence follows, e.g. Europe/London.",
        },
    },
    "required": ["action"],
}

_LIST_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}

_PROMPT = """\
SCHEDULED TASKS:
- `{list_tool}` shows the user's scheduled tasks. `{manage_tool}` creates, changes, pauses, resumes or \
deletes one. Use the schedule_id `{list_tool}` returns; never guess one.
- The user approves every change on a confirmation card the tool shows them, and nothing changes \
unless they confirm it there. So call the tool rather than asking "shall I?" in chat. Ask first only \
when you don't know what to run, how often, or when. If they cancel, nothing was changed: ask what \
they'd like instead, and don't make the change another way.
- Every run starts a brand-new conversation with none of this chat's history, files or decisions. \
Write the prompt as complete instructions for someone who has never seen this chat: name the exact \
connections, tables, files, artifact names or paths, filters, recipients, and what to produce and \
where to put it. Express dates relative to the run ("the previous 7 days"), not as today's dates. \
Never refer back to the chat ("as discussed", "the report above", "same as before").
- next_run_at is the first run, in the future, ISO 8601 with a UTC offset; recurring tasks repeat \
at that time. timezone is an IANA name. This device's timezone is {device_tz}: use it unless the user \
names another.
- Tasks created here belong to this chat's project and run on the account's default model."""


def _device_timezone() -> str:
    """The IANA name of this machine's zone, which is the user's on the desktop."""
    from cowork.schedule_timing import resolve_timezone

    return resolve_timezone("local").key


def _units_scope():
    return current_settings_scope() or LOCAL_SCOPE


# ---------------------------------------------------------------------------
# Database units. Each returns plain data, so nothing crosses back to the
# event loop attached to a session.
# ---------------------------------------------------------------------------


def _summary(schedule) -> dict:
    return {
        "schedule_id": str(schedule.id),
        "title": schedule.title,
        "prompt": schedule.prompt,
        "cadence": str(getattr(schedule.cadence, "value", schedule.cadence)),
        "timezone": schedule.timezone,
        "next_run_at": ensure_utc(schedule.next_run_at).isoformat(),
        "paused": not schedule.enabled,
    }


def _list_unit(session: ScopedSession) -> list[dict]:
    return [_summary(s) for s in ScheduleService(session).list_schedules()]


def _get_unit(session: ScopedSession, *, schedule_id: UUID) -> dict:
    return _summary(ScheduleService(session).get_schedule(schedule_id))


def _create_unit(session: ScopedSession, *, body: ScheduleCreateRequest, project_id: UUID | None) -> dict:
    return _summary(ScheduleService(session).create_schedule(
        title=body.title,
        prompt=body.prompt,
        cadence=body.cadence,
        next_run_at=body.next_run_at,
        model=DEFAULT_MODEL_SENTINEL,
        timezone=body.timezone,
        project_id=project_id,
        enabled=True,
    ))


def _update_unit(session: ScopedSession, *, schedule_id: UUID, changes: dict) -> dict:
    return _summary(ScheduleService(session).update_schedule(schedule_id, **changes))


def _pause_unit(session: ScopedSession, *, schedule_id: UUID) -> dict:
    return _summary(ScheduleService(session).pause_schedule(schedule_id))


def _resume_unit(session: ScopedSession, *, schedule_id: UUID) -> dict:
    return _summary(ScheduleService(session).resume_schedule(schedule_id))


def _delete_unit(session: ScopedSession, *, schedule_id: UUID) -> bool:
    return ScheduleService(session).delete_schedule(schedule_id)


# ---------------------------------------------------------------------------
# Validation: the request schemas the HTTP routes parse with, plus the
# page's form rules.
# ---------------------------------------------------------------------------


class _Invalid(Exception):
    """Input the tool refuses, with a message the model can act on."""


def _schema_error(exc: ValidationError) -> _Invalid:
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first.get("loc", ())) or "input"
    return _Invalid(f"`{field}`: {first.get('msg', 'invalid value')}.")


def _check_timezone(name: str) -> None:
    # resolve_timezone falls back to UTC for an unknown name, which would
    # quietly run the task at the wrong hour.
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise _Invalid(f"`timezone`: {name!r} isn't an IANA timezone name, such as Europe/London.")


def _check_next_run(at: datetime) -> None:
    if at.tzinfo is None:
        raise _Invalid("`next_run_at` needs a UTC offset, e.g. 2026-10-27T09:00:00+01:00.")
    if at <= datetime.now(timezone.utc):
        raise _Invalid("`next_run_at` must be in the future.")


def _check_weekday_start(cadence: str, at: datetime, tz_name: str) -> None:
    # The first run happens at next_run_at whatever its day, so a weekday
    # task started on a Saturday would run on a weekend once.
    if cadence == Cadence.weekdays.value and _local(at, tz_name).weekday() >= 5:
        raise _Invalid("`next_run_at` falls on a weekend; a weekdays task must start on a Monday to Friday.")


def _check_prompt(prompt: str) -> None:
    if not prompt.strip():
        raise _Invalid("`prompt` is required: the instructions each run starts from.")


def _fields(tc_input: dict, names: tuple[str, ...]) -> dict:
    return {name: tc_input[name] for name in names if tc_input.get(name) not in (None, "")}


def _create_request(tc_input: dict) -> ScheduleCreateRequest:
    fields = _fields(tc_input, ("title", "prompt", "cadence", "next_run_at", "timezone"))
    missing = [name for name in ("prompt", "cadence", "next_run_at", "timezone") if name not in fields]
    if missing:
        raise _Invalid(f"create needs {', '.join(f'`{m}`' for m in missing)}.")
    fields["title"] = str(fields.get("title") or "").strip() or str(fields["prompt"]).strip()[:_TITLE_FROM_PROMPT_CHARS]
    try:
        body = ScheduleCreateRequest.model_validate(fields)
    except ValidationError as exc:
        raise _schema_error(exc)
    _check_prompt(body.prompt)
    _check_timezone(body.timezone)
    _check_next_run(body.next_run_at)
    _check_weekday_start(body.cadence.value, body.next_run_at, body.timezone)
    return body


def _update_changes(tc_input: dict) -> dict:
    fields = _fields(tc_input, ("title", "prompt", "cadence", "next_run_at", "timezone"))
    if not fields:
        raise _Invalid("update needs at least one of `title`, `prompt`, `cadence`, `next_run_at`, `timezone`.")
    try:
        body = ScheduleUpdateRequest.model_validate(fields)
    except ValidationError as exc:
        raise _schema_error(exc)
    if body.prompt is not None:
        _check_prompt(body.prompt)
    if body.title is not None and not body.title.strip():
        raise _Invalid("`title` can't be blank.")
    if body.timezone is not None:
        _check_timezone(body.timezone)
    if body.next_run_at is not None:
        _check_next_run(body.next_run_at)
    return body.model_dump(exclude_none=True)


def _schedule_id(tc_input: dict) -> UUID:
    raw = str(tc_input.get("schedule_id") or "").strip()
    if not raw:
        raise _Invalid(f"`schedule_id` is required. Call `{LIST_TOOL_NAME}` to find it.")
    try:
        return UUID(raw)
    except ValueError:
        raise _Invalid(f"`schedule_id` {raw!r} isn't a task id. Call `{LIST_TOOL_NAME}` to find it.")


# ---------------------------------------------------------------------------
# The confirmation card.
# ---------------------------------------------------------------------------


def _local(at: datetime, tz_name: str) -> datetime:
    return ensure_utc(at).astimezone(ZoneInfo(tz_name)) if tz_name else ensure_utc(at)


def _clock(at: datetime) -> str:
    return at.strftime("%H:%M")


def describe_cadence(cadence: str, at: datetime, tz_name: str) -> str:
    """How often, in words, at the task's local time."""
    local = _local(at, tz_name)
    text = {
        Cadence.once.value: "Once",
        Cadence.hourly.value: f"Every hour at :{local.strftime('%M')}",
        Cadence.daily.value: f"Every day at {_clock(local)}",
        Cadence.weekdays.value: f"Every weekday (Mon–Fri) at {_clock(local)}",
        Cadence.weekly.value: f"Every {local.strftime('%A')} at {_clock(local)}",
    }.get(cadence, cadence)
    return f"{text} ({tz_name})"


def describe_start(at: datetime, tz_name: str) -> str:
    local = _local(at, tz_name)
    # `day` rather than %-d, which Windows' strftime rejects.
    return f"{local.strftime('%a')} {local.day} {local.strftime('%b %Y, %H:%M')}"


def _quoted(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.strip().splitlines())


def _task_lines(task: dict, *, starts_label: str) -> list[str]:
    at = datetime.fromisoformat(task["next_run_at"])
    return [
        f"- **How often:** {describe_cadence(task['cadence'], at, task['timezone'])}",
        f"- **{starts_label}:** {describe_start(at, task['timezone'])}",
        "",
        "**What runs** (in a new conversation each time, without this chat's context):",
        "",
        _quoted(task["prompt"]),
    ]


_CHANGE_LABELS = {
    "title": "title",
    "prompt": "what runs",
    "cadence": "how often",
    "timezone": "timezone",
    "next_run_at": "start time",
}


def card_prompt(action: str, task: dict, *, before: dict | None = None) -> str:
    """The card's markdown: the task as it will be once the user confirms."""
    title = task["title"]
    if action == "create":
        lines = [f"**Schedule “{title}”?**", "", *_task_lines(task, starts_label="Starts")]
    elif action == "update":
        changed = [label for field, label in _CHANGE_LABELS.items() if before and before[field] != task[field]]
        heading = f"**Change “{before['title'] if before else title}”?**"
        lines = [heading, "", f"Changing: {', '.join(changed)}.", ""]
        if before and before["title"] != title:
            lines += [f"- **Title:** {title}"]
        lines += _task_lines(task, starts_label="Next run")
    elif action == "pause":
        lines = [
            f"**Pause “{title}”?**", "",
            "It won't run again until it's resumed. It stays on the Scheduled Tasks page.", "",
            f"- **How often:** {describe_cadence(task['cadence'], datetime.fromisoformat(task['next_run_at']), task['timezone'])}",
        ]
    elif action == "resume":
        lines = [f"**Resume “{title}”?**", "", *_task_lines(task, starts_label="Next run")]
    else:
        lines = [
            f"**Delete “{title}”?**", "",
            "This removes the task and its run history. Conversations from earlier runs are kept.",
        ]
    return "\n".join(lines)


_CONFIRM_LABELS = {
    "create": ("Schedule", "Create this scheduled task"),
    "update": ("Save changes", "Save these changes to the task"),
    "pause": ("Pause", "Stop this task from running"),
    "resume": ("Resume", "Start running this task again"),
    "delete": ("Delete", "Delete this task"),
}


async def confirm(session: Any, action: str, prompt: str) -> str:
    """Show the confirmation card. Returns "confirmed", "cancelled",
    "timeout", "unavailable" or "error"; only "confirmed" lets the change run.

    The same card as the write-approval gate's: two buttons, no typed answer,
    no default, and outside the model's question budget because the tool asks
    it, not the model.
    """
    from anton.core.interaction.elicit import AskOption, AskRequest, elicit

    label, detail = _CONFIRM_LABELS[action]
    request = AskRequest(
        prompt=prompt,
        options=(
            AskOption(value="confirm", label=label, detail=detail),
            AskOption(value="cancel", label="Cancel", detail="Leave scheduled tasks as they are"),
        ),
        allow_custom=False,
        timeout_s=getattr(getattr(session, "elicitor", None), "timeout_s", None),
    )
    try:
        answer = await elicit(session, f"schedule:{uuid.uuid4().hex}", request, counted=False)
    except Exception as exc:
        logger.warning("scheduled task confirmation failed (%s)", type(exc).__name__)
        return "error"
    if answer.status == "answered" and answer.values == ("confirm",):
        return "confirmed"
    if answer.status in ("answered", "cancelled"):
        return "cancelled"
    if answer.status in ("timeout", "unavailable"):
        return answer.status
    return "error"


_NOT_CONFIRMED = {
    "cancelled": (
        "The user didn't confirm it, so nothing was created or changed. Don't make this change "
        "another way. Ask what they'd like different."
    ),
    "timeout": "The user didn't answer the confirmation in time, so nothing was created or changed.",
    "unavailable": (
        "Confirmation isn't available in this conversation, so scheduled tasks can't be changed "
        "from here. Nothing was created or changed. Point the user to the Scheduled Tasks page."
    ),
    "error": "The confirmation card failed, so nothing was created or changed.",
}


# ---------------------------------------------------------------------------
# Handlers.
# ---------------------------------------------------------------------------


def _result(message: str, task: dict | None = None) -> str:
    return json.dumps({"result": message, **({"task": task} if task else {})})


async def _list_tasks(_session: Any, _tc_input: dict) -> str:
    tasks = await run_db(_list_unit, scope=_units_scope())
    if not tasks:
        return json.dumps({"tasks": [], "result": "The user has no scheduled tasks."})
    return json.dumps({"tasks": tasks})


async def _manage_task(session: Any, tc_input: dict, *, project_id: UUID | None) -> str:
    action = str(tc_input.get("action") or "").strip()
    if action not in _ACTIONS:
        return _result(f"`action` must be one of {', '.join(_ACTIONS)}.")
    scope = _units_scope()
    try:
        if action == "create":
            body = _create_request(tc_input)
            preview = {
                "title": body.title, "prompt": body.prompt, "cadence": body.cadence.value,
                "timezone": body.timezone, "next_run_at": ensure_utc(body.next_run_at).isoformat(),
            }
            apply = partial(_create_unit, body=body, project_id=project_id)
            before = None
        else:
            schedule_id = _schedule_id(tc_input)
            before = await run_db(partial(_get_unit, schedule_id=schedule_id), scope=scope)
            preview = dict(before)
            if action == "update":
                changes = _update_changes(tc_input)
                for field, value in changes.items():
                    preview[field] = (
                        ensure_utc(value).isoformat() if isinstance(value, datetime)
                        else getattr(value, "value", value)
                    )
                _check_weekday_start(
                    preview["cadence"], datetime.fromisoformat(preview["next_run_at"]), preview["timezone"],
                )
                if preview == before:
                    return _result("Those values are what the task already has, so there's nothing to change.", before)
                apply = partial(_update_unit, schedule_id=schedule_id, changes=changes)
            elif action == "pause":
                if before["paused"]:
                    return _result("This task is already paused.", before)
                apply = partial(_pause_unit, schedule_id=schedule_id)
            elif action == "resume":
                if not before["paused"]:
                    return _result("This task isn't paused.", before)
                if before["cadence"] == Cadence.once.value and (
                    datetime.fromisoformat(before["next_run_at"]) <= datetime.now(timezone.utc)
                ):
                    # The scheduler disables an overdue one-off again without running it.
                    return _result(
                        "This one-off task's run time has passed. Update `next_run_at` to a future "
                        "time first, then resume it.", before,
                    )
                apply = partial(_resume_unit, schedule_id=schedule_id)
            else:
                apply = partial(_delete_unit, schedule_id=schedule_id)
    except _Invalid as exc:
        return _result(f"Not sent for confirmation: {exc}")
    except ValueError:
        return _result(f"No scheduled task with that id. Call `{LIST_TOOL_NAME}` to find it.")

    decision = await confirm(session, action, card_prompt(action, preview, before=before))
    if decision != "confirmed":
        return _result(_NOT_CONFIRMED[decision])

    try:
        outcome = await run_db(apply, scope=scope)
    except ValueError as exc:
        # The task was deleted, or its project went away, while the card waited.
        return _result(f"Nothing was changed: {exc}.")
    if action == "delete":
        if not outcome:
            return _result("Nothing was deleted: the task was already gone.")
        return _result(f"Deleted “{preview['title']}”.")
    verb = {"create": "Scheduled", "update": "Updated", "pause": "Paused", "resume": "Resumed"}[action]
    return _result(f"{verb} “{outcome['title']}”. It's on the Scheduled Tasks page.", outcome)


def build_schedule_tools(*, project_id: UUID | None) -> list:
    """The two tools, with new tasks landing in `project_id` (the chat's own)."""
    from anton.core.tools.tool_defs import ToolDef

    async def manage(session: Any, tc_input: dict) -> str:
        return await _manage_task(session, tc_input, project_id=project_id)

    prompt = _PROMPT.format(list_tool=LIST_TOOL_NAME, manage_tool=MANAGE_TOOL_NAME, device_tz=_device_timezone())
    return [
        ToolDef(
            name=LIST_TOOL_NAME,
            description="List the user's scheduled tasks: id, title, prompt, cadence, timezone, next run, and whether it's paused.",
            input_schema=_LIST_SCHEMA,
            handler=_list_tasks,
        ),
        ToolDef(
            name=MANAGE_TOOL_NAME,
            description=(
                "Create, change, pause, resume or delete one of the user's scheduled tasks. The user "
                "sees a confirmation card with what will run, how often and when it starts, and the "
                "change happens only if they confirm. Returns what happened."
            ),
            input_schema=_MANAGE_SCHEMA,
            handler=manage,
            prompt=prompt,
        ),
    ]
