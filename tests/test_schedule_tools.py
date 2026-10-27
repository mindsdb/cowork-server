"""The agent's scheduled-task tools: nothing is created or changed until the
user confirms a card showing what will run, how often and when, and a task
made from chat runs from its own prompt alone.

The card tests drive the handler through anton's real `elicit()`, with a
stand-in elicitor that answers the card. The end-to-end tests run a chat turn
and a scheduled run through the real ResponsesHandler and AntonHarness, as in
test_turn_pool_connections, and read the result back through the Scheduled
Tasks page's own API.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from anton.core.interaction.elicit import AskAnswer

import test_turn_pool_connections as turn_pool
from _fakes import PausedModel

import cowork.scheduler as scheduler
from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.units import unit_session
from cowork.harnesses.anton_harness import harness as harness_mod
from cowork.harnesses.anton_harness import schedule_tools
from cowork.harnesses.anton_harness.schedule_tools import LIST_TOOL_NAME, MANAGE_TOOL_NAME, build_schedule_tools
from cowork.models.schedule import Schedule, ScheduleRun
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID

_turn_runs_in_this_process = turn_pool._turn_runs_in_this_process

PROMPT = (
    "Query the `acme_sales` Postgres connection for orders placed in the previous 24 hours, "
    "grouped by region, and update the `daily-sales` artifact with a table and a short summary."
)


def _next_weekday_9am(weekday: int = 0) -> datetime:
    """09:00 UTC on the next given weekday (Monday by default)."""
    today = datetime.now(timezone.utc).date()
    day = today + timedelta(days=(weekday - today.weekday() - 1) % 7 + 1)
    return datetime(day.year, day.month, day.day, 9, 0, tzinfo=timezone.utc)


def _create_input(**overrides) -> dict:
    return {
        "action": "create",
        "title": "Daily sales",
        "prompt": PROMPT,
        "cadence": "weekdays",
        "next_run_at": _next_weekday_9am().isoformat(),
        "timezone": "UTC",
        **overrides,
    }


class _Card:
    """Answers every card the same way and keeps what it was shown."""

    supported_kinds = ("choice",)
    answer_hint = ""
    timeout_s = 300

    def __init__(self, answer: AskAnswer) -> None:
        self.answer = answer
        self.shown = []

    async def begin(self, _question_id, _request) -> None:
        pass

    async def ask(self, _question_id, request) -> AskAnswer:
        self.shown.append(request)
        return self.answer

    async def end(self, _question_id) -> None:
        pass


class _Session:
    """What anton's elicit() reads off a ChatSession."""

    def __init__(self, card: _Card | None) -> None:
        self.elicitor = card
        self.emitter = object()
        self.question_count = 0
        self.answer_wait_s = 0.0

    async def emit(self, _event) -> None:
        pass


CONFIRM = AskAnswer(status="answered", values=("confirm",))
CANCEL = AskAnswer(status="answered", values=("cancel",))


def _delete_every_schedule() -> None:
    with unit_session(scope=LOCAL_SCOPE) as session:
        for run in session.exec(session.select(ScheduleRun)):
            session.delete(run)
        for schedule in session.exec(session.select(Schedule)):
            session.delete(schedule)


@pytest.fixture(autouse=True)
def _only_this_tests_schedules():
    """The page lists every schedule in the shared test database, so start
    and end each test with none."""
    _delete_every_schedule()
    yield
    _delete_every_schedule()


async def _call(tool_name: str, session, tc_input: dict) -> dict:
    tools = {tool.name: tool for tool in build_schedule_tools(project_id=GENERAL_PROJECT_ID)}
    return json.loads(await tools[tool_name].handler(session, tc_input))


async def _page_schedules() -> list[dict]:
    """The tasks as the Scheduled Tasks page reads them."""
    async with turn_pool._client() as client:
        response = await client.get("/api/v1/schedules/")
    assert response.status_code == 200, response.text
    return response.json()["schedules"]


async def _create_confirmed() -> dict:
    result = await _call(MANAGE_TOOL_NAME, _Session(_Card(CONFIRM)), _create_input())
    return result["task"]


# --- create -----------------------------------------------------------------


async def test_a_confirmed_task_appears_on_the_scheduled_tasks_page():
    card = _Card(CONFIRM)
    result = await _call(MANAGE_TOOL_NAME, _Session(card), _create_input())

    (shown,) = card.shown
    assert "Schedule “Daily sales”?" in shown.prompt
    assert "Every weekday (Mon–Fri) at 09:00 (UTC)" in shown.prompt
    assert "Starts:" in shown.prompt
    assert PROMPT in shown.prompt
    assert [o.value for o in shown.options] == ["confirm", "cancel"]
    assert shown.allow_custom is False

    (page_task,) = await _page_schedules()
    assert page_task["id"] == result["task"]["schedule_id"]
    assert page_task["prompt"] == PROMPT
    assert page_task["cadence"] == "weekdays"
    assert page_task["timezone"] == "UTC"
    assert page_task["enabled"] is True
    assert page_task["model"] == "default"
    assert page_task["projectId"] == str(GENERAL_PROJECT_ID)


@pytest.mark.parametrize(
    "answer",
    [CANCEL, AskAnswer(status="cancelled"), AskAnswer(status="timeout")],
    ids=["cancel", "skipped", "no-answer"],
)
async def test_nothing_is_created_unless_the_user_confirms(answer):
    result = await _call(MANAGE_TOOL_NAME, _Session(_Card(answer)), _create_input())

    assert "nothing was created or changed" in result["result"]
    assert await _page_schedules() == []


async def test_nothing_is_created_where_no_card_can_be_shown():
    result = await _call(MANAGE_TOOL_NAME, _Session(None), _create_input())

    assert "Confirmation isn't available" in result["result"]
    assert await _page_schedules() == []


@pytest.mark.parametrize(
    "overrides, says",
    [
        ({"next_run_at": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()}, "in the future"),
        ({"next_run_at": "2030-01-01T09:00:00"}, "UTC offset"),
        ({"timezone": "Mars/Olympus"}, "IANA"),
        ({"prompt": "   "}, "`prompt` is required"),
        ({"cadence": "fortnightly"}, "`cadence`"),
        ({"prompt": None}, "`prompt`"),
        ({"next_run_at": _next_weekday_9am(5).isoformat()}, "falls on a weekend"),
    ],
    ids=["past", "naive", "timezone", "blank-prompt", "cadence", "missing-prompt", "weekend-start"],
)
async def test_invalid_input_is_refused_before_any_card(overrides, says):
    card = _Card(CONFIRM)
    result = await _call(MANAGE_TOOL_NAME, _Session(card), _create_input(**overrides))

    assert says in result["result"]
    assert card.shown == []
    assert await _page_schedules() == []


async def test_the_card_shows_times_in_the_tasks_timezone():
    card = _Card(CONFIRM)
    at = datetime(2030, 1, 7, 9, 0, tzinfo=timezone.utc)  # a Monday
    await _call(MANAGE_TOOL_NAME, _Session(card), _create_input(
        cadence="weekly", next_run_at=at.isoformat(), timezone="America/New_York",
    ))

    (shown,) = card.shown
    assert "Every Monday at 04:00 (America/New_York)" in shown.prompt
    assert "Mon 7 Jan 2030, 04:00" in shown.prompt


async def test_an_untitled_task_is_titled_from_its_prompt_as_the_page_does():
    result = await _call(MANAGE_TOOL_NAME, _Session(_Card(CONFIRM)), _create_input(title=None))

    assert result["task"]["title"] == PROMPT[:80]


# --- change, pause, resume, delete ------------------------------------------


async def test_a_confirmed_update_changes_the_task_and_the_card_shows_the_result():
    task = await _create_confirmed()
    card = _Card(CONFIRM)
    result = await _call(MANAGE_TOOL_NAME, _Session(card), {
        "action": "update", "schedule_id": task["schedule_id"], "cadence": "daily",
    })

    (shown,) = card.shown
    assert "Change “Daily sales”?" in shown.prompt
    assert "Changing: how often." in shown.prompt
    assert "Every day at 09:00 (UTC)" in shown.prompt
    assert result["task"]["cadence"] == "daily"
    (page_task,) = await _page_schedules()
    assert page_task["cadence"] == "daily"
    assert page_task["prompt"] == PROMPT


@pytest.mark.parametrize(
    "action, tc_input",
    [
        ("update", {"prompt": "Something else entirely."}),
        ("pause", {}),
        ("delete", {}),
    ],
)
async def test_a_change_the_user_cancels_leaves_the_task_as_it_was(action, tc_input):
    task = await _create_confirmed()
    before = await _page_schedules()

    result = await _call(MANAGE_TOOL_NAME, _Session(_Card(CANCEL)), {
        "action": action, "schedule_id": task["schedule_id"], **tc_input,
    })

    assert "nothing was created or changed" in result["result"]
    assert await _page_schedules() == before


async def test_an_auto_paused_task_resumes_and_a_finished_one_off_does_not_pause():
    task = await _create_confirmed()
    with unit_session(scope=LOCAL_SCOPE) as session:
        schedule = session.get(Schedule, UUID(task["schedule_id"]))
        schedule.enabled = False
        schedule.auto_paused_at = datetime.now(timezone.utc)
        session.add(schedule)

    card = _Card(CONFIRM)
    await _call(MANAGE_TOOL_NAME, _Session(card), {"action": "resume", "schedule_id": task["schedule_id"]})
    assert [t["state"] for t in await _page_schedules()] == ["active"]

    with unit_session(scope=LOCAL_SCOPE) as session:
        schedule = session.get(Schedule, UUID(task["schedule_id"]))
        schedule.cadence = "once"
        schedule.enabled = False
        session.add(schedule)

    result = await _call(MANAGE_TOOL_NAME, _Session(card), {"action": "pause", "schedule_id": task["schedule_id"]})
    assert result["result"] == schedule_tools._NOT_SCHEDULED
    assert len(card.shown) == 1


async def test_pause_resume_and_delete_each_take_a_confirmation():
    task = await _create_confirmed()
    schedule_id = task["schedule_id"]

    card = _Card(CONFIRM)
    await _call(MANAGE_TOOL_NAME, _Session(card), {"action": "pause", "schedule_id": schedule_id})
    assert [t["state"] for t in await _page_schedules()] == ["paused"]

    await _call(MANAGE_TOOL_NAME, _Session(card), {"action": "resume", "schedule_id": schedule_id})
    assert [t["state"] for t in await _page_schedules()] == ["active"]

    result = await _call(MANAGE_TOOL_NAME, _Session(card), {"action": "delete", "schedule_id": schedule_id})
    assert result["result"] == "Deleted “Daily sales”."
    assert await _page_schedules() == []

    assert [r.prompt.splitlines()[0] for r in card.shown] == [
        "**Pause “Daily sales”?**", "**Resume “Daily sales”?**", "**Delete “Daily sales”?**",
    ]


async def test_an_unknown_task_is_reported_without_a_card():
    card = _Card(CONFIRM)
    result = await _call(MANAGE_TOOL_NAME, _Session(card), {"action": "pause", "schedule_id": str(uuid4())})

    assert "No scheduled task with that id" in result["result"]
    assert card.shown == []


async def test_list_returns_the_tasks_the_page_shows():
    task = await _create_confirmed()

    listed = await _call(LIST_TOOL_NAME, _Session(None), {})

    assert listed["tasks"] == [task]


# --- prompt guidance ----------------------------------------------------------


def test_the_tool_tells_the_agent_each_run_starts_without_this_chat():
    manage = {tool.name: tool for tool in build_schedule_tools(project_id=GENERAL_PROJECT_ID)}[MANAGE_TOOL_NAME]

    assert "brand-new conversation with none of this chat's history" in manage.prompt
    assert "Never refer back to the chat" in manage.prompt
    assert "stand on its own" in manage.input_schema["properties"]["prompt"]["description"]


# --- end to end ---------------------------------------------------------------


async def test_a_task_scheduled_in_chat_runs_from_its_prompt_alone(monkeypatch):
    """A chat turn has the tools and schedules a task. The scheduled run then
    starts a new conversation with no history, from the stored prompt, and
    has no schedule tools of its own."""
    chat_id = turn_pool._conversation(turns=[("Our sales database is acme_sales.", "Noted.")])
    created: list[dict] = []
    built: list = []

    async def approve(_session, _action, _prompt):
        return "confirmed"

    async def schedule_from_chat(session):
        tools = {tool.name: tool for tool in session.config.tools}
        if MANAGE_TOOL_NAME in tools:
            created.append(json.loads(await tools[MANAGE_TOOL_NAME].handler(session, _create_input())))

    model = PausedModel(before_answer=schedule_from_chat)
    model.release.set()

    def build(config):
        built.append(config)
        return model.build(config)

    monkeypatch.setattr(schedule_tools, "confirm", approve)
    monkeypatch.setattr(harness_mod, "build_chat_session", build)

    async with turn_pool._client() as client:
        answered = await turn_pool._ask(client, chat_id, "Every weekday at 9, refresh the daily sales artifact.")
    assert answered.status_code == 200, answered.text

    chat_config = built[0]
    assert {LIST_TOOL_NAME, MANAGE_TOOL_NAME} <= {tool.name for tool in chat_config.tools}
    assert chat_config.initial_history  # the chat had context
    (result,) = created
    schedule_id = UUID(result["task"]["schedule_id"])
    (page_task,) = await _page_schedules()
    assert page_task["id"] == str(schedule_id)

    await scheduler.execute_schedule(schedule_id, is_manual=True)

    run_config = built[1]
    assert {LIST_TOOL_NAME, MANAGE_TOOL_NAME}.isdisjoint(tool.name for tool in run_config.tools)
    assert not run_config.initial_history
    finished = turn_pool._finished_run(schedule_id)
    assert finished.conversation_id is not None and finished.conversation_id != chat_id
    with unit_session(scope=LOCAL_SCOPE) as session:
        messages = ConversationService(session).get_ordered_messages(finished.conversation_id)
        first_question = next(m.content for m in messages if str(getattr(m.role, "value", m.role)) == "user")
    assert first_question == PROMPT
