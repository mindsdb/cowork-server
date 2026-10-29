"""A task can pin a model. When the pin can't be served, the run uses the
account default instead of failing, and says which model it couldn't use.

Each test names the guarantee it pins.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from sqlmodel import select

import cowork.handlers.responses as responses_mod
import cowork.scheduler as scheduler_mod
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.handlers.turn_errors import response_failed_sse
from cowork.models.schedule import ScheduleRun
from cowork.schemas.schedules import RunStatus, ScheduleRunResponse
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleService

DUE = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
PINNED = "claude-retired-1"


def _failed(code: str, *, acted: bool = False) -> scheduler_mod._TurnEvidence:
    return scheduler_mod._TurnEvidence(
        reason="error",
        acted=acted,
        failure=scheduler_mod._TurnFailure(message=f"{code} text", code=code, assistant_message_id="0" * 32),
    )


_COMPLETED = scheduler_mod._TurnEvidence(reason="completed", acted=True)


@pytest.fixture
def turns(monkeypatch):
    """Fakes the turn: records each request's model, and ends each turn the
    way the next queued evidence says. Records which failed turns were
    deleted."""
    state = {"models": [], "evidence": [], "dropped": []}

    class FakeHandler:
        def __init__(self, *, principal=None, interactive=True, schedule_id=None):
            pass

        async def handle(self, request):
            state["models"].append(request.model)

            async def _gen():
                if False:
                    yield

            return _gen()

    async def _evidence(_conversation_id):
        return state["evidence"].pop(0)

    def _drop(session, *, principal, conversation_id, message_id):
        state["dropped"].append(message_id.hex)

    monkeypatch.setattr(responses_mod, "ResponsesHandler", FakeHandler)
    monkeypatch.setattr(scheduler_mod, "_turn_evidence", _evidence)
    monkeypatch.setattr(scheduler_mod, "_drop_failed_turn", _drop)
    return state


@pytest.fixture
def run_schedule():
    """Creates a schedule on the given model, runs it once, and returns its runs."""
    created = []

    def _run(model: str) -> list[ScheduleRun]:
        session = get_open_session()
        schedule_id = ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).create_schedule(
            title="model test",
            prompt="do the thing",
            cadence="daily",
            next_run_at=DUE,
            model=model,
            project_id=GENERAL_PROJECT_ID,
        ).id
        session.close()
        created.append(schedule_id)
        asyncio.run(scheduler_mod.execute_schedule(schedule_id, is_manual=False))
        session = get_open_session()
        runs = list(session.exec(select(ScheduleRun).where(ScheduleRun.schedule_id == schedule_id)).all())
        session.close()
        return runs

    yield _run
    session = get_open_session()
    for schedule_id in created:
        ScheduleService(ScopedSession(session, SYSTEM_SCOPE)).delete_schedule(schedule_id)
    session.close()


@pytest.mark.parametrize("code", ["model_not_found", "model_access_denied", "model_disabled", "model_restricted"])
def test_an_unavailable_pinned_model_runs_on_the_default_in_place_of_the_failed_turn(turns, run_schedule, code):
    turns["evidence"] = [_failed(code), _COMPLETED]

    [run] = run_schedule(PINNED)

    assert turns["models"] == [PINNED, None]
    assert turns["dropped"] == ["0" * 32]
    assert (run.status, run.error, run.unavailable_model) == (RunStatus.success, None, PINNED)


def test_a_failed_fallback_still_records_the_model_it_could_not_use(turns, run_schedule):
    turns["evidence"] = [_failed("model_not_found"), _failed("provider_auth")]

    [run] = run_schedule(PINNED)

    assert turns["models"] == [PINNED, None]
    assert (run.status, run.error, run.unavailable_model) == (RunStatus.failed, "provider_auth text", PINNED)


def test_a_task_on_the_default_model_is_not_run_again(turns, run_schedule):
    turns["evidence"] = [_failed("model_not_found")]

    [run] = run_schedule("default")

    assert turns["models"] == [None]
    assert (run.status, run.unavailable_model) == (RunStatus.failed, None)


@pytest.mark.parametrize("evidence", [
    pytest.param(_failed("model_not_found", acted=True), id="after-it-acted"),
    pytest.param(_failed("provider_auth"), id="other-failure"),
])
def test_only_a_model_failure_before_the_turn_acted_falls_back(turns, run_schedule, evidence):
    turns["evidence"] = [evidence]

    [run] = run_schedule(PINNED)

    assert turns["models"] == [PINNED]
    assert run.unavailable_model is None


def test_a_served_pinned_model_runs_once_on_that_model(turns, run_schedule):
    turns["evidence"] = [_COMPLETED]

    [run] = run_schedule(PINNED)

    assert turns["models"] == [PINNED]
    assert (run.status, run.unavailable_model) == (RunStatus.success, None)


def test_the_failure_frame_names_the_failed_turns_saved_answer():
    frame = response_failed_sse(
        "That model isn't available.", "model_not_found", request_id="r1", assistant_message_id="m1",
    )

    failure = scheduler_mod._failure_from_record({"sse": frame})

    assert (failure.code, failure.assistant_message_id) == ("model_not_found", "m1")


def test_the_run_response_carries_the_unavailable_model():
    run = ScheduleRun(
        schedule_id=GENERAL_PROJECT_ID, started_at=DUE, status=RunStatus.success, unavailable_model=PINNED,
    )

    assert ScheduleRunResponse.serialize(run)["unavailableModel"] == PINNED


def test_dropping_the_failed_turn_leaves_the_conversation_empty_for_the_rerun():
    from cowork.handlers.turn_errors import response_failed_payload
    from cowork.services.conversations import ConversationService

    session = get_open_session()
    conversations = ConversationService(ScopedSession(session, SYSTEM_SCOPE))
    conversation_id = conversations.create_conversation(topic="model test").id
    conversations.save_user_message(conversation_id, "do the thing")
    answer = conversations.save_assistant_turn(
        conversation_id, "", [response_failed_payload("That model isn't available.", "model_not_found")],
    )
    session.commit()

    scheduler_mod._drop_failed_turn(
        ScopedSession(session, SYSTEM_SCOPE), principal=None, conversation_id=conversation_id,
        message_id=answer.id,
    )

    assert conversations.get_messages(conversation_id) == []
    conversations.delete_conversation(conversation_id)
    session.close()
