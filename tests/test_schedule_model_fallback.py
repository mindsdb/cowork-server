"""A task pinned to a model the gateway can't serve runs on the account
default instead, within the same run.

The pinned turn must have failed before its agent acted, with a code that says
the model can't be served. The run deletes the failed turn, runs the prompt
again with no model override, and records the pin it couldn't use.
"""

from __future__ import annotations

import sys

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from cowork import scheduler
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.db.session import get_open_session
from cowork.models.schedule import Schedule, ScheduleRun
from cowork.schemas.schedules import AttemptOutcome, RunStatus
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.schedules import ScheduleRunService, ScheduleService

SLOT = (datetime.now(timezone.utc) - timedelta(minutes=1)).replace(microsecond=0)
PINNED = "claude-retired-1"


@pytest.fixture(autouse=True)
def _fresh_scheduler(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "SLOT",
                        (datetime.now(timezone.utc) - timedelta(minutes=1)).replace(microsecond=0))
    monkeypatch.setattr(scheduler, "_scheduler_task", None)
    monkeypatch.setattr(scheduler, "_scheduled_runs", set())
    monkeypatch.setattr(scheduler, "_admission", scheduler._Admission(scheduler._runs_per_process))


def _scoped():
    session = get_open_session()
    return session, ScopedSession(session, SYSTEM_SCOPE)


@pytest.fixture
def schedule_id():
    session, scoped = _scoped()
    try:
        created = ScheduleService(scoped).create_schedule(
            title=f"fallback {uuid4()}", prompt="refresh the numbers", cadence="daily", next_run_at=SLOT,
            model=PINNED, timezone="UTC", project_id=GENERAL_PROJECT_ID, enabled=True,
        ).id
    finally:
        session.close()
    yield created
    from sqlmodel import select

    from cowork.models.conversation import Conversation
    from cowork.services.conversations import ConversationService

    session, scoped = _scoped()
    try:
        title = session.get(Schedule, created).title
        for conversation_id in session.exec(select(Conversation.id).where(Conversation.topic == title)).all():
            ConversationService(scoped).delete_conversation(conversation_id)
        ScheduleService(scoped).delete_schedule(created)
    finally:
        session.close()


def _failed(code: str, *, acted: bool = False, message_id: str | None = None):
    return SimpleNamespace(
        reason="error", acted=acted,
        failure=scheduler._TurnFailure(message="That model isn't available.", code=code, assistant_message_id=message_id),
    )


COMPLETED = SimpleNamespace(reason="completed", acted=True, failure=None)


@pytest.fixture
def turns(monkeypatch):
    """A turn handler that records each request, and the endings its turns
    show, in order. Also records which failed turns the run deleted."""
    import cowork.handlers.responses as responses_mod

    requests: list = []
    endings: list = []
    dropped: list = []

    class RecordingHandler:
        def __init__(self, *, principal=None, interactive=True, schedule_id=None):
            pass

        async def handle(self, request):
            requests.append(request)

            async def stream():
                if False:
                    yield

            return stream()

    async def evidence(_conversation_id):
        ending = endings.pop(0)
        return scheduler._TurnEvidence(reason=ending.reason, acted=ending.acted, failure=ending.failure)

    def drop(_session, *, principal, conversation_id, message_id):
        dropped.append(message_id)

    monkeypatch.setattr(responses_mod, "ResponsesHandler", RecordingHandler)
    monkeypatch.setattr(scheduler, "_turn_evidence", evidence)
    monkeypatch.setattr(scheduler, "_drop_failed_turn", drop)
    return SimpleNamespace(requests=requests, endings=endings, dropped=dropped)


def _runs(schedule_id: UUID) -> list[ScheduleRun]:
    session, scoped = _scoped()
    try:
        return list(reversed(ScheduleRunService(scoped).list_runs(schedule_id)))
    finally:
        session.close()


async def _run_due_slot(schedule_id: UUID) -> None:
    session, scoped = _scoped()
    try:
        (slot,) = [s for s in scheduler._poll(scoped) if s.schedule_id == schedule_id]
    finally:
        session.close()
    await scheduler.execute_schedule(slot.schedule_id, slot_at=slot.slot_at)


async def test_an_unservable_pin_runs_again_on_the_account_default(schedule_id, turns):
    answer = str(uuid4())
    turns.endings.extend([_failed("model_not_found", message_id=answer), COMPLETED])

    await _run_due_slot(schedule_id)

    assert [r.model for r in turns.requests] == [PINNED, None]
    assert turns.dropped == [UUID(answer)], "the failed turn is deleted so the prompt shows once"
    (run,) = _runs(schedule_id)
    assert run.status == RunStatus.success
    assert run.outcome == AttemptOutcome.completed
    assert run.unavailable_model == PINNED


async def test_the_fallback_goes_ahead_when_the_failed_turn_has_no_answer_to_delete(schedule_id, turns):
    turns.endings.extend([_failed("model_disabled"), COMPLETED])

    await _run_due_slot(schedule_id)

    assert [r.model for r in turns.requests] == [PINNED, None]
    assert turns.dropped == []
    (run,) = _runs(schedule_id)
    assert run.status == RunStatus.success
    assert run.unavailable_model == PINNED


async def test_a_failed_fallback_settles_the_run_failed_and_still_names_the_pin(schedule_id, turns):
    turns.endings.extend([_failed("model_not_found"), _failed("provider_overloaded", acted=True)])

    await _run_due_slot(schedule_id)

    assert len(turns.requests) == 2
    (run,) = _runs(schedule_id)
    assert run.status == RunStatus.failed
    assert run.outcome == AttemptOutcome.failed_during_execution
    assert run.unavailable_model == PINNED


@pytest.mark.parametrize(
    "first",
    [
        pytest.param(_failed("rate_limited"), id="another-code"),
        pytest.param(_failed("model_not_found", acted=True), id="after-acting"),
    ],
)
async def test_no_fallback_unless_the_pin_failed_before_acting_for_its_model(schedule_id, turns, first):
    turns.endings.append(first)

    await _run_due_slot(schedule_id)

    assert [r.model for r in turns.requests] == [PINNED]
    (run,) = _runs(schedule_id)
    assert run.status == RunStatus.failed
    assert run.unavailable_model is None


async def test_a_task_on_the_account_default_has_nothing_to_fall_back_from(schedule_id, turns):
    session, scoped = _scoped()
    try:
        ScheduleService(scoped).update_schedule(schedule_id, model="default")
    finally:
        session.close()
    turns.endings.append(_failed("model_not_found"))

    await _run_due_slot(schedule_id)

    assert [r.model for r in turns.requests] == [None]
    (run,) = _runs(schedule_id)
    assert run.unavailable_model is None


def test_the_unavailable_model_migration_upgrades_and_downgrades(tmp_path, monkeypatch):
    from alembic import command
    from sqlalchemy import create_engine, inspect

    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.migrations import _alembic_config, run_schema_migrations

    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    uri = f"sqlite:///{tmp_path / 'unavailable_model.db'}"
    engine = create_engine(uri)
    run_schema_migrations(engine, uri)

    def columns():
        return {c["name"] for c in inspect(engine).get_columns("schedule_runs")}

    assert "unavailable_model" in columns()
    config = _alembic_config(uri)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, "3e7a91c4d2b8")
    assert "unavailable_model" not in columns()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    assert "unavailable_model" in columns()


@pytest.mark.parametrize("ending", ["error", "timeout", "shutdown"])
async def test_fallback_interruptions_keep_the_correct_outcome_and_model(schedule_id, turns, monkeypatch, ending):
    import asyncio
    import cowork.handlers.responses as responses_mod

    turns.endings.append(_failed("model_not_found"))
    original = responses_mod.ResponsesHandler.handle

    async def handle(self, request):
        if request.model is not None:
            return await original(self, request)
        if ending == "shutdown":
            raise scheduler.ServerShuttingDown()

        async def stream():
            if ending == "timeout":
                raise asyncio.TimeoutError()
            raise RuntimeError("private provider diagnostics")
            yield  # pragma: no cover
        return stream()

    async def stop(_conversation_id):
        pass

    monkeypatch.setattr(responses_mod.ResponsesHandler, "handle", handle)
    monkeypatch.setattr(scheduler, "_stop_timed_out_turn", stop)
    await _run_due_slot(schedule_id)
    (run,) = _runs(schedule_id)
    assert run.status == RunStatus.failed
    assert run.outcome == {
        "error": AttemptOutcome.uncertain,
        "timeout": AttemptOutcome.interrupted,
        "shutdown": AttemptOutcome.failed_before_execution,
    }[ending]
    assert run.unavailable_model == (None if ending == "shutdown" else PINNED)
    assert "private provider diagnostics" not in run.error


async def test_fallback_shutdown_marker_does_not_extend_failure_streak(schedule_id, turns, monkeypatch):
    turns.endings.extend([_failed("model_not_found"), COMPLETED])
    original = scheduler._turn_evidence

    async def evidence(conversation_id):
        result = await original(conversation_id)
        if result.reason == "completed":
            return scheduler._TurnEvidence(reason="interrupted", acted=True, shutdown_interrupted=True)
        return result

    monkeypatch.setattr(scheduler, "_turn_evidence", evidence)
    await _run_due_slot(schedule_id)
    (run,) = _runs(schedule_id)
    assert run.outcome == AttemptOutcome.interrupted
    assert run.unavailable_model == PINNED
    session, _ = _scoped()
    try:
        assert session.get(Schedule, schedule_id).consecutive_failures == 0
    finally:
        session.close()


@pytest.mark.parametrize("refusal", [scheduler.ServerShuttingDown, scheduler.TurnClaimUnavailable])
@pytest.mark.parametrize("created", [True, False])
@pytest.mark.parametrize("deleted", [True, False])
async def test_refused_fallback_discards_only_its_new_empty_conversation(
    schedule_id, turns, monkeypatch, refusal, created, deleted,
):
    import cowork.handlers.responses as responses_mod
    from cowork.models.conversation import Conversation
    from cowork.services.conversations import ConversationService

    turns.endings.append(_failed("model_not_found", message_id=str(uuid4())))
    original = responses_mod.ResponsesHandler.handle

    async def handle(self, request):
        if request.model is None:
            if refusal is scheduler.TurnClaimUnavailable:
                raise refusal(conversation_id=request.conversation)
            raise refusal()
        return await original(self, request)

    monkeypatch.setattr(responses_mod.ResponsesHandler, "handle", handle)
    if not deleted:
        def cannot_delete(*args, **kwargs):
            raise RuntimeError("delete failed")
        monkeypatch.setattr(scheduler, "_drop_failed_turn", cannot_delete)

    conversation_id = None
    if not created:
        session, scoped = _scoped()
        try:
            conversation_id = ConversationService(scoped).create_conversation(
                topic=session.get(Schedule, schedule_id).title,
            ).id
        finally:
            session.close()
    await scheduler.execute_schedule(schedule_id, is_manual=True, conversation_id=conversation_id)
    (run,) = _runs(schedule_id)
    assert run.status == RunStatus.failed
    assert run.outcome == AttemptOutcome.failed_before_execution
    assert run.unavailable_model is None
    assert "That model isn't available" in run.error
    original_id = UUID(turns.requests[0].conversation)
    session, _ = _scoped()
    try:
        if created and deleted:
            assert run.conversation_id is None
            assert session.get(Conversation, original_id) is None
        else:
            assert run.conversation_id == original_id
            assert session.get(Conversation, original_id) is not None
    finally:
        session.close()


@pytest.mark.parametrize("refused", ["first_turn", "fallback"])
async def test_a_discarded_conversation_loses_its_folders_only_once_a_turn_ran(
    schedule_id, turns, monkeypatch, refused,
):
    """A refused fallback discards a conversation whose pinned turn ran first.
    That turn may have staged files into the conversation's folder, a cloud
    turn's copy of the project's files among them, so the delete removes the
    folders too. A first turn refused before it started made none, so its
    delete stays rows only."""
    import cowork.handlers.responses as responses_mod
    from cowork.services.conversations import ConversationService

    turns.endings.append(_failed("model_not_found", message_id=str(uuid4())))
    original = responses_mod.ResponsesHandler.handle

    async def handle(self, request):
        if refused == "first_turn" or request.model is None:
            raise scheduler.ServerShuttingDown()
        return await original(self, request)

    monkeypatch.setattr(responses_mod.ResponsesHandler, "handle", handle)
    cleanups: list[bool] = []
    delete = ConversationService.delete_conversation

    def recording_delete(self, conversation_id, *, cleanup_project_files=True):
        cleanups.append(cleanup_project_files)
        return delete(self, conversation_id, cleanup_project_files=cleanup_project_files)

    monkeypatch.setattr(ConversationService, "delete_conversation", recording_delete)
    await scheduler.execute_schedule(schedule_id, is_manual=True)

    assert cleanups == [refused == "fallback"]
    (run,) = _runs(schedule_id)
    assert run.conversation_id is None
