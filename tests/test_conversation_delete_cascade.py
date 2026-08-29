"""Deleting a conversation (or clearing all its history) must drop the
conversation's object index too.

Regression: a stale `task_objects` row outlived the chat that produced it,
so a cleared conversation kept resurfacing an artifact it no longer owned —
even after the artifact file itself was gone.
"""

from __future__ import annotations

import shutil

import pytest
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.task_object import TaskObject
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.task_objects import TaskObjectService


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def _rows_for(session, conversation_id) -> list[TaskObject]:
    return list(
        session.exec(
            select(TaskObject).where(TaskObject.conversation_id == conversation_id)
        ).all()
    )


def test_delete_conversation_drops_task_objects(session):
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    TaskObjectService(ScopedSession(session, LOCAL_SCOPE)).index_artifact(
        conv.id, GENERAL_PROJECT_ID, "my-artifact"
    )
    assert _rows_for(session, conv.id), "precondition: artifact indexed"

    assert svc.delete_conversation(conv.id) is True
    assert (
        _rows_for(session, conv.id) == []
    ), "index rows must be gone with the conversation"


def test_clear_all_history_drops_task_objects(session):
    """Truncating from turn 0 (the UI's 'delete chat history') leaves the
    conversation but removes everything — its object index goes too."""
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    # One full turn: user message + assistant turn (mirrors a real exchange).
    from cowork.models.message import Message

    session.add(Message(conversation_id=conv.id, role="user", content='"make a plan"'))
    session.commit()
    svc.save_assistant_turn(conv.id, "done", events=[])
    TaskObjectService(ScopedSession(session, LOCAL_SCOPE)).index_artifact(
        conv.id, GENERAL_PROJECT_ID, "my-artifact"
    )
    assert _rows_for(session, conv.id), "precondition: artifact indexed"

    svc.delete_turn(conv.id, 0)  # clear from the first turn = clear all history
    assert _rows_for(session, conv.id) == [], "cleared history must drop the index"


def test_partial_turn_delete_keeps_task_objects(session):
    """Deleting a later turn (not turn 0) is a partial truncation — the
    surviving turns may still reference the artifact, so leave the index."""
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    from cowork.models.message import Message

    session.add(Message(conversation_id=conv.id, role="user", content='"first"'))
    session.commit()
    svc.save_assistant_turn(conv.id, "a1", events=[])
    session.add(Message(conversation_id=conv.id, role="user", content='"second"'))
    session.commit()
    svc.save_assistant_turn(conv.id, "a2", events=[])
    TaskObjectService(ScopedSession(session, LOCAL_SCOPE)).index_artifact(
        conv.id, GENERAL_PROJECT_ID, "my-artifact"
    )

    svc.delete_turn(conv.id, 1)  # drop only the second turn
    assert _rows_for(session, conv.id), "partial delete must keep the index"


# ── ENG-701: attachment cleanup on conversation / project delete ──────────
from pathlib import Path  # noqa: E402

from cowork.models.conversation import Conversation  # noqa: E402
from cowork.models.project import Project  # noqa: E402
from cowork.services.files import FileService, attachment_purpose, unlink_file_dirs  # noqa: E402
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession  # noqa: E402
from cowork.services.projects import ProjectService  # noqa: E402


def _attach(session, conversation_id, name="doc.txt"):
    return FileService(ScopedSession(session, LOCAL_SCOPE)).create_file_from_bytes(
        filename=name,
        content_type="text/plain",
        data=b"hello",
        purpose=attachment_purpose(str(conversation_id)),
    )


def _attachment_rows(session, conversation_id):
    return FileService(ScopedSession(session, LOCAL_SCOPE)).list_file_rows(
        attachment_purpose(str(conversation_id))
    )


def test_delete_conversation_removes_attachment_rows_and_bytes(session):
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    f = _attach(session, conv.id)
    path = Path(f.path)
    assert path.exists() and _attachment_rows(session, conv.id), "precondition"

    assert svc.delete_conversation(conv.id) is True
    assert _attachment_rows(session, conv.id) == [], "rows gone with the conversation"
    assert not path.exists(), "bytes unlinked"


def test_delete_conversation_leaves_other_conversations_and_purposes(session):
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    keep = svc.create_conversation("keep", project_id=GENERAL_PROJECT_ID)
    doomed = svc.create_conversation("doomed", project_id=GENERAL_PROJECT_ID)
    keep_file = _attach(session, keep.id)
    _attach(session, doomed.id)
    # A non-attachment purpose must never be touched.
    other = FileService(ScopedSession(session, LOCAL_SCOPE)).create_file_from_bytes(
        filename="c.bin",
        content_type="application/octet-stream",
        data=b"x",
        purpose="channel:some-channel",
    )

    svc.delete_conversation(doomed.id)

    assert _attachment_rows(session, keep.id) and Path(keep_file.path).exists()
    assert FileService(ScopedSession(session, LOCAL_SCOPE)).list_file_rows(
        "channel:some-channel"
    ), "non-attachment untouched"
    assert Path(other.path).exists()


def test_delete_project_cascades_conversations_and_attachments(session):
    proj = ProjectService(ScopedSession(session, LOCAL_SCOPE)).create_project(
        "eng701-cascade-test"
    )
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation("topic", project_id=proj.id)
    f = _attach(session, conv.id)
    path = Path(f.path)
    assert path.exists()

    assert (
        ProjectService(ScopedSession(session, LOCAL_SCOPE)).delete_project(proj.id)
        is True
    )
    # The conversation itself is gone (no more orphaned rows) …
    assert session.get(Conversation, conv.id) is None
    # … along with its attachment rows + bytes.
    assert _attachment_rows(session, conv.id) == []
    assert not path.exists()


def test_delete_project_aborts_all_conversation_stages_on_one_failure(
    session,
    monkeypatch,
):
    """One failed child stage leaves the project and every chat untouched."""
    proj = ProjectService(ScopedSession(session, LOCAL_SCOPE)).create_project(
        "eng701-fault-test"
    )
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    bad = svc.create_conversation("bad", project_id=proj.id)
    good = svc.create_conversation("good", project_id=proj.id)
    bad_file = _attach(session, bad.id)
    good_file = _attach(session, good.id)

    real = ConversationService.stage_delete_conversation_row

    def flaky(self, conversation):
        if str(conversation.id) == str(bad.id):
            raise RuntimeError("boom")
        return real(self, conversation)

    monkeypatch.setattr(
        ConversationService,
        "stage_delete_conversation_row",
        flaky,
    )

    with pytest.raises(RuntimeError, match="boom"):
        ProjectService(ScopedSession(session, LOCAL_SCOPE)).delete_project(proj.id)

    session.expire_all()
    assert session.get(Project, proj.id) is not None
    assert session.get(Conversation, bad.id) is not None
    assert session.get(Conversation, good.id) is not None
    assert _attachment_rows(session, bad.id)
    assert _attachment_rows(session, good.id)
    assert Path(bad_file.path).exists()
    assert Path(good_file.path).exists()


def test_delete_project_rolls_back_a_partially_staged_failed_conversation(
    session, monkeypatch
):
    """A conversation whose delete fails mid-flight — after staging its row
    deletes but before its own commit — must be rolled back. Otherwise the next
    commit in the cascade (the good conversation's, or the project's) flushes
    those pending deletes, wiping the failed conversation's data while its row
    survives (the ghost ea-rus flagged on #187)."""
    proj = ProjectService(ScopedSession(session, LOCAL_SCOPE)).create_project(
        "eng701-rollback-test"
    )
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    bad = svc.create_conversation("bad", project_id=proj.id)
    good = svc.create_conversation("good", project_id=proj.id)
    bad_file = _attach(session, bad.id)
    good_file = _attach(session, good.id)

    real = FileService.delete_by_purpose_for_parent_cascade

    def stage_then_raise(self, purpose):
        dirs = real(self, purpose)  # actually stage the attachment-row deletes
        if purpose == attachment_purpose(str(bad.id)):
            raise RuntimeError("boom after staging")
        return dirs

    monkeypatch.setattr(
        FileService,
        "delete_by_purpose_for_parent_cascade",
        stage_then_raise,
    )

    with pytest.raises(RuntimeError, match="boom after staging"):
        ProjectService(ScopedSession(session, LOCAL_SCOPE)).delete_project(proj.id)

    # Every partial child stage is rolled back. No later project commit can
    # flush a preceding conversation's deletes after another child fails.
    session.expire_all()
    assert session.get(Project, proj.id) is not None
    assert session.get(Conversation, bad.id) is not None, "bad conversation survives"
    assert session.get(Conversation, good.id) is not None
    assert _attachment_rows(
        session, bad.id
    ), "bad's staged attachment delete was rolled back, not flushed"
    assert _attachment_rows(session, good.id)
    assert Path(bad_file.path).exists()
    assert Path(good_file.path).exists()


def test_project_delete_audit_failure_preserves_chat_and_attachment_bytes(session):
    project_service = ProjectService(ScopedSession(session, LOCAL_SCOPE))
    project = project_service.create_project("eng701-audit-rollback-test")
    conversation_service = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conversation = conversation_service.create_conversation(
        "topic",
        project_id=project.id,
    )
    attachment = _attach(session, conversation.id)
    attachment_path = Path(attachment.path)
    ordinary = Path(project.path) / "ordinary.txt"
    ordinary.write_text("survive", encoding="utf-8")

    def fail_audit():
        raise RuntimeError("audit unavailable")

    with pytest.raises(RuntimeError, match="audit unavailable"):
        project_service.delete_project(project.id, before_commit=fail_audit)

    session.expire_all()
    assert session.get(Project, project.id) is not None
    assert session.get(Conversation, conversation.id) is not None
    assert _attachment_rows(session, conversation.id)
    assert attachment_path.read_bytes() == b"hello"
    assert ordinary.read_text(encoding="utf-8") == "survive"


def test_project_delete_commit_failure_preserves_chat_and_attachment_bytes(
    session,
    monkeypatch,
):
    scoped = ScopedSession(session, LOCAL_SCOPE)
    project_service = ProjectService(scoped)
    project = project_service.create_project("eng701-commit-rollback-test")
    conversation = ConversationService(scoped).create_conversation(
        "topic",
        project_id=project.id,
    )
    conversation_id = conversation.id
    attachment = _attach(session, conversation_id)
    attachment_path = Path(attachment.path)
    ordinary = Path(project.path) / "ordinary.txt"
    ordinary.write_text("survive commit failure", encoding="utf-8")

    def fail_commit():
        raise RuntimeError("database commit unavailable")

    monkeypatch.setattr(scoped, "commit", fail_commit)

    with pytest.raises(RuntimeError, match="database commit unavailable"):
        project_service.delete_project(project.id)

    session.expire_all()
    assert session.get(Project, project.id) is not None
    assert session.get(Conversation, conversation_id) is not None
    assert _attachment_rows(session, conversation_id)
    assert attachment_path.read_bytes() == b"hello"
    assert ordinary.read_text(encoding="utf-8") == "survive commit failure"


def test_delete_by_purpose_stages_without_committing(session):
    """The attachment-row delete must land in the CALLER's transaction, not its
    own — otherwise a crash between it and the conversation-row delete leaves a
    'ghost' conversation (row present, contents gone). Proof: after
    delete_by_purpose, a rollback brings the rows back, and the bytes are still
    on disk (unlink is the caller's post-commit step)."""
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    _attach(session, conv.id)

    dirs = FileService(ScopedSession(session, LOCAL_SCOPE)).delete_by_purpose(
        attachment_purpose(str(conv.id))
    )
    session.rollback()

    assert _attachment_rows(
        session, conv.id
    ), "delete_by_purpose must not commit on its own"
    assert dirs and all(
        d.exists() for d in dirs
    ), "bytes must survive until the caller commits"

    # And the helper only removes bytes once called explicitly (post-commit).
    FileService(ScopedSession(session, LOCAL_SCOPE)).delete_by_purpose(
        attachment_purpose(str(conv.id))
    )
    session.commit()
    unlink_file_dirs(dirs)
    assert not any(d.exists() for d in dirs), "bytes removed after commit + unlink"


# ── ENG-645: skill-draft sweep when a turn holding a skill card is deleted ──
from cowork.services.conversations import _skill_created_slug  # noqa: E402


def _general_drafts_dir(session) -> Path:
    project = session.get(Project, GENERAL_PROJECT_ID)
    return Path(project.path) / ".anton" / "skill_drafts"


def test_skill_created_slug_parses_only_matching_events():
    assert (
        _skill_created_slug({"type": "response.skill_created", "skill": {"slug": "a"}})
        == "a"
    )
    assert (
        _skill_created_slug(
            {"type": "response.artifact_created", "skill": {"slug": "a"}}
        )
        is None
    )
    assert _skill_created_slug({"type": "response.skill_created", "skill": {}}) is None
    assert _skill_created_slug({"type": "response.skill_created"}) is None
    assert _skill_created_slug("not a dict") is None


def test_delete_turn_sweeps_skill_draft(session):
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    from cowork.models.message import Message

    session.add(
        Message(conversation_id=conv.id, role="user", content='"build a skill"')
    )
    session.commit()
    svc.save_assistant_turn(
        conv.id,
        "made it",
        events=[{"type": "response.skill_created", "skill": {"slug": "turn-sweep-me"}}],
    )
    draft = _general_drafts_dir(session) / "turn-sweep-me"
    draft.mkdir(parents=True, exist_ok=True)
    (draft / "SKILL.md").write_text("---\nname: x\n---\n", encoding="utf-8")

    svc.delete_turn(conv.id, 0)
    assert not draft.exists(), "draft of a deleted skill-card turn must be swept"


def test_delete_turn_without_skill_card_keeps_drafts(session):
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    from cowork.models.message import Message

    session.add(Message(conversation_id=conv.id, role="user", content='"hi"'))
    session.commit()
    svc.save_assistant_turn(conv.id, "no skill here", events=[])
    draft = _general_drafts_dir(session) / "unrelated-draft"
    draft.mkdir(parents=True, exist_ok=True)
    (draft / "SKILL.md").write_text("x", encoding="utf-8")

    svc.delete_turn(conv.id, 0)
    assert draft.exists(), "a turn without a skill card must not sweep drafts"
    shutil.rmtree(draft, ignore_errors=True)


def test_new_skill_card_supersedes_earlier_versions(session):
    """A later skill_created for the same slug drops the earlier one; a
    different slug is untouched — history keeps one card per skill (latest)."""
    from cowork.models.message import Message
    from cowork.models.message_event import MessageEvent

    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)

    def _emit(user, text, slug, instr):
        session.add(Message(conversation_id=conv.id, role="user", content=f'"{user}"'))
        session.commit()
        svc.save_assistant_turn(
            conv.id,
            text,
            events=[
                {
                    "type": "response.skill_created",
                    "skill": {"slug": slug, "instructions": instr},
                }
            ],
        )

    _emit("build", "v1", "dedup-me", "v1")
    _emit("also", "other", "other-skill", "keep")
    _emit("refine", "v2", "dedup-me", "v2")

    msg_ids = [
        m.id
        for m in session.exec(
            select(Message).where(Message.conversation_id == conv.id)
        ).all()
    ]
    events = session.exec(
        select(MessageEvent).where(MessageEvent.message_id.in_(msg_ids))
    ).all()
    dedup = [e for e in events if _skill_created_slug(e.event_data) == "dedup-me"]
    other = [e for e in events if _skill_created_slug(e.event_data) == "other-skill"]

    assert len(dedup) == 1, "only the latest same-slug card survives"
    assert dedup[0].event_data["skill"]["instructions"] == "v2"
    assert len(other) == 1, "a different slug is not superseded"


def test_delete_turn_discards_stream_buffers(session):
    """turn_id == message count, so after truncation the next turn reuses a
    deleted turn's buffer file (append mode) and replays the old answer. The
    delete must drop the conversation's stream buffers."""
    from cowork.models.message import Message
    from cowork.streaming.backend import get_streams_dir
    from cowork.streaming.buffer import turn_buffer_path

    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    session.add(Message(conversation_id=conv.id, role="user", content='"hi"'))
    session.commit()
    svc.save_assistant_turn(conv.id, "answer", events=[])

    buf = turn_buffer_path(get_streams_dir(), str(conv.id), 0)
    buf.parent.mkdir(parents=True, exist_ok=True)
    buf.write_text('{"seq": 0}\n', encoding="utf-8")
    assert buf.exists()

    svc.delete_turn(conv.id, 0)
    assert not buf.exists(), "stale stream buffers must be discarded on turn delete"


# ── schedule + channel references: released, not deleted ──────────────────
from datetime import datetime, timezone  # noqa: E402

import sqlalchemy as sa  # noqa: E402
from sqlalchemy import event  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402
from sqlmodel import SQLModel, create_engine  # noqa: E402

from cowork.models.channel import ChannelBinding, ChannelSession  # noqa: E402
from cowork.models.schedule import Schedule, ScheduleRun  # noqa: E402
from cowork.schemas.schedules import RunStatus  # noqa: E402
from cowork.scheduler import _apply_success_write_back  # noqa: E402
from cowork.services.channel_bindings import ChannelBindingService  # noqa: E402
from cowork.services.schedules import ScheduleRunService, ScheduleService  # noqa: E402

_WHEN = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _schedule(session, project_id=GENERAL_PROJECT_ID, title="Daily report") -> Schedule:
    schedule = Schedule(
        title=title,
        prompt="Summarize",
        cadence="daily",
        next_run_at=_WHEN,
        model="default",
        project_id=project_id,
    )
    session.add(schedule)
    session.commit()
    session.refresh(schedule)
    return schedule


def _finished_run(session, schedule_id, conversation_id) -> ScheduleRun:
    """A run that produced a conversation, built the way the scheduler builds
    one, create_run then finish_run, so the row carries a real status and
    duration rather than hand-set values."""
    runs = ScheduleRunService(ScopedSession(session, LOCAL_SCOPE))
    run = runs.create_run(schedule_id, is_manual=True)
    return runs.finish_run(run.id, conversation_id=conversation_id)


def _binding(
    session, conversation_id, group_id, project_id=GENERAL_PROJECT_ID
) -> ChannelBinding:
    binding = ChannelBinding(
        channel_type="telegram",
        external_group_id=group_id,
        external_thread_key="__default__",
        anton_conversation_id=conversation_id,
        anton_project_id=project_id,
        trigger_rule="always",
    )
    session.add(binding)
    session.commit()
    session.refresh(binding)
    return binding


def _orphaned_run_count(session, schedule_id) -> int:
    """The desktop-side check from the ticket's walkthrough: SQLite will not
    raise on its own, so a left join is the only thing that catches an orphan.
    Scoped to one schedule's runs, because the shared test database carries
    rows other tests wrote raw and this must not assert on those."""
    return len(
        session.exec(
            sa.select(ScheduleRun.id)
            .outerjoin(Conversation, Conversation.id == ScheduleRun.conversation_id)
            .where(
                ScheduleRun.schedule_id == schedule_id,
                ScheduleRun.conversation_id.is_not(None),
                Conversation.id.is_(None),
            )
        ).all()
    )


def test_delete_conversation_releases_its_schedule_and_binding(session):
    """The three references nothing else clears. On Postgres leaving any of them
    is a ForeignKeyViolation that 500s the delete; on SQLite it is a silent
    orphan. Either way the rows themselves must survive: a run is audit history,
    and a binding is a live external chat."""
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation("scheduled run", project_id=GENERAL_PROJECT_ID)
    schedule = _schedule(session)
    run = _finished_run(session, schedule.id, conv.id)
    schedule.last_result_conversation_id = conv.id
    session.add(schedule)
    session.commit()
    binding = _binding(session, conv.id, "release-on-delete-test")
    session.add(
        ChannelSession(
            binding_id=binding.id,
            external_session_key="k",
            anton_session_id=str(conv.id),
        )
    )
    session.commit()
    run_status, run_duration = run.status, run.duration_ms
    assert run_duration is not None, "precondition: the run recorded a duration"

    try:
        assert svc.delete_conversation(conv.id) is True
        session.expire_all()

        assert session.get(Conversation, conv.id) is None, "the conversation is gone"
        # The run survives with its verdict intact. Only the link is released.
        kept_run = session.get(ScheduleRun, run.id)
        assert kept_run is not None, "run history must outlive the chat it produced"
        assert kept_run.conversation_id is None
        assert (kept_run.status, kept_run.duration_ms) == (run_status, run_duration)
        # So does the schedule, still on its cadence.
        kept_schedule = session.get(Schedule, schedule.id)
        assert kept_schedule is not None and kept_schedule.enabled is True
        assert kept_schedule.last_result_conversation_id is None
        # So does the binding, so the external chat stays bound to its project.
        kept_binding = session.get(ChannelBinding, binding.id)
        assert kept_binding is not None
        assert kept_binding.anton_conversation_id is None
        assert kept_binding.anton_project_id == GENERAL_PROJECT_ID
        # Its session rows go with the pointer, as in every other detach.
        # anton_session_id would otherwise name a conversation that is gone.
        assert (
            session.exec(
                select(ChannelSession).where(ChannelSession.binding_id == binding.id)
            ).all()
            == []
        )
        assert _orphaned_run_count(session, schedule.id) == 0
    finally:
        row = session.get(ChannelBinding, binding.id)
        if row is not None:
            ChannelBindingService(ScopedSession(session, LOCAL_SCOPE)).delete(row.id)
        ScheduleService(ScopedSession(session, LOCAL_SCOPE)).delete_schedule(
            schedule.id
        )
        # A red run above leaves the conversation behind in the shared
        # session-scoped test.db; sweep it so later tests never see it.
        if session.get(Conversation, conv.id) is not None:
            ConversationService(
                ScopedSession(session, LOCAL_SCOPE)
            ).delete_conversation(conv.id)


def test_delete_conversation_leaves_another_conversations_run_alone(session):
    """The release is keyed on one conversation id. A sibling run pointing at a
    different conversation must keep its link."""
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    doomed = svc.create_conversation("doomed", project_id=GENERAL_PROJECT_ID)
    keep = svc.create_conversation("keep", project_id=GENERAL_PROJECT_ID)
    schedule = _schedule(session, title="Two runs")
    doomed_run = _finished_run(session, schedule.id, doomed.id)
    keep_run = _finished_run(session, schedule.id, keep.id)

    try:
        assert svc.delete_conversation(doomed.id) is True
        session.expire_all()

        assert session.get(ScheduleRun, doomed_run.id).conversation_id is None
        assert session.get(ScheduleRun, keep_run.id).conversation_id == keep.id
        assert _orphaned_run_count(session, schedule.id) == 0
    finally:
        ScheduleService(ScopedSession(session, LOCAL_SCOPE)).delete_schedule(
            schedule.id
        )
        # `keep` (and `doomed`, on a red run) live in the shared test.db;
        # sweep them so later tests never see them.
        for conv_id in (doomed.id, keep.id):
            if session.get(Conversation, conv_id) is not None:
                ConversationService(
                    ScopedSession(session, LOCAL_SCOPE)
                ).delete_conversation(conv_id)


def _fk_enforcing_engine():
    """A throwaway engine that enforces foreign keys, which the shared test
    database deliberately does not.

    This is the only way the unit suite sees what cloud Postgres sees. It must
    stay local to this test: the shared conftest engine has tests that rely on
    enforcement being off, notably the two project-delete fault-injection tests
    above, which leave a live conversation under a deleted project on purpose.
    """
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    SQLModel.metadata.create_all(engine)
    return engine


def test_delete_conversation_does_not_violate_a_foreign_key(tmp_path):
    """The production failure itself, reproduced. With foreign keys enforced the
    unfixed cascade raises IntegrityError on the conversations DELETE, which is
    the 500 users saw. Runs on its own engine, with autoflush off to match the
    session factory the server actually uses."""
    engine = _fk_enforcing_engine()
    with Session(engine, autoflush=False, expire_on_commit=False) as s:
        project = Project(name="fk-project", path=str(tmp_path / "fk-project"))
        s.add(project)
        s.commit()
        scoped = ScopedSession(s, LOCAL_SCOPE)
        conv = ConversationService(scoped).create_conversation(
            "run", project_id=project.id
        )
        schedule = _schedule(s, project_id=project.id, title="FK schedule")
        run = _finished_run(s, schedule.id, conv.id)
        schedule.last_result_conversation_id = conv.id
        s.add(schedule)
        s.commit()
        binding = _binding(s, conv.id, "fk-group", project_id=project.id)

        assert ConversationService(scoped).delete_conversation(conv.id) is True

        s.expire_all()
        assert s.get(Conversation, conv.id) is None
        assert s.get(ScheduleRun, run.id).conversation_id is None
        assert s.get(Schedule, schedule.id).last_result_conversation_id is None
        assert s.get(ChannelBinding, binding.id).anton_conversation_id is None


def test_finishing_a_run_whose_conversation_was_deleted_mid_flight(tmp_path):
    """A run holds its conversation id for the whole run and writes it back at
    the end, so deleting the chat mid-run used to point the run at a row that
    was already gone. On Postgres that write raises, the scheduler logs and
    moves on, and the run is stranded at `running` forever: has_active_run then
    reports the schedule busy and the due-check skips it on every poll.

    Two sessions on purpose, because production is two sessions. The scheduler
    keeps one for the whole run and it CREATED the conversation, so the
    instance sits in that session's identity map (the factory runs
    expire_on_commit=False) while the user's request session deletes the row.
    A single-session version expunges the instance at the delete's commit, so
    even a `Session.get` in still_exists then truly queries — it would
    green-light an existence check that never asks the database.
    """
    engine = _fk_enforcing_engine()
    with Session(
        engine, autoflush=False, expire_on_commit=False
    ) as scheduler_s, Session(
        engine, autoflush=False, expire_on_commit=False
    ) as request_s:
        project = Project(name="midflight", path=str(tmp_path / "midflight"))
        scheduler_s.add(project)
        scheduler_s.commit()
        scoped = ScopedSession(scheduler_s, LOCAL_SCOPE)
        # Created on the scheduler's own session, exactly as execute_schedule's
        # cron path does — this is what parks it in the identity map.
        conv = ConversationService(scoped).create_conversation(
            "in flight", project_id=project.id
        )
        schedule = _schedule(
            scheduler_s, project_id=project.id, title="Mid-flight schedule"
        )
        runs = ScheduleRunService(scoped)
        run = runs.create_run(schedule.id, is_manual=True)
        runs.set_run_conversation(run.id, conv.id)

        # The user deletes the chat from their own request session while the
        # turn is still streaming.
        assert (
            ConversationService(
                ScopedSession(request_s, LOCAL_SCOPE)
            ).delete_conversation(conv.id)
            is True
        )

        # The race ordering: the scheduler attaches the conversation again
        # after the delete. Must neither raise nor resurrect the pointer.
        runs.set_run_conversation(run.id, conv.id)

        # The success branch points the schedule back at the conversation id
        # it captured before the delete.
        _apply_success_write_back(schedule, runs, conv.id, scheduler_s)
        scheduler_s.commit()

        # And the finally block still reports the outcome with the same id.
        finished = runs.finish_run(run.id, conversation_id=conv.id)

        assert finished.status == RunStatus.success, "the run must not be left running"
        assert finished.conversation_id is None
        scheduler_s.expire_all()
        assert (
            scheduler_s.get(Schedule, schedule.id).last_result_conversation_id is None
        )
        assert scheduler_s.get(ScheduleRun, run.id).conversation_id is None
        assert _orphaned_run_count(scheduler_s, schedule.id) == 0
        assert (
            runs.has_active_run(schedule.id) is False
        ), "a stranded run would wedge the schedule"


def test_run_writes_converge_when_the_delete_wins_the_race(tmp_path, monkeypatch):
    """still_exists is a read followed by a separate write, so a delete can
    commit between the two; the write's own commit then raises. Both writers
    must roll back and converge on the released pointer instead of stranding
    the run at `running`."""
    engine = _fk_enforcing_engine()
    with Session(engine, autoflush=False, expire_on_commit=False) as s:
        project = Project(name="race", path=str(tmp_path / "race"))
        s.add(project)
        s.commit()
        scoped = ScopedSession(s, LOCAL_SCOPE)
        conv = ConversationService(scoped).create_conversation(
            "raced", project_id=project.id
        )
        schedule = _schedule(s, project_id=project.id, title="Race schedule")
        runs = ScheduleRunService(scoped)
        run = runs.create_run(schedule.id, is_manual=True)

        assert ConversationService(scoped).delete_conversation(conv.id) is True
        # Freeze the check on the answer it gave before the delete committed.
        monkeypatch.setattr(runs, "still_exists", lambda cid: cid)

        runs.set_run_conversation(run.id, conv.id)
        s.expire_all()
        assert s.get(ScheduleRun, run.id).conversation_id is None

        finished = runs.finish_run(run.id, conversation_id=conv.id)
        assert finished.status == RunStatus.success
        assert finished.conversation_id is None
