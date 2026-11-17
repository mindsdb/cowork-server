"""A remote turn's workspace staging, its seed and skills reads and the memory
saves its pod reports work on the shared file store (COWORK_HOME), so each runs
as one unit on the file store's threads (run_efs_unit), as does the SKILL.md
read that decides whether an always-on skill reaches a turn. A slow file store
then holds file store slots, which EFS_SLOTS and the caller's EFS_SHARE count,
and a busy one refuses the work in POOL_TIMEOUT instead of holding a database
unit's thread outside them.

The guard tests run the real staging, the real memory save, and a turn start
and remote web turn with an always-on skill, in org tenancy, where the
blocking guard (tests/_blocking_guard.py) records file store work on any
thread run_efs did not start.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import NamedTuple
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

import _blocking_guard as blocking_guard
import cowork.db.session as db_session
from cowork.db.scoped import LOCAL_SCOPE, TenantScope, scope_from_principal
from cowork.db.units import FilesystemBusy
from cowork.handlers import responses as responses_mod
from cowork.handlers.responses import ResponsesHandler, save_turn_memory, stage_remote_workspace
from cowork.harnesses.memory.store import GlobalMemoryStore
from cowork.principal import Principal
from cowork.server import create_app
from test_run_efs import holding

ORG = "5d6c1f0e-2b7a-4c3d-8e9f-0a1b2c3d4e72"
USER = "6e7d2a1f-3c8b-4d4e-9f0a-1b2c3d4e5f72"
MEMBER = {"X-User-Id": USER, "X-Organization-Id": ORG}
PRINCIPAL = Principal(user_id=USER, org_id=ORG)
MEMORY_WARNING = "[responses] no database connection or file store slot freed in time to apply a turn's memory"


class OrgConversation(NamedTuple):
    id: UUID
    project_path: Path
    scope: TenantScope


def make_org_conversation(principal: Principal, project_name: str) -> OrgConversation:
    """A member's conversation in a project of their organization, made
    through the app, so the project's folder is under COWORK_HOME."""
    member = {"X-User-Id": principal.user_id, "X-Organization-Id": principal.org_id}
    client = TestClient(create_app())
    project = client.post("/api/v1/projects/", json={"name": project_name}, headers=member)
    assert project.status_code == 201, project.text
    conversation = client.post(
        "/api/v1/conversations/", json={"title": "t", "projectId": project.json()["id"]}, headers=member,
    )
    assert conversation.status_code == 201, conversation.text
    return OrgConversation(
        id=UUID(conversation.json()["id"]),
        project_path=Path(project.json()["path"]),
        scope=scope_from_principal(principal),
    )


@pytest.fixture
def org_conversation(org_deployment) -> OrgConversation:
    """A member's conversation in a project of their own."""
    return make_org_conversation(PRINCIPAL, f"staging-{uuid4().hex[:8]}")


def copy_folder(conversation: OrgConversation) -> Path:
    return conversation.project_path / "conversations" / str(conversation.id) / "project-files"


@pytest.fixture
def one_second_wait(monkeypatch):
    monkeypatch.setattr(db_session.settings.database, "pool_timeout", 1)


def test_a_remote_turns_staging_runs_on_the_file_store_threads(org_conversation):
    """The real staging, unstubbed: it makes the project's artifacts base and
    copies the project's instructions into the conversation's workspace, and
    the guard finds no file store work off run_efs."""
    instructions = org_conversation.project_path / ".anton" / "anton.md"
    instructions.parent.mkdir(parents=True, exist_ok=True)
    instructions.write_text("Answer in charts.")

    with blocking_guard.watching() as found:
        asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))

    assert sorted(found) == []
    assert (org_conversation.project_path / ".anton" / "artifacts").is_dir()
    staged = org_conversation.project_path / "conversations" / str(org_conversation.id) / ".anton" / "anton.md"
    assert staged.read_text() == "Answer in charts."


def test_a_remote_turns_project_file_copy_runs_on_the_file_store_threads_with_no_session(
    org_conversation, monkeypatch, caplog,
):
    """The copy of the project's files can take seconds on a first turn, so it
    runs after the staging unit, on the file store's threads, while no
    database session is open: it never holds a pooled connection. It logs one
    project_files_staged line with the turn's cost, at WARNING, which the
    hosted LOG_LEVEL keeps."""
    from contextlib import contextmanager

    from cowork.db import units
    from cowork.services import project_file_copy

    (org_conversation.project_path / "chapter-1.md").write_text("The porter kept the keys.")
    open_sessions: list[int] = []
    real_unit_session = units.unit_session

    @contextmanager
    def counted_unit_session(*, scope):
        open_sessions.append(1)
        try:
            with real_unit_session(scope=scope) as session:
                yield session
        finally:
            open_sessions.pop()

    seen: list[tuple[bool, int]] = []
    real_stage = project_file_copy.stage_project_files

    def watched_stage(*args, **kwargs):
        seen.append((units.on_efs_thread(), len(open_sessions)))
        return real_stage(*args, **kwargs)

    monkeypatch.setattr(units, "unit_session", counted_unit_session)
    monkeypatch.setattr(project_file_copy, "stage_project_files", watched_stage)

    with caplog.at_level(logging.WARNING, logger=responses_mod.logger.name):
        with blocking_guard.watching() as found:
            asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))

    assert sorted(found) == []
    assert seen == [(True, 0)]
    copy = org_conversation.project_path / "conversations" / str(org_conversation.id) / "project-files"
    assert (copy / "chapter-1.md").read_text() == "The porter kept the keys."
    [record] = [r for r in caplog.records if getattr(r, "event", None) == "project_files_staged"]
    assert (record.name, record.levelname) == (responses_mod.logger.name, "WARNING")
    assert record.getMessage() == "Copied the project's files for a cloud turn"
    # The deployments' JSON log keeps every field of the line.
    from cowork.common.structured_logging import make_formatter

    line = json.loads(make_formatter().format(record))
    assert {key: line.get(key) for key in (
        "event", "outcome", "files", "bytes", "copied", "copied_bytes", "not_copied", "conversation_id",
    )} == {
        "event": "project_files_staged", "outcome": "staged", "files": 1, "bytes": 25, "copied": 1,
        "copied_bytes": 25, "not_copied": 0, "conversation_id": str(org_conversation.id),
    }
    assert isinstance(line["duration_ms"], int)


def test_a_project_file_copy_that_cant_finish_empties_the_copy_and_the_turn_goes_on(
    org_conversation, monkeypatch, caplog,
):
    """A failed copy never fails the turn. It leaves no file behind that the
    project may no longer hold, only the list that tells the agent the copy
    failed, so it says so rather than answering from the web."""
    from cowork.services import project_file_copy

    (org_conversation.project_path / "chapter-1.md").write_text("The porter kept the keys.")
    asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))
    copy = org_conversation.project_path / "conversations" / str(org_conversation.id) / "project-files"
    assert (copy / "chapter-1.md").exists()

    def broken(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(project_file_copy, "_stage_project_files", broken)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=responses_mod.logger.name):
        asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))

    assert [path.name for path in copy.iterdir()] == [project_file_copy.NOT_COPIED_FILENAME]
    listed = (copy / project_file_copy.NOT_COPIED_FILENAME).read_bytes()
    assert project_file_copy.NotCopiedFiles.model_validate_json(listed) == project_file_copy.NotCopiedFiles(
        copy_failed=True,
    )
    [record] = [r for r in caplog.records if getattr(r, "event", None) == "project_files_staged"]
    assert (record.outcome, record.files) == ("failed", 0)
    assert record.getMessage() == "Could not copy the project's files for a cloud turn; its copy was emptied"


def test_a_copy_that_cant_empty_its_folder_refuses_the_turn(org_conversation, monkeypatch):
    """A folder that may still hold a copy the project no longer backs is
    never handed to a turn."""
    from cowork.services import project_file_copy

    (org_conversation.project_path / "chapter-1.md").write_text("The porter kept the keys.")
    asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))

    def broken(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(project_file_copy, "_prune_folder", broken)
    monkeypatch.setattr(project_file_copy._CopyFolders, "set_aside", broken)

    with pytest.raises(project_file_copy.ProjectFilesRefused) as refused:
        asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))

    assert refused.value.refusal is project_file_copy.ProjectFilesRefusal.NOT_EMPTIED
    assert str(refused.value) == (
        f"the project files copy of conversation {org_conversation.id} was refused: "
        "the copy could not finish and its folder could not be emptied"
    )


def test_a_refused_copys_log_lines_carry_their_values_as_fields(org_conversation, monkeypatch, caplog):
    """The deployments' JSON log keeps a line's static message and drops its
    template values, and a line with an exception becomes "Operation
    failed". So every line of a refused copy names its conversation, and
    the staged line its refusal, as fields, never as template values."""
    from cowork.common.structured_logging import make_formatter
    from cowork.services import project_file_copy

    (org_conversation.project_path / "chapter-1.md").write_text("The porter kept the keys.")
    asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))

    def broken(*args, **kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(project_file_copy, "_prune_folder", broken)
    monkeypatch.setattr(project_file_copy._CopyFolders, "set_aside", broken)
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(project_file_copy.ProjectFilesRefused):
            asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))

    records = [
        r for r in caplog.records
        if r.name in {project_file_copy.logger.name, responses_mod.logger.name} and r.levelno >= logging.WARNING
    ]
    assert [(r.name, r.event) for r in records] == [
        (project_file_copy.logger.name, "project_files_copy_failed"),
        (project_file_copy.logger.name, "project_files_empty_set_aside"),
        (project_file_copy.logger.name, "project_files_copy_not_emptied"),
        (responses_mod.logger.name, "project_files_staged"),
    ]
    lines = [json.loads(make_formatter().format(r)) for r in records]
    for record, line in zip(records, lines, strict=True):
        assert record.args == ()
        assert (line["event"], line["conversation_id"]) == (record.event, str(org_conversation.id))
    assert (lines[-1]["outcome"], lines[-1]["refusal"]) == ("refused", "not_emptied")


def _delete_conversation(conversation: OrgConversation) -> None:
    """The member's own delete, through the app: it removes the row, then
    the conversation's folder and the copy's record."""
    deleted = TestClient(create_app()).delete(f"/api/v1/conversations/{conversation.id}", headers=MEMBER)
    assert deleted.status_code == 200, deleted.text


def test_a_conversation_deleted_before_its_copy_starts_keeps_no_folder_or_record(
    org_conversation, monkeypatch, caplog,
):
    """The delete lands after the staging unit read the conversation and
    before the copy opens its folder. The copy makes the folder, its copies
    and its record again, and its own check passes on the folder it made,
    so the staging reads the conversation's row again once the copy ends and
    removes them, and the turn's line says the conversation went away."""
    from cowork.services import project_file_copy

    (org_conversation.project_path / "chapter-1.md").write_text("The porter kept the keys.")
    asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))
    conversation = org_conversation.project_path / "conversations" / str(org_conversation.id)
    record = project_file_copy.copy_record_path(org_conversation.scope, org_conversation.id)
    assert (conversation / "project-files" / "chapter-1.md").exists() and record.exists()
    real_stage = project_file_copy.stage_project_files
    made_again: list[bool] = []

    def deleted_first(*args, **kwargs):
        _delete_conversation(org_conversation)
        assert not conversation.exists() and not record.exists()
        staged = real_stage(*args, **kwargs)
        made_again.append(conversation.exists() and record.exists())
        return staged

    monkeypatch.setattr(project_file_copy, "stage_project_files", deleted_first)
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger=responses_mod.logger.name):
        asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))

    assert made_again == [True]
    assert not conversation.exists()
    assert not record.exists()
    [line] = [r for r in caplog.records if getattr(r, "event", None) == "project_files_staged"]
    assert (line.outcome, line.conversation_id) == ("conversation_gone", str(org_conversation.id))
    # Nothing is kept, and what the turn wrote stays its cost.
    assert (line.files, line.bytes, line.copied, line.copied_bytes) == (0, 0, 1, 25)


@pytest.mark.parametrize("taken_from", ["project-files/docs", "."], ids=["a folder of its copy", "its own folder"])
def test_a_deleted_conversation_whose_pod_took_a_folders_access_away_keeps_no_folder(org_conversation, taken_from):
    """The pod runs as the server's user, so it can make any folder in its
    conversation's folder 0o000, its mount included. The delete gives each
    folder its owner's access back and removes the whole folder, with the
    copy of the project's files in it."""
    import os

    (org_conversation.project_path / "docs").mkdir()
    (org_conversation.project_path / "docs" / "plan.md").write_text("The porter kept the keys.")
    asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))
    conversation = org_conversation.project_path / "conversations" / str(org_conversation.id)
    assert (copy_folder(org_conversation) / "docs" / "plan.md").exists()
    taken = conversation / taken_from
    os.chmod(taken, 0o000)
    try:
        _delete_conversation(org_conversation)
    finally:
        if taken.exists():
            os.chmod(taken, 0o700)

    assert not conversation.exists()


def test_a_conversation_folder_the_delete_cant_remove_is_logged_with_its_id(org_conversation, monkeypatch, caplog):
    """A fixed message with the conversation as a field: the log formatter
    drops template values, so the hosted line names the conversation."""
    import errno

    from cowork.common.structured_logging import make_formatter
    from cowork.services import files as files_mod

    conversation = org_conversation.project_path / "conversations" / str(org_conversation.id)
    conversation.mkdir(parents=True)

    def refused(folder, name):
        raise PermissionError(errno.EACCES, "refused")

    monkeypatch.setattr(files_mod, "dir_rmtree", refused)

    with caplog.at_level(logging.WARNING, logger=files_mod.logger.name):
        files_mod.remove_conversation_workspace_dir(org_conversation.project_path, org_conversation.id)

    [record] = [r for r in caplog.records if r.name == files_mod.logger.name and r.levelno == logging.WARNING]
    assert (record.getMessage(), record.args) == ("Could not remove a deleted conversation's folder", ())
    line = json.loads(make_formatter().format(record))
    assert (line["event"], line["conversation_id"], line["error_type"]) == (
        "conversation_folder_not_removed", str(org_conversation.id), "PermissionError",
    )
    assert conversation.exists()


def _staged_line(caplog) -> logging.LogRecord:
    [line] = [
        r for r in caplog.records
        if r.name == responses_mod.logger.name and getattr(r, "event", None) == "project_files_staged"
    ]
    return line


def test_a_conversation_moved_before_its_copy_starts_leaves_no_copy_in_the_project_it_left(
    org_conversation, monkeypatch, caplog,
):
    """The move lands after the staging unit read the conversation and
    before the copy opens its folder. The move removes the old copy, and the
    copy makes it again in the project the conversation left, where nothing
    would remove it before that project's delete. So the staging finds the
    conversation in another project once the copy ends, and removes it."""
    from cowork.services import project_file_copy

    (org_conversation.project_path / "chapter-1.md").write_text("The porter kept the keys.")
    asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))
    conversation = org_conversation.project_path / "conversations" / str(org_conversation.id)
    left_copy = copy_folder(org_conversation)
    assert (left_copy / "chapter-1.md").exists()
    client = TestClient(create_app())
    joined = client.post("/api/v1/projects/", json={"name": f"joined-{uuid4().hex[:8]}"}, headers=MEMBER)
    assert joined.status_code == 201, joined.text
    real_stage = project_file_copy.stage_project_files
    made_again: list[bool] = []

    def moved_first(*args, **kwargs):
        moved = client.patch(
            f"/api/v1/conversations/{org_conversation.id}", json={"projectId": joined.json()["id"]}, headers=MEMBER,
        )
        assert moved.status_code == 200, moved.text
        assert not left_copy.exists()
        staged = real_stage(*args, **kwargs)
        made_again.append((left_copy / "chapter-1.md").exists())
        return staged

    monkeypatch.setattr(project_file_copy, "stage_project_files", moved_first)
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger=responses_mod.logger.name):
        asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))

    assert made_again == [True]
    assert not left_copy.exists()
    assert not (conversation / project_file_copy.REMOVED_DIRNAME).exists()
    line = _staged_line(caplog)
    assert (line.outcome, line.files, line.bytes) == ("conversation_gone", 0, 0)


def test_a_project_renamed_while_its_copy_runs_keeps_the_copy(org_conversation, monkeypatch, caplog):
    """A rename moves the project's folder with the copy in it, and the
    turn's pod mounts the renamed folder. The conversation's project path
    changed, but nothing is left at the old one, so the copy stays and the
    turn's line says it was staged."""
    from cowork.services import project_file_copy

    (org_conversation.project_path / "chapter-1.md").write_text("The porter kept the keys.")
    client = TestClient(create_app())
    project_id = client.get(f"/api/v1/conversations/{org_conversation.id}", headers=MEMBER).json()["projectId"]
    real_stage = project_file_copy.stage_project_files
    renamed: list[Path] = []

    def renamed_after_the_copy(*args, **kwargs):
        staged = real_stage(*args, **kwargs)
        answer = client.patch(f"/api/v1/projects/{project_id}", json={"name": f"renamed-{uuid4().hex[:8]}"}, headers=MEMBER)
        assert answer.status_code == 200, answer.text
        renamed.append(Path(answer.json()["path"]))
        return staged

    monkeypatch.setattr(project_file_copy, "stage_project_files", renamed_after_the_copy)

    with caplog.at_level(logging.WARNING, logger=responses_mod.logger.name):
        asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))

    [new_path] = renamed
    assert not org_conversation.project_path.exists()
    copied = new_path / "conversations" / str(org_conversation.id) / "project-files" / "chapter-1.md"
    assert copied.read_text() == "The porter kept the keys."
    line = _staged_line(caplog)
    assert (line.outcome, line.files) == ("staged", 1)


async def test_a_turn_cancelled_while_its_removal_waits_for_a_slot_still_removes_the_copy(
    org_conversation, monkeypatch,
):
    """The delete lands before the copy opens the folder, and did not cancel
    the turn (another replica served it, say). The copy ends normally and
    makes the folder again; the staging finds the row gone, and its removal
    waits for a file store slot. A cancel then, from the idle watchdog or a
    shutdown, still lets the removal run once a slot frees, since nothing
    else removes the folder and the record."""
    from cowork.db import units
    from cowork.services import project_file_copy

    (org_conversation.project_path / "chapter-1.md").write_text("The porter kept the keys.")
    conversation = org_conversation.project_path / "conversations" / str(org_conversation.id)
    record = project_file_copy.copy_record_path(org_conversation.scope, org_conversation.id)
    real_stage = project_file_copy.stage_project_files

    def deleted_first(*args, **kwargs):
        _delete_conversation(org_conversation)
        return real_stage(*args, **kwargs)

    monkeypatch.setattr(project_file_copy, "stage_project_files", deleted_first)
    real_run_db = responses_mod.run_db
    row_read, slots_held = asyncio.Event(), asyncio.Event()

    async def row_read_then_wait(fn, **kwargs):
        result = await real_run_db(fn, **kwargs)
        row_read.set()
        await slots_held.wait()
        return result

    monkeypatch.setattr(responses_mod, "run_db", row_read_then_wait)
    turn = asyncio.create_task(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))
    await asyncio.wait_for(row_read.wait(), timeout=30)
    assert conversation.exists() and record.exists()

    async with holding(db_session.settings.storage.efs_slots, LOCAL_SCOPE):
        slots_held.set()
        for _ in range(3000):
            if units._efs_slots().statistics().tasks_waiting:
                break
            await asyncio.sleep(0.01)
        assert units._efs_slots().statistics().tasks_waiting == 1
        turn.cancel()
        await asyncio.sleep(0.05)
        assert conversation.exists()

    with pytest.raises(asyncio.CancelledError):
        await turn

    assert not conversation.exists()
    assert not record.exists()


async def test_a_delete_that_cancels_the_turn_during_its_copy_keeps_no_folder_or_record(
    org_conversation, monkeypatch,
):
    """A conversation delete cancels its turn, and a copy that started runs
    to its end first. When the delete removed the folder before the copy
    opened it, the copy makes it again, so the cancelled staging still reads
    the row once the copy ends and removes what the copy made."""
    import threading

    from cowork.services import project_file_copy

    (org_conversation.project_path / "chapter-1.md").write_text("The porter kept the keys.")
    conversation = org_conversation.project_path / "conversations" / str(org_conversation.id)
    record = project_file_copy.copy_record_path(org_conversation.scope, org_conversation.id)
    real_stage = project_file_copy.stage_project_files
    started, deleted = threading.Event(), threading.Event()
    made_again: list[bool] = []

    def copy_after_the_delete(*args, **kwargs):
        started.set()
        assert deleted.wait(timeout=30)
        staged = real_stage(*args, **kwargs)
        made_again.append(conversation.exists() and record.exists())
        return staged

    monkeypatch.setattr(project_file_copy, "stage_project_files", copy_after_the_delete)
    turn = asyncio.create_task(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))
    assert await asyncio.to_thread(started.wait, 30)
    turn.cancel()
    await asyncio.to_thread(_delete_conversation, org_conversation)
    deleted.set()

    with pytest.raises(asyncio.CancelledError):
        await turn

    assert made_again == [True]
    assert not conversation.exists()
    assert not record.exists()


def test_a_conversation_staging_cant_read_refuses_the_turn(org_conversation, monkeypatch):
    """Without the conversation, its project's folder is unknown, so the copy
    an earlier turn left can be neither checked nor emptied: the turn is
    refused rather than run on it."""
    (org_conversation.project_path / "chapter-1.md").write_text("The porter kept the keys.")
    asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))

    class Unreadable:
        def __init__(self, session):
            pass

        def get_conversation(self, conv_id):
            raise RuntimeError("the conversation read failed")

    monkeypatch.setattr(responses_mod, "ConversationService", Unreadable)
    staged: list = []
    monkeypatch.setattr("cowork.services.project_file_copy.stage_project_files", lambda *a, **k: staged.append(a))

    with pytest.raises(RuntimeError, match="conversation read failed"):
        asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))

    assert staged == []


def test_a_staging_unit_whose_commit_fails_still_copies_the_projects_files(
    org_conversation, monkeypatch, caplog,
):
    """The project's folder was read before the commit failed, so the copy
    still runs and the turn goes on with the project's files as they are."""
    from contextlib import contextmanager

    from cowork.db import units

    project_file = org_conversation.project_path / "chapter-1.md"
    project_file.write_text("first")
    asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))
    project_file.unlink()
    (org_conversation.project_path / "chapter-2.md").write_text("second")
    real_unit_session = units.unit_session

    @contextmanager
    def commit_fails(*, scope):
        with real_unit_session(scope=scope) as session:
            yield session
        raise RuntimeError("the unit's commit failed")

    monkeypatch.setattr(units, "unit_session", commit_fails)

    with caplog.at_level(logging.WARNING, logger=responses_mod.logger.name):
        asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))

    assert sorted(path.name for path in copy_folder(org_conversation).iterdir()) == ["chapter-2.md"]
    assert any(r.getMessage() == "[responses] failed to stage workspace files" for r in caplog.records)


def test_a_busy_file_store_refuses_a_remote_turns_project_file_copy(org_conversation, monkeypatch):
    """The copy waits for its own copy slot and file store slot after the
    staging unit (run_efs_copy); a busy store refuses the turn there too."""
    from cowork.services import project_file_copy

    async def busy(call, **_kwargs):
        if getattr(call, "func", None) is project_file_copy.stage_project_files:
            raise FilesystemBusy()
        raise AssertionError(f"unexpected file store call {call!r}")

    monkeypatch.setattr(responses_mod, "run_efs_copy", busy)

    with pytest.raises(FilesystemBusy):
        asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))


async def test_a_turns_copy_waits_for_a_copy_slot_and_full_slots_refuse_it(
    org_conversation, monkeypatch, one_second_wait,
):
    """The copy takes one of the event loop's copy slots before its file
    store slot (run_efs_copy). With every copy slot held by other turns'
    copies, this turn's copy is refused after POOL_TIMEOUT with
    FilesystemBusy and copies nothing, while the file store keeps its slots
    for everyone else's work."""
    import threading

    from cowork.db.units import run_efs_copy

    (org_conversation.project_path / "chapter-1.md").write_text("The porter kept the keys.")
    monkeypatch.setenv("COWORK_TURN_PROJECT_FILES_COPY_SLOTS", "1")
    release = threading.Event()
    started = threading.Semaphore(0)

    def another_turns_copy():
        started.release()
        release.wait(timeout=30)

    other = TenantScope(org_mode=True, org_id=str(uuid4()), user_id=str(uuid4()))
    held = asyncio.create_task(run_efs_copy(another_turns_copy, scope=other))
    await asyncio.to_thread(started.acquire)
    try:
        with pytest.raises(FilesystemBusy):
            await stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope)
    finally:
        release.set()
        await held

    assert not (copy_folder(org_conversation) / "chapter-1.md").exists()


def test_a_project_outside_the_organizations_projects_copies_nothing_and_its_turn_runs(
    org_conversation, tmp_path, caplog,
):
    """A project seeded before projects were kept per organization can keep
    its old folder (ProjectService._repoint_if_stale). Its turn runs, as it
    did before the copy existed: nothing is copied into the old folder and
    nothing is refused, and the one WARNING line of the turn says the copy
    was skipped."""
    import sqlalchemy as sa

    from cowork.db.session import get_open_session
    from cowork.models.project import Project

    legacy = tmp_path / "projects" / "general"
    legacy.mkdir(parents=True)
    (legacy / "chapter-1.md").write_text("The porter kept the keys.")
    session = get_open_session()
    try:
        session.execute(
            sa.update(Project).where(Project.path == str(org_conversation.project_path)).values(path=str(legacy))
        )
        session.commit()
    finally:
        session.close()

    with caplog.at_level(logging.WARNING):
        asyncio.run(stage_remote_workspace(conv_id=org_conversation.id, scope=org_conversation.scope))

    assert not (legacy / "conversations" / str(org_conversation.id) / "project-files").exists()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert [(r.name, r.getMessage(), getattr(r, "outcome", None)) for r in warnings] == [(
        responses_mod.logger.name,
        "Did not copy the project's files for a cloud turn: the project's folder is outside the "
        "organization's projects",
        "skipped",
    )]


def test_a_turn_gets_only_its_own_organizations_project_files(org_deployment):
    """Two organizations each hold a project of the same name with different
    files. Each one's turn gets its own organization's files, in its own
    organization's tree, and its copy's record sits in that organization's
    store."""
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.services import project_file_copy

    name = f"research-{uuid4().hex[:8]}"
    turns = {}
    for label in ("a", "b"):
        principal = Principal(user_id=str(uuid4()), org_id=str(uuid4()))
        turns[label] = (principal, make_org_conversation(principal, name))
        (turns[label][1].project_path / "notes.md").write_text(f"organization {label}'s notes")

    for _principal, conversation in turns.values():
        asyncio.run(stage_remote_workspace(conv_id=conversation.id, scope=conversation.scope))

    shared = Path(get_app_settings().storage.shared_root)
    for label, (principal, conversation) in turns.items():
        own = shared / principal.org_id
        copy = copy_folder(conversation)
        assert own in copy.parents
        assert sorted(path.name for path in copy.iterdir()) == ["notes.md"]
        assert (copy / "notes.md").read_text() == f"organization {label}'s notes"
        record = project_file_copy.copy_record_path(conversation.scope, conversation.id)
        assert own in record.parents and record.is_file()
        assert sorted(project_file_copy.CopyRecord.model_validate_json(record.read_bytes()).files) == ["notes.md"]


def test_a_remote_turns_memory_save_runs_on_the_file_store_threads(org_conversation):
    """The real memory save, apply_turn_memory included: the entry lands in
    the member's memory store, and the guard finds no file store work off
    run_efs."""
    text = f"The team reads charts, not tables ({uuid4().hex[:8]})."

    with blocking_guard.watching() as found:
        asyncio.run(save_turn_memory(
            conv_id=org_conversation.id, entries=[{"text": text, "kind": "always"}],
            principal=PRINCIPAL, scope=org_conversation.scope, request_id="corr-1",
        ))

    assert sorted(found) == []
    root = GlobalMemoryStore(scope=org_conversation.scope).root
    assert any(text in path.read_text() for path in root.rglob("*") if path.is_file())


async def test_an_always_on_skill_is_read_on_the_file_store_threads_at_a_turns_start_and_in_its_pod_request(
    org_deployment, monkeypatch,
):
    """An organization whose store holds one skill shared with everyone and
    always on, with its row. The turn start reads it as handle() does: the
    first unit, a database unit for a turn whose project is named by id, sees
    the always-on row, and the SKILL.md that decides whether it reaches the
    turn is parsed in a unit of its own (_settle_always_on). The remote web
    turn's producer then reads the pod's seed, project memory included, and
    builds its skills block with the real remote_turn_skills, which reads the
    skill's folder. The guard finds no site the allowlist doesn't list, and
    the pod's request carries the skill always on.

    A new organization, whose store also holds a skill from before sharing
    with no row, so its first turn backfills that row (backfill_legacy_access),
    which lists the store, as an organization's first turn after the rollout
    does."""
    from functools import partial

    from anton.core.tools.skill_format import dump_skill

    from cowork.common.settings.user_settings import use_turn_settings
    from cowork.db.scoped import ScopedSession
    from cowork.db.session import get_open_session
    from cowork.db.units import run_db
    from cowork.models.skill import Skill
    from cowork.schemas.responses import ResponsesRequest
    from cowork.services.skill_access import access_rows, stage_access
    from cowork.services.skill_sharing import SkillAccessRule
    from cowork.services.skills import SkillService
    from tests._org_sharing import MEMBER_BEARER, mindshub_sharing_answers

    org_id, user_id = str(uuid4()), str(uuid4())
    principal = Principal(user_id=user_id, org_id=org_id)
    scope = scope_from_principal(principal)
    member = {"X-User-Id": user_id, "X-Organization-Id": org_id}
    client = TestClient(create_app())
    project = client.post("/api/v1/projects/", json={"name": "charts"}, headers=member)
    assert project.status_code == 201, project.text
    project_id = UUID(project.json()["id"])
    conversation = client.post(
        "/api/v1/conversations/", json={"title": "t", "projectId": str(project_id)}, headers=member,
    )
    assert conversation.status_code == 201, conversation.text
    conv_id = UUID(conversation.json()["id"])

    slug, legacy = "house-style", "legacy-notes"
    for name in (slug, legacy):
        folder = SkillService(scope).root / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(
            dump_skill(Skill(name=name, description=f"About {name}.", instructions="Answer in charts."))
        )
    session = get_open_session()
    try:
        stage_access(ScopedSession(session, scope), slug, SkillAccessRule(org_allowed=True, always_on=True))
        session.commit()
    finally:
        session.close()
    mindshub_sharing_answers(monkeypatch, teams=())
    sent: list[dict] = []

    async def pod(**kwargs):
        sent.append(kwargs)
        yield "turn_delta", {"text": "Here is the chart."}
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod)
    handler = ResponsesHandler(principal=principal, mindshub_bearer=MEMBER_BEARER)
    request = ResponsesRequest(input="What is 12 times 7?", conversation=str(conv_id), project_id=project_id)
    buffer = _Frames()

    with blocking_guard.watching() as found:
        await handler._resolve_skill_viewer(scope)
        start = await run_db(
            partial(handler._read_turn_start, request=request, has_disabled_connections=False), scope=scope,
        )
        assert start.always_on_unread
        start = await handler._settle_always_on(start)
        with use_turn_settings(scope, start.settings):
            await handler._produce_remote(
                conv_id=conv_id, input_text="What is 12 times 7?", original_content="What is 12 times 7?",
                model="anton", harness_id="anton", buffer=buffer,
            )

    assert blocking_guard.unlisted(found) == []
    assert (start.always_on_skills, start.gate_rows) == (True, [])
    assert buffer.closed == ["completed"]
    [job] = sent
    assert job["skills"][slug].always_on
    assert job["skills"][slug].files["SKILL.md"].startswith("---\n")
    assert not job["skills"][legacy].always_on
    session = get_open_session()
    try:
        backfilled = access_rows(ScopedSession(session, scope))[legacy]
        assert (backfilled.org_allowed, backfilled.always_on) == (True, False)
    finally:
        session.close()


class _Frames:
    """A turn's buffer that keeps only how it closed."""

    def __init__(self) -> None:
        self.closed: list[str] = []
        self.is_closed = False

    async def append(self, kind, record) -> None:
        pass

    async def close(self, reason) -> None:
        self.closed.append(reason)
        self.is_closed = True


async def test_a_busy_file_store_refuses_a_remote_turns_staging(monkeypatch, one_second_wait):
    """With every file store slot held, staging is refused with
    FilesystemBusy, so the turn is refused rather than run without what the
    user sent, and the staging never runs."""
    staged: list[UUID] = []
    monkeypatch.setattr(
        ResponsesHandler, "_stage_remote_workspace_files",
        staticmethod(lambda session, conv_id: staged.append(conv_id)),
    )

    async with holding(db_session.settings.storage.efs_slots, LOCAL_SCOPE):
        with pytest.raises(FilesystemBusy):
            await stage_remote_workspace(
                conv_id=uuid4(), scope=TenantScope(org_mode=True, org_id=ORG, user_id=USER),
            )

    assert staged == []


async def test_a_busy_file_store_drops_a_memory_save_with_a_warning(monkeypatch, one_second_wait, caplog):
    """With every file store slot held, the memory save is dropped with one
    WARNING naming the conversation and the request, and raises nothing, so
    the turn goes on to its reply."""
    applied: list[list[dict]] = []
    monkeypatch.setattr(
        ResponsesHandler, "_persist_turn_memory",
        staticmethod(lambda session, conv_id, entries, principal: applied.append(entries)),
    )
    conv_id = uuid4()

    with caplog.at_level(logging.WARNING, logger=responses_mod.logger.name):
        async with holding(db_session.settings.storage.efs_slots, LOCAL_SCOPE):
            await save_turn_memory(
                conv_id=conv_id, entries=[{"text": "remember", "kind": "always"}], principal=PRINCIPAL,
                scope=TenantScope(org_mode=True, org_id=ORG, user_id=USER), request_id="corr-1",
            )

    assert applied == []
    [record] = [r for r in caplog.records if r.name == responses_mod.logger.name]
    assert (record.levelname, record.getMessage()) == ("WARNING", MEMORY_WARNING)
    assert (record.conversation_id, record.request_id) == (str(conv_id), "corr-1")
