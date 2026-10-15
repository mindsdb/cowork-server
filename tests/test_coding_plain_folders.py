"""A folder without Git runs in place and is reviewed against how it started."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from coding_service_fakes import CREDS, FakeEngine, git, service_with, wait_for_status

from cowork.coding.contracts import SessionCreateRequest, SessionStatus, WorkspaceKind, WorkspaceMode
from cowork.coding.control_errors import StateConflict
from cowork.coding.project_models import ProjectCreateRequest, ProjectFolder
from cowork.coding.service import CodingService
from cowork.coding.workspace_mode import resolve_workspace_mode, supported_workspace_modes

pytestmark = pytest.mark.skipif(
    sys.platform not in {"darwin", "linux"}, reason="in-place tasks are offered on macOS and Linux only",
)


def plain_folder(tmp_path: Path, name: str = "notes") -> Path:
    folder = tmp_path / name
    folder.mkdir()
    (folder / "todo.md").write_text("one\ntwo\n", encoding="utf-8")
    (folder / "old.txt").write_text("remove me\n", encoding="utf-8")
    (folder / "photo.bin").write_bytes(b"\0" * 64)
    (folder / "node_modules" / "pkg").mkdir(parents=True)
    (folder / "node_modules" / "pkg" / "index.js").write_text("x\n", encoding="utf-8")
    return folder


def start(service: CodingService, folder: Path, prompt: str = "Tidy up", **request):
    return service.create_session(
        SessionCreateRequest(path=str(folder), prompt=prompt, allow_folder_copy=True, **request),
        CREDS,
        "fake",
        "fake-model",
    )


def bump(path: Path) -> None:
    """Move a file's modification time on, as an edit would."""
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 5_000_000_000))


def test_a_folder_without_git_offers_only_local_where_in_place_runs() -> None:
    assert supported_workspace_modes(runs_locally=True, org_mode=False, git_backed=False, platform="darwin") == (WorkspaceMode.in_place,)
    assert resolve_workspace_mode(None, None, runs_locally=True, org_mode=False, git_backed=False, platform="linux") == WorkspaceMode.in_place
    # Windows keeps copying until its sandbox is verified.
    assert resolve_workspace_mode(None, None, runs_locally=True, org_mode=False, git_backed=False, platform="win32") == WorkspaceMode.worktree
    with pytest.raises(ValueError, match="no Git repository to make a worktree from"):
        resolve_workspace_mode(WorkspaceMode.worktree, None, runs_locally=True, org_mode=False, git_backed=False, platform="darwin")


def test_the_composer_learns_a_plain_folder_runs_in_place(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    folder = plain_folder(tmp_path)
    holder = tmp_path / "repos"
    holder.mkdir()
    (holder / "app").mkdir()
    git(holder / "app", "init")
    git(holder / "app", "config", "user.email", "cowork@example.invalid")
    git(holder / "app", "config", "user.name", "Cowork Test")
    (holder / "app" / "a.txt").write_text("a\n", encoding="utf-8")
    git(holder / "app", "add", ".")
    git(holder / "app", "commit", "-m", "a")

    plain = service.workspace_mode_options(None, None, str(folder))
    assert plain.supported == [WorkspaceMode.in_place]
    assert plain.default == WorkspaceMode.in_place
    # A folder of repositories can still be isolated as worktrees.
    assert service.workspace_mode_options(None, None, str(holder)).supported == [WorkspaceMode.worktree, WorkspaceMode.in_place]


def test_a_plain_folder_task_works_in_the_folder_without_copying_it(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    folder = plain_folder(tmp_path)

    session = wait_for_status(service, start(service, folder).id, SessionStatus.completed)

    assert session.workspace_mode == WorkspaceMode.in_place
    assert session.workspace_kind == WorkspaceKind.direct_folder
    assert Path(session.workspace_path) == folder.resolve()
    assert session.base_revision is None
    # Nothing was copied: the managed workspaces folder holds no task folder.
    assert list(service.workspaces.worktrees_root.iterdir()) == []
    ready = [event for event in service.events(session.id).items if event.title == "Task workspace ready"]
    assert "no Git history here" in (ready[-1].text or "")


def test_review_shows_what_changed_since_the_task_started(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    folder = plain_folder(tmp_path)
    session = wait_for_status(service, start(service, folder).id, SessionStatus.completed)

    (folder / "todo.md").write_text("one\ntwo\nthree\n", encoding="utf-8")
    bump(folder / "todo.md")
    (folder / "old.txt").unlink()
    (folder / "new.py").write_text("print('hi')\n", encoding="utf-8")
    (folder / "photo.bin").write_bytes(b"\0" * 65)
    (folder / "node_modules" / "pkg" / "index.js").write_text("changed\n", encoding="utf-8")

    files = {item.path: item for item in service.diff(session.id)}

    assert set(files) == {"todo.md", "old.txt", "new.py", "photo.bin"}
    assert files["todo.md"].status == "M" and "+three" in files["todo.md"].patch
    assert files["old.txt"].status == "D" and "-remove me" in files["old.txt"].patch
    assert files["new.py"].status == "A" and files["new.py"].additions == 1
    assert files["photo.bin"].binary


def test_a_file_saved_unchanged_is_not_a_change(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    folder = plain_folder(tmp_path)
    session = wait_for_status(service, start(service, folder).id, SessionStatus.completed)

    bump(folder / "todo.md")

    assert service.diff(session.id) == []


def test_deleting_a_plain_folder_task_keeps_the_folder_and_drops_its_snapshot(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    folder = plain_folder(tmp_path)
    session = wait_for_status(service, start(service, folder).id, SessionStatus.completed)
    (folder / "todo.md").write_text("edited\n", encoding="utf-8")

    service.delete_session(session.id)

    assert (folder / "todo.md").read_text(encoding="utf-8") == "edited\n"
    assert (folder / "node_modules" / "pkg" / "index.js").exists()
    assert not service.workspaces.folder_baselines.exists(session.id)


def test_a_plain_folder_task_cannot_be_forked(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    folder = plain_folder(tmp_path)
    session = wait_for_status(service, start(service, folder).id, SessionStatus.completed)

    assert service.get_session(session.id).task_capabilities.fork is False
    with pytest.raises(StateConflict, match="cannot be forked"):
        service.fork_session(session.id, CREDS)


def test_an_in_place_task_in_a_containing_folder_blocks_one_inside_it(tmp_path: Path) -> None:
    engine = FakeEngine()
    engine.block_until_release = True
    service = service_with(tmp_path, engine)
    folder = plain_folder(tmp_path)
    (folder / "sub").mkdir()
    try:
        first = start(service, folder, "Outer")
        assert engine.started.wait(timeout=3)
        with pytest.raises(RuntimeError, match="Another task is working in this folder"):
            start(service, folder / "sub", "Inner")
    finally:
        engine.release_events.set()
    wait_for_status(service, first.id, SessionStatus.completed)


def test_an_empty_repository_is_reviewed_against_a_snapshot(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    folder = plain_folder(tmp_path, "fresh")
    git(folder, "init")

    session = wait_for_status(service, start(service, folder).id, SessionStatus.completed)
    (folder / "todo.md").write_text("one\n", encoding="utf-8")
    bump(folder / "todo.md")

    assert Path(session.workspace_path) == folder.resolve()
    assert [item.path for item in service.diff(session.id)] == ["todo.md"]


def test_windows_still_copies_a_plain_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cowork.coding.workspace_mode._IN_PLACE_PLATFORMS", frozenset())
    service = service_with(tmp_path, FakeEngine())
    folder = plain_folder(tmp_path)

    session = wait_for_status(service, start(service, folder).id, SessionStatus.completed)

    assert session.workspace_kind == WorkspaceKind.local_copy
    assert Path(session.workspace_path) != folder.resolve()


def test_a_project_folder_without_git_works_in_place_and_is_reviewed_per_folder(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    folder = plain_folder(tmp_path)
    project = service.projects.create(ProjectCreateRequest(
        name="Notes",
        folders=[ProjectFolder(id="notes", name="Notes", path=str(folder))],
        default_engine_id="fake",
        default_model="fake-model",
    ))
    created = service.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Tidy", workspace_mode=WorkspaceMode.in_place),
        CREDS,
        "fake",
        "fake-model",
    )
    session = wait_for_status(service, created.id, SessionStatus.completed)
    (folder / "new.md").write_text("hello\n", encoding="utf-8")

    assert Path(session.workspaces[0].workspace_path) == folder.resolve()
    assert [(item.folder_id, item.path) for item in service.diff(session.id)] == [("notes", "new.md")]
