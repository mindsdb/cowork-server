"""The server chooses and records each task's workspace mode."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from coding_service_fakes import CREDS, FakeEngine, git, repository, service_with, wait_for_status, wait_for_workspace

from cowork.coding.contracts import CodingSession, SessionCreateRequest, SessionStatus, WorkspaceKind, WorkspaceMode
from cowork.coding.workspace import WorkspaceError
from cowork.coding.project_models import ProjectCreateRequest, ProjectFolder, ProjectUpdateRequest
from cowork.coding.workspace_mode import resolve_workspace_mode, supported_workspace_modes


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_a_task_here_works_in_the_folder_unless_it_asks_for_a_worktree(platform: str) -> None:
    assert resolve_workspace_mode(
        None, None, runs_locally=True, org_mode=False, platform=platform,
    ) == WorkspaceMode.in_place


@pytest.mark.parametrize(
    ("runs_locally", "org_mode", "platform"),
    [(False, False, "darwin"), (True, True, "darwin"), (False, True, "darwin"), (True, False, "win32")],
)
def test_where_in_place_is_unavailable_a_task_defaults_to_a_worktree(
    runs_locally: bool, org_mode: bool, platform: str,
) -> None:
    assert resolve_workspace_mode(
        None, None, runs_locally=runs_locally, org_mode=org_mode, platform=platform,
    ) == WorkspaceMode.worktree


def test_an_explicit_worktree_is_honoured_everywhere() -> None:
    for runs_locally, org_mode in [(True, False), (False, False), (True, True)]:
        assert resolve_workspace_mode(
            WorkspaceMode.worktree, WorkspaceMode.in_place, runs_locally=runs_locally, org_mode=org_mode,
        ) == WorkspaceMode.worktree


@pytest.mark.parametrize(("runs_locally", "org_mode"), [(False, False), (True, True)])
def test_an_in_place_request_is_refused_where_there_is_no_user_checkout(runs_locally: bool, org_mode: bool) -> None:
    assert WorkspaceMode.in_place not in supported_workspace_modes(runs_locally=runs_locally, org_mode=org_mode)
    with pytest.raises(ValueError, match="run in an isolated worktree"):
        resolve_workspace_mode(WorkspaceMode.in_place, None, runs_locally=runs_locally, org_mode=org_mode)


def test_a_project_default_the_computer_cannot_honour_falls_back_instead_of_failing() -> None:
    assert resolve_workspace_mode(
        None, WorkspaceMode.in_place, runs_locally=False, org_mode=False,
    ) == WorkspaceMode.worktree


def test_older_clients_copy_flag_still_validates_under_either_name() -> None:
    legacy = SessionCreateRequest.model_validate({"path": "/tmp/x", "prompt": "Go", "allow_direct_folder": True})
    current = SessionCreateRequest.model_validate({"path": "/tmp/x", "prompt": "Go", "allow_folder_copy": True})
    assert legacy.allow_folder_copy is True
    assert current.allow_folder_copy is True
    assert SessionCreateRequest(path="/tmp/x", prompt="Go").allow_folder_copy is False


def test_a_new_folder_task_records_its_resolved_mode(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())

    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Build it"), CREDS, "fake", "fake-model",
    )

    assert created.workspace_mode == WorkspaceMode.in_place
    prepared = wait_for_workspace(service, created.id)
    assert prepared.workspace_mode == WorkspaceMode.in_place
    assert prepared.workspace_kind == WorkspaceKind.direct_folder


def test_an_unsupported_in_place_task_is_refused_before_anything_is_prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cowork.coding.workspace_mode._IN_PLACE_PLATFORMS", frozenset())
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())

    with pytest.raises(ValueError, match="not available on this platform"):
        service.create_session(
            SessionCreateRequest(path=str(repo), prompt="Build it", workspace_mode=WorkspaceMode.in_place),
            CREDS,
            "fake",
            "fake-model",
        )

    assert service.list_sessions().items == []


def test_a_project_task_records_its_mode_and_the_project_default_can_be_cleared(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    project = service.projects.create(ProjectCreateRequest(
        name="Modes",
        folders=[ProjectFolder(id="app", name="App", path=str(repo))],
        default_engine_id="fake",
        default_model="fake-model",
        default_workspace_mode=WorkspaceMode.worktree,
    ))
    assert service.projects.get(project.id).default_workspace_mode == WorkspaceMode.worktree

    created = service.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Build it"), CREDS, "fake", "fake-model",
    )
    assert wait_for_workspace(service, created.id).workspace_mode == WorkspaceMode.worktree

    service.projects.update(project.id, ProjectUpdateRequest(default_workspace_mode=None))
    assert service.projects.get(project.id).default_workspace_mode is None


def test_a_task_saved_before_modes_existed_still_loads(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    created = service.create_session(
        SessionCreateRequest(path=str(repo), prompt="Build it", workspace_mode=WorkspaceMode.worktree),
        CREDS, "fake", "fake-model",
    )
    wait_for_workspace(service, created.id)
    raw = json.loads(service.store._meta_path(created.id).read_text(encoding="utf-8"))
    raw.pop("workspace_mode")

    legacy = CodingSession.model_validate(raw)

    assert legacy.workspace_mode is None
    assert legacy.workspace_kind == WorkspaceKind.git_worktree


def test_the_composer_learns_which_modes_a_new_task_can_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    project = service.projects.create(ProjectCreateRequest(
        name="Modes",
        folders=[ProjectFolder(id="app", name="App", path=str(repo))],
        default_engine_id="fake",
        default_model="fake-model",
    ))

    monkeypatch.setattr("cowork.coding.workspace_mode._IN_PLACE_PLATFORMS", frozenset({sys.platform}))
    folder = service.workspace_mode_options(None, None)
    assert folder.supported == [WorkspaceMode.worktree, WorkspaceMode.in_place]
    assert folder.default == WorkspaceMode.in_place
    assert service.workspace_mode_options(project.id, None).supported == [WorkspaceMode.worktree, WorkspaceMode.in_place]
    assert service.workspace_mode_options(None, "another-computer").supported == [WorkspaceMode.worktree]

    monkeypatch.setattr("cowork.coding.workspace_mode._IN_PLACE_PLATFORMS", frozenset())
    unavailable = service.workspace_mode_options(None, None)
    assert unavailable.supported == [WorkspaceMode.worktree]
    assert "not available on this platform" in (unavailable.in_place_unavailable_reason or "")


@pytest.mark.parametrize("kind", ["repository", "plain", "repositories", "empty repository"])
def test_the_modes_offered_for_a_folder_are_the_modes_creation_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    monkeypatch.setattr("cowork.coding.workspace_mode._IN_PLACE_PLATFORMS", frozenset({sys.platform}))
    if kind == "repository":
        folder = repository(tmp_path)
    else:
        folder = tmp_path / "folder"
        folder.mkdir()
        (folder / "notes.md").write_text("notes\n", encoding="utf-8")
        if kind == "repositories":
            (folder / "app").mkdir()
            repository(folder / "app")
        if kind == "empty repository":
            git(folder, "init")
    service = service_with(tmp_path, FakeEngine())

    options = service.workspace_mode_options(None, None, str(folder))

    # Every folder can be worked in, and that is the default.
    assert options.default == WorkspaceMode.in_place
    for mode in (WorkspaceMode.worktree, WorkspaceMode.in_place):
        request = SessionCreateRequest(path=str(folder), prompt="Go", workspace_mode=mode)
        if mode in options.supported:
            created = service.create_session(request, CREDS, "fake", "fake-model")
            wait_for_status(service, created.id, SessionStatus.completed)
        else:
            with pytest.raises((RuntimeError, ValueError, WorkspaceError), match="Worktree"):
                service.create_session(request, CREDS, "fake", "fake-model")
    assert (WorkspaceMode.worktree in options.supported) == (kind == "repository")
    assert WorkspaceMode.in_place in options.supported
    if kind in {"plain", "repositories"}:
        assert "Initialize Git" in (options.worktree_unavailable_reason or "")
        assert options.can_initialize_git is True
    elif kind == "empty repository":
        assert "first commit" in (options.worktree_unavailable_reason or "")
        assert options.can_initialize_git is False


def test_where_in_place_is_unavailable_a_folder_without_git_has_no_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cowork.coding.workspace_mode._IN_PLACE_PLATFORMS", frozenset())
    folder = tmp_path / "plain"
    folder.mkdir()
    service = service_with(tmp_path, FakeEngine())

    options = service.workspace_mode_options(None, None, str(folder))

    assert options.supported == []
    assert options.default is None
    assert options.can_initialize_git is True
    with pytest.raises(WorkspaceError, match="Initialize Git to use a Worktree"):
        service.create_session(SessionCreateRequest(path=str(folder), prompt="Go"), CREDS, "fake", "fake-model")


def test_initializing_git_offers_a_worktree_after_a_first_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cowork.coding.workspace_mode._IN_PLACE_PLATFORMS", frozenset({sys.platform}))
    folder = tmp_path / "plain"
    folder.mkdir()
    inner = repository(folder)
    (folder / "notes.md").write_text("notes\n", encoding="utf-8")
    service = service_with(tmp_path, FakeEngine())

    options = service.initialize_git(str(folder))

    assert (folder / ".git").is_dir()
    # Nothing is committed for the user, and the inner repository stays its own.
    assert options.can_initialize_git is False
    assert "first commit" in (options.worktree_unavailable_reason or "")
    assert (inner / ".git").is_dir()
    with pytest.raises(WorkspaceError, match="already in a Git repository"):
        service.initialize_git(str(folder))
    with pytest.raises(WorkspaceError, match="already in a Git repository"):
        service.initialize_git(str(inner))

    git(folder, "add", "notes.md")
    git(folder, "-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-m", "first")
    assert WorkspaceMode.worktree in service.workspace_mode_options(None, None, str(folder)).supported
