"""The server chooses and records each task's workspace mode."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from coding_service_fakes import CREDS, FakeEngine, repository, service_with, wait_for_workspace

from cowork.coding.contracts import CodingSession, SessionCreateRequest, WorkspaceKind, WorkspaceMode
from cowork.coding.project_models import ProjectCreateRequest, ProjectFolder, ProjectUpdateRequest
from cowork.coding.workspace_mode import resolve_workspace_mode, supported_workspace_modes


@pytest.mark.parametrize(
    ("runs_locally", "org_mode"),
    [(True, False), (False, False), (True, True), (False, True)],
)
def test_every_computer_defaults_to_a_worktree(runs_locally: bool, org_mode: bool) -> None:
    assert resolve_workspace_mode(None, None, runs_locally=runs_locally, org_mode=org_mode) == WorkspaceMode.worktree


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


def test_older_clients_still_enable_folder_copies_with_the_previous_field_name() -> None:
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

    assert created.workspace_mode == WorkspaceMode.worktree
    prepared = wait_for_workspace(service, created.id)
    assert prepared.workspace_mode == WorkspaceMode.worktree
    assert prepared.workspace_kind == WorkspaceKind.git_worktree


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
        SessionCreateRequest(path=str(repo), prompt="Build it"), CREDS, "fake", "fake-model",
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
    assert folder.default == WorkspaceMode.worktree
    assert service.workspace_mode_options(project.id, None).supported == [WorkspaceMode.worktree]
    assert service.workspace_mode_options(None, "another-computer").supported == [WorkspaceMode.worktree]

    monkeypatch.setattr("cowork.coding.workspace_mode._IN_PLACE_PLATFORMS", frozenset())
    assert service.workspace_mode_options(None, None).supported == [WorkspaceMode.worktree]
