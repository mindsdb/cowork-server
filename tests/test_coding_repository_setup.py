from pathlib import Path

import pytest
from pydantic import ValidationError

from coding_service_fakes import CREDS, FakeEngine, service_with
from test_coding_projects import git, repository
from cowork.coding.contracts import SessionCreateRequest
from cowork.coding.control_models import TaskResourceScope
from cowork.coding.project_models import (
    CodeProject,
    LocalFolderResource,
    ProjectCreateRequest,
    ProjectFolder,
    RepositoryResource,
)
from cowork.coding.project_workspaces import ProjectWorkspaceManager
from cowork.coding.repository_setup import RepositorySetupService, task_project
from cowork.coding.repository_setup_models import TaskRepositorySetup
from cowork.coding.workspace import WorkspaceError, WorkspaceManager


def project_for(*repos: Path) -> CodeProject:
    return CodeProject(
        id="project",
        name="Example",
        resources=[
            RepositoryResource(
                id=repo.name, name=repo.name, local_path=str(repo), computer_id="local"
            )
            for repo in repos
        ],
    )


def test_status_and_diff_leave_head_index_and_files_untouched(tmp_path):
    repo = repository(tmp_path, "app")
    git(repo, "branch", "staging")
    (repo / "README.md").write_text("staged\n")
    git(repo, "add", ".")
    (repo / "README.md").write_text("unstaged\n")
    (repo / "new file.txt").write_text("new\n")
    index = (repo / ".git/index").read_bytes()
    head = git(repo, "rev-parse", "HEAD")
    service = RepositorySetupService(WorkspaceManager(tmp_path / "runtime"), "local")
    status = service.status(project_for(repo))[0]
    assert status.local and status.available and status.change_count == 2
    assert "staging" in status.branches
    assert "new file.txt" in status.changes
    assert len(service.diff(project_for(repo), "app")) == 2
    assert (repo / ".git/index").read_bytes() == index
    assert git(repo, "rev-parse", "HEAD") == head
    assert (repo / "README.md").read_text() == "unstaged\n"


@pytest.mark.parametrize("include", [False, True])
def test_prepare_copies_changes_only_when_requested_without_staging_source(
    tmp_path, include
):
    repo = repository(tmp_path, "app")
    (repo / ".gitignore").write_text("ignored.txt\n")
    (repo / "remove.txt").write_text("remove me\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "files")
    (repo / "README.md").write_text("staged\n")
    git(repo, "add", ".")
    (repo / "README.md").write_text("unstaged\n")
    (repo / "new file.txt").write_text("untracked\n")
    (repo / "binary.bin").write_bytes(b"\x00\xff\x80")
    (repo / "ignored.txt").write_text("private\n")
    (repo / "remove.txt").unlink()
    index = (repo / ".git/index").read_bytes()
    head = git(repo, "rev-parse", "HEAD")
    manager = ProjectWorkspaceManager(WorkspaceManager(tmp_path / "runtime"))
    prepared = manager.prepare(
        "task",
        project_for(repo),
        TaskRepositorySetup(branch="feat/example", include_local_changes=include),
    )
    target = Path(prepared.primary.workspace_path)
    assert git(target, "branch", "--show-current") == "feat/example"
    assert (target / "README.md").read_text() == ("unstaged\n" if include else "app\n")
    assert (target / "new file.txt").exists() == include
    assert (target / "remove.txt").exists() != include
    if include:
        assert (target / "binary.bin").read_bytes() == b"\x00\xff\x80"
    assert not (target / "ignored.txt").exists()
    assert (repo / ".git/index").read_bytes() == index
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "branch", "--show-current") != "feat/example"


def test_selected_base_branch_and_subset_do_not_change_project_defaults(tmp_path):
    first = repository(tmp_path, "first")
    second = repository(tmp_path, "second")
    git(first, "branch", "staging")
    (first / "later.txt").write_text("later")
    git(first, "add", ".")
    git(first, "commit", "-m", "later")
    original = project_for(first, second)
    options = TaskRepositorySetup(base_branches={"first": "staging"}, branch="feat/new")
    selected = task_project(original, options, ["first"])
    selected.resources = [selected.resources[0]]
    prepared = ProjectWorkspaceManager(WorkspaceManager(tmp_path / "runtime")).prepare(
        "task", selected, options
    )
    assert not (Path(prepared.primary.workspace_path) / "later.txt").exists()
    assert original.resources[0].default_branch is None
    assert prepared.primary.base_branch == "staging"
    assert len(prepared.workspaces) == 1


@pytest.mark.parametrize(
    "branch", ["--orphan", "bad name", "../oops", "x..y", "x.lock"]
)
def test_invalid_branch_cleans_up_worktree(tmp_path, branch):
    repo = repository(tmp_path, "app")
    manager = ProjectWorkspaceManager(WorkspaceManager(tmp_path / "runtime"))
    with pytest.raises(WorkspaceError):
        manager.prepare("task", project_for(repo), TaskRepositorySetup(branch=branch))
    assert len(git(repo, "worktree", "list", "--porcelain").split("worktree ")) == 2


def test_branch_collision_in_second_repo_rolls_back_first_without_deleting_existing_branch(
    tmp_path,
):
    first, second = repository(tmp_path, "first"), repository(tmp_path, "second")
    git(second, "branch", "feat/existing")
    existing = git(second, "rev-parse", "feat/existing")
    manager = ProjectWorkspaceManager(WorkspaceManager(tmp_path / "runtime"))
    with pytest.raises(WorkspaceError):
        manager.prepare(
            "task",
            project_for(first, second),
            TaskRepositorySetup(branch="feat/existing"),
        )
    assert "feat/existing" not in git(first, "branch", "--list")
    assert git(second, "rev-parse", "feat/existing") == existing
    assert not list(manager.workspaces.worktrees_root.glob("task/*"))


def test_conflicting_local_changes_fail_without_touching_either_checkout(tmp_path):
    repo = repository(tmp_path, "app")
    git(repo, "branch", "old")
    (repo / "README.md").write_text("new commit\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "new")
    (repo / "README.md").write_text("my work\n")
    manager = WorkspaceManager(tmp_path / "runtime")
    index = (repo / ".git/index").read_bytes()
    with pytest.raises(WorkspaceError, match="conflict"):
        manager.prepare("task", str(repo), True, "old", include_local_changes=True)
    assert (repo / "README.md").read_text() == "my work\n"
    assert (repo / ".git/index").read_bytes() == index
    assert not (manager.worktrees_root / "task").exists()


def test_task_choices_cannot_escape_resource_scope(tmp_path):
    repo = repository(tmp_path, "app")
    with pytest.raises(WorkspaceError, match="included"):
        task_project(
            project_for(repo),
            TaskRepositorySetup(base_branches={"other": "main"}),
            ["app"],
        )
    with pytest.raises(ValidationError, match="Code Project"):
        SessionCreateRequest(
            path=str(repo), prompt="hello", repository_setup=TaskRepositorySetup()
        )


def test_unavailable_and_remote_repositories_are_truthful_and_never_inspect_foreign_paths(
    tmp_path,
):
    repo = repository(tmp_path, "app")
    project = project_for(repo)
    project.resources[0].computer_id = "another-computer"
    service = RepositorySetupService(WorkspaceManager(tmp_path / "runtime"), "local")
    assert not service.status(project)[0].local
    with pytest.raises(WorkspaceError, match="original computer"):
        service.diff(project, "app")
    project.resources[0].computer_id = "local"
    project.resources[0].local_path = str(tmp_path / "missing")
    assert not service.status(project)[0].available
    project.resources[0].source_url = "https://github.com/example/app.git"
    status = service.status(project)[0]
    assert status.available and not status.local
    assert status.detail == "Downloaded when the task starts"


def test_empty_repository_diff_reports_the_limitation_instead_of_claiming_no_changes(
    tmp_path,
):
    repo = tmp_path / "empty"
    repo.mkdir()
    git(repo, "init")
    (repo / "new.txt").write_text("untracked")
    service = RepositorySetupService(WorkspaceManager(tmp_path / "runtime"), "local")
    assert service.status(project_for(repo))[0].change_count == 1
    with pytest.raises(WorkspaceError, match="no commits"):
        service.diff(project_for(repo), repo.name)


def test_oversized_changes_fail_without_altering_source_or_leaking_worktree(
    tmp_path, monkeypatch
):
    repo = repository(tmp_path, "app")
    (repo / "large.txt").write_text("a large patch")
    manager = WorkspaceManager(tmp_path / "runtime")
    index = (repo / ".git/index").read_bytes()
    monkeypatch.setattr("cowork.coding.workspace.MAX_TOTAL_DIFF_BYTES", 8)
    with pytest.raises(WorkspaceError, match="too large"):
        manager.prepare("task", str(repo), True, include_local_changes=True)
    assert (repo / ".git/index").read_bytes() == index
    assert (repo / "large.txt").read_text() == "a large patch"
    assert not (manager.worktrees_root / "task").exists()


def test_include_local_changes_never_copies_untracked_repository_cache_files(tmp_path):
    remote = repository(tmp_path, "remote")
    project = CodeProject(
        id="p",
        name="Remote",
        resources=[
            RepositoryResource(id="remote", name="Remote", source_url=str(remote))
        ],
    )
    manager = ProjectWorkspaceManager(WorkspaceManager(tmp_path / "runtime"))
    first = manager.prepare("first", project)
    Path(first.primary.source_path, "cache-only.txt").write_text("internal")
    second = manager.prepare(
        "second", project, TaskRepositorySetup(include_local_changes=True)
    )
    assert not Path(second.primary.workspace_path, "cache-only.txt").exists()


def test_local_folders_are_copied_and_cannot_receive_branch_overrides(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "note.md").write_text("original")
    project = CodeProject(
        id="p",
        name="Notes",
        resources=[
            LocalFolderResource(
                id="notes", name="Notes", path=str(folder), computer_id="local"
            )
        ],
    )
    with pytest.raises(WorkspaceError, match="repositories"):
        task_project(
            project, TaskRepositorySetup(base_branches={"notes": "main"}), None
        )
    prepared = ProjectWorkspaceManager(WorkspaceManager(tmp_path / "runtime")).prepare(
        "task", project, TaskRepositorySetup()
    )
    (Path(prepared.primary.workspace_path) / "note.md").write_text("task")
    assert (folder / "note.md").read_text() == "original"


@pytest.mark.parametrize("origin", [None, "local-only-base", "shared-base"])
def test_real_session_factory_persists_choices_and_restores_named_workspace(tmp_path, origin):
    repo = repository(tmp_path, "app")
    if origin:
        remote = repository(tmp_path, "remote")
        git(repo, "remote", "add", "origin", str(remote))
        if origin == "shared-base":
            git(remote, "branch", "staging")
    git(repo, "branch", "staging")
    (repo / "README.md").write_text("staged work\n")
    git(repo, "add", ".")
    (repo / "README.md").write_text("unstaged work\n")
    (repo / "new.txt").write_text("user work")
    index = (repo / ".git/index").read_bytes()
    head = git(repo, "rev-parse", "HEAD")
    service = service_with(tmp_path, FakeEngine())
    project = service.projects.create(
        ProjectCreateRequest(
            name="Example",
            folders=[ProjectFolder(id="app", name="App", path=str(repo))],
        )
    )
    options = TaskRepositorySetup(
        branch="feat/session",
        base_branches={"app": "staging"},
        include_local_changes=True,
    )
    session = service.create_session(
        SessionCreateRequest(
            project_id=project.id,
            prompt="Read files",
            engine_id="fake",
            repository_setup=options,
        ),
        CREDS,
        "fake",
        "fake-model",
    )
    stored = service.control.store.get_task(session.id)
    assert stored.repository_setup == options
    assert stored.execution_project.resources[0].default_branch == "staging"
    assert stored.execution_project.resources[0].local_path == str(repo.resolve())
    assert stored.execution_project.resources[0].computer_id == service.control.local_computer.id
    assert service.projects.get(project.id).resources[0].default_branch is None
    assert service.projects.get(project.id).resources[0].computer_id == project.resources[0].computer_id
    if origin:
        assert project.resources[0].computer_id is None
    assert Path(session.workspace_path, "README.md").read_text() == "unstaged work\n"
    assert (repo / ".git/index").read_bytes() == index
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "branch", "--show-current") != "feat/session"
    assert Path(session.workspace_path, "new.txt").read_text() == "user work"
    assert session.workspaces[0].task_branch == "feat/session"
    records = service.control.store.list_workspaces(session.run_id)
    restored = service.project_workspaces.restore(
        session.id, stored.execution_project, records
    )
    assert restored.primary.task_branch == "feat/session"
    assert Path(restored.primary.workspace_path, "new.txt").read_text() == "user work"
    assert all(
        option.computer.id == session.computer_id
        for option in service.control.recovery_plan(session.run_id).options
    )


def test_task_setup_never_rebinds_a_checkout_owned_by_another_computer(tmp_path):
    project = project_for(repository(tmp_path, "app"))
    project.resources[0].computer_id = "other"
    project.resources[0].source_url = "https://github.com/example/app.git"
    service = service_with(tmp_path, FakeEngine())
    selected = task_project(
        project, TaskRepositorySetup(), None,
        local_computer_id=service.control.local_computer.id,
    )
    assert selected.resources[0].computer_id == "other"
    runtime = service.control.runtime_project(selected, TaskResourceScope(), service.control.local_computer.id)
    assert runtime.resources[0].local_path is None
    assert project.resources[0].local_path is not None


def test_remote_target_rejected_before_any_task_is_created(tmp_path):
    repo = repository(tmp_path, "app")
    service = service_with(tmp_path, FakeEngine())
    project = service.projects.create(
        ProjectCreateRequest(
            name="Example",
            folders=[ProjectFolder(id="app", name="App", path=str(repo))],
        )
    )
    with pytest.raises(ValueError, match="this computer"):
        service.create_session(
            SessionCreateRequest(
                project_id=project.id,
                computer_id="other",
                prompt="Read",
                repository_setup=TaskRepositorySetup(branch="feat/test"),
            ),
            CREDS,
            "codex",
            "gpt",
        )
    assert not service.store.list_sessions()


def test_remote_branch_selection_works_on_the_first_clone_and_next_task(tmp_path):
    remote = repository(tmp_path, "remote")
    git(remote, "branch", "staging")
    (remote / "later.txt").write_text("later")
    git(remote, "add", ".")
    git(remote, "commit", "-m", "later")
    project = CodeProject(
        id="p",
        name="Remote",
        resources=[
            RepositoryResource(id="remote", name="Remote", source_url=str(remote))
        ],
    )
    manager = WorkspaceManager(tmp_path / "runtime")
    inspection = RepositorySetupService(manager, "local")
    assert "staging" in inspection.branches(project, "remote")
    assert not inspection.status(project)[0].local
    options = TaskRepositorySetup(base_branches={"remote": "staging"})
    project = task_project(project, options, None)
    for number in range(2):
        prepared = ProjectWorkspaceManager(manager).prepare(
            f"task-{number}", project, options
        )
        assert not Path(prepared.primary.workspace_path, "later.txt").exists()
        assert prepared.primary.base_branch == "staging"


def test_empty_repository_never_silently_ignores_branch_or_exclude_choice(tmp_path):
    repo = tmp_path / "empty"
    repo.mkdir()
    git(repo, "init")
    (repo / "uncommitted.txt").write_text("new")
    manager = ProjectWorkspaceManager(WorkspaceManager(tmp_path / "runtime"))
    for options in (
        TaskRepositorySetup(),
        TaskRepositorySetup(branch="feat/new", include_local_changes=True),
    ):
        with pytest.raises(WorkspaceError, match="no commits"):
            manager.prepare("task", project_for(repo), options)
    prepared = manager.prepare(
        "task", project_for(repo), TaskRepositorySetup(include_local_changes=True)
    )
    assert Path(prepared.primary.workspace_path, "uncommitted.txt").read_text() == "new"


def test_failure_after_preparation_removes_only_the_new_unmodified_branch(
    tmp_path, monkeypatch
):
    repo = repository(tmp_path, "app")
    service = service_with(tmp_path, FakeEngine())
    project = service.projects.create(
        ProjectCreateRequest(
            name="Example",
            folders=[ProjectFolder(id="app", name="App", path=str(repo))],
        )
    )

    def fail(*args, **kwargs):
        raise RuntimeError("skill resolution failed")

    monkeypatch.setattr(service.session_factory.skills, "resolve", fail)
    with pytest.raises(RuntimeError, match="skill resolution failed"):
        service.create_session(
            SessionCreateRequest(
                project_id=project.id,
                prompt="Read",
                engine_id="fake",
                repository_setup=TaskRepositorySetup(branch="feat/retry"),
            ),
            CREDS,
            "fake",
            "fake-model",
        )
    assert "feat/retry" not in git(repo, "branch", "--list")
    assert len(git(repo, "worktree", "list", "--porcelain").split("worktree ")) == 2
