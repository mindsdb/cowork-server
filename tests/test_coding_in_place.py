"""An in-place task works in the user's own repository instead of a copy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from coding_service_fakes import (
    CREDS,
    FakeEngine,
    git,
    repository,
    service_with,
    wait_for_status,
    wait_for_workspace,
)

from cowork.coding.contracts import (
    PermissionMode,
    SessionCreateRequest,
    SessionStatus,
    WorkspaceKind,
    WorkspaceMode,
)
from cowork.coding.engines.codex_config import sandbox_policy
from cowork.coding.project_models import ProjectCommand, ProjectCreateRequest, ProjectFolder, RepositoryResource
from cowork.coding.repository_setup_models import TaskRepositorySetup
from cowork.coding.service import CodingService
from cowork.coding.workspace import WorkspaceError
from cowork.coding.workspace_mode import resolve_workspace_mode, supported_workspace_modes

pytestmark = pytest.mark.skipif(
    sys.platform not in {"darwin", "linux"}, reason="in-place tasks are offered on macOS and Linux only",
)


def in_place(service: CodingService, path: Path, prompt: str = "Change it"):
    return service.create_session(
        SessionCreateRequest(path=str(path), prompt=prompt, workspace_mode=WorkspaceMode.in_place),
        CREDS,
        "fake",
        "fake-model",
    )


def head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_macos_and_linux_offer_in_place_for_a_folder_task(platform: str) -> None:
    modes = supported_workspace_modes(runs_locally=True, org_mode=False, platform=platform)
    assert WorkspaceMode.in_place in modes
    assert resolve_workspace_mode(None, None, runs_locally=True, org_mode=False, platform=platform) == WorkspaceMode.worktree


def test_windows_keeps_worktrees_until_its_sandbox_is_verified() -> None:
    assert supported_workspace_modes(runs_locally=True, org_mode=False, platform="win32") == (WorkspaceMode.worktree,)
    with pytest.raises(ValueError, match="not available on this platform"):
        resolve_workspace_mode(WorkspaceMode.in_place, None, runs_locally=True, org_mode=False, platform="win32")


def test_a_project_default_applies_here_and_falls_back_on_another_computer() -> None:
    assert resolve_workspace_mode(
        None, WorkspaceMode.in_place, runs_locally=True, org_mode=False, platform="darwin",
    ) == WorkspaceMode.in_place
    assert resolve_workspace_mode(
        None, WorkspaceMode.in_place, runs_locally=False, org_mode=False, platform="darwin",
    ) == WorkspaceMode.worktree


def test_an_in_place_task_works_in_the_repository_and_reviews_against_its_start(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)

    session = wait_for_workspace(service, in_place(service, repo).id)

    assert session.workspace_mode == WorkspaceMode.in_place
    assert session.workspace_kind == WorkspaceKind.direct_folder
    assert Path(session.workspace_path) == repo.resolve()
    assert session.base_revision == head(repo)
    assert engine.opened_workspaces[-1] == session.workspace_path
    worktrees = subprocess.run(
        ["git", "worktree", "list", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout
    assert worktrees.count("worktree ") == 1

    # What the agent writes lands in the user's folder and shows in review.
    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    (repo / "new.txt").write_text("new\n", encoding="utf-8")
    assert {item.path for item in service.diff(session.id)} == {"README.md", "new.txt"}


def test_the_sandbox_limits_writes_to_the_users_folder() -> None:
    policy = sandbox_policy(PermissionMode.supervised, Path("/work/repo"))
    assert policy["writableRoots"] == ["/work/repo"]


def test_a_subfolder_runs_in_its_repository(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    (repo / "src").mkdir()
    service = service_with(tmp_path, FakeEngine())

    session = wait_for_workspace(service, in_place(service, repo / "src").id)

    assert Path(session.workspace_path) == repo.resolve()


@pytest.mark.parametrize("kind", ["plain", "empty-repo"])
def test_a_folder_review_cannot_compare_against_is_refused_before_the_task_exists(tmp_path: Path, kind: str) -> None:
    folder = tmp_path / kind
    folder.mkdir()
    if kind == "empty-repo":
        git(folder, "init")
    service = service_with(tmp_path, FakeEngine())

    with pytest.raises(WorkspaceError, match="choose Worktree"):
        in_place(service, folder)

    assert service.list_sessions().items == []


def test_existing_changes_are_called_out_when_the_task_starts(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    (repo / "README.md").write_text("mine\n", encoding="utf-8")
    service = service_with(tmp_path, FakeEngine())

    session = wait_for_workspace(service, in_place(service, repo).id)

    assert session.source_dirty
    ready = [event for event in service.events(session.id).items if event.title == "Task workspace ready"]
    assert "already had uncommitted changes" in (ready[-1].text or "")


def test_only_one_in_place_task_runs_in_a_repository_at_a_time(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    engine.block_until_release = True
    service = service_with(tmp_path, engine)
    try:
        first = in_place(service, repo, "First")
        assert engine.started.wait(timeout=3)

        with pytest.raises(RuntimeError, match="choose Worktree"):
            in_place(service, repo / ".", "Second")
        # A worktree task on the same repository runs alongside it.
        isolated = service.create_session(
            SessionCreateRequest(path=str(repo), prompt="Isolated"), CREDS, "fake", "fake-model",
        )
        assert isolated.workspace_mode == WorkspaceMode.worktree
    finally:
        engine.release_events.set()
    wait_for_status(service, first.id, SessionStatus.completed)

    # Once the first finishes, another in-place task can start.
    engine.block_until_release = False
    second = wait_for_workspace(service, in_place(service, repo, "Second").id)
    assert second.workspace_kind == WorkspaceKind.direct_folder


def test_a_follow_up_waits_while_another_in_place_task_runs_in_the_repository(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    first = wait_for_status(service, in_place(service, repo, "First").id, SessionStatus.completed)
    engine.block_until_release = True
    try:
        second = in_place(service, repo, "Second")
        assert engine.started.wait(timeout=3)
        with pytest.raises(RuntimeError, match="Another task is working in this folder"):
            service.submit_turn(first.id, "Follow up", CREDS)
    finally:
        engine.release_events.set()
    wait_for_status(service, second.id, SessionStatus.completed)


def test_discarding_a_finished_task_is_refused_while_another_works_in_the_folder(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    first = wait_for_status(service, in_place(service, repo, "First").id, SessionStatus.completed)
    engine.block_until_release = True
    try:
        second = in_place(service, repo, "Second")
        assert engine.started.wait(timeout=3)
        # The second task edits the file the first one's review still lists.
        (repo / "README.md").write_text("second task's work\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="Another task is working in this folder"):
            service.review_file_action(first.id, None, "README.md", "discard")
        with pytest.raises(RuntimeError, match="Another task is working in this folder"):
            service.commit(first.id, "Commit someone else's work")
        assert (repo / "README.md").read_text(encoding="utf-8") == "second task's work\n"
    finally:
        engine.release_events.set()
    wait_for_status(service, second.id, SessionStatus.completed)


def test_a_turn_cannot_start_while_another_task_changes_the_folder(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    first = wait_for_status(service, in_place(service, repo, "First").id, SessionStatus.completed)
    second = wait_for_status(service, in_place(service, repo, "Second").id, SessionStatus.completed)

    with service._maintenance_session(first.id, "busy"):
        with pytest.raises(RuntimeError, match="Another task is changing files in this folder"):
            in_place(service, repo, "Third")
        with pytest.raises(RuntimeError, match="Another task is changing files in this folder"):
            service.submit_turn(second.id, "Follow up", CREDS)
        with pytest.raises(RuntimeError, match="Another task is changing files in this folder"):
            service.review_file_action(second.id, None, "README.md", "discard")

    # The reservation ends with the operation, even one that failed.
    with pytest.raises(ValueError):
        with service._maintenance_session(first.id, "busy"):
            raise ValueError("operation failed")
    wait_for_status(service, service.submit_turn(second.id, "Follow up", CREDS).id, SessionStatus.completed)


def test_a_repository_inside_another_is_coordinated_with_it(tmp_path: Path) -> None:
    outer = repository(tmp_path)
    inner_parent = tmp_path / "inner-parent"
    inner_parent.mkdir()
    inner = repository(inner_parent)
    nested = outer / "vendor" / "inner"
    nested.parent.mkdir()
    inner.rename(nested)
    service = service_with(tmp_path, FakeEngine())
    outer_task = wait_for_status(service, in_place(service, outer, "Outer").id, SessionStatus.completed)

    with service._maintenance_session(outer_task.id, "busy"):
        with pytest.raises(RuntimeError, match="Another task is changing files in this folder"):
            in_place(service, nested, "Inner")


def test_applying_a_worktree_task_waits_for_an_in_place_task_in_its_source(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    engine = FakeEngine()
    service = service_with(tmp_path, engine)
    isolated = wait_for_status(
        service,
        service.create_session(SessionCreateRequest(path=str(repo), prompt="Isolated"), CREDS, "fake", "fake-model").id,
        SessionStatus.completed,
    )
    (Path(isolated.workspace_path) / "README.md").write_text("isolated change\n", encoding="utf-8")
    engine.block_until_release = True
    try:
        active = in_place(service, repo, "In place")
        assert engine.started.wait(timeout=3)
        (repo / "README.md").write_text("in-place work\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="Another task is working in this folder"):
            service.apply_to_source(isolated.id)
        assert (repo / "README.md").read_text(encoding="utf-8") == "in-place work\n"
    finally:
        engine.release_events.set()
    wait_for_status(service, active.id, SessionStatus.completed)


def test_apply_explains_the_changes_are_already_in_the_folder(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    session = wait_for_status(service, in_place(service, repo).id, SessionStatus.completed)

    with pytest.raises(WorkspaceError, match="already there"):
        service.apply_to_source(session.id)


def test_deleting_an_in_place_task_never_touches_the_users_files(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    session = wait_for_status(service, in_place(service, repo).id, SessionStatus.completed)
    (repo / "README.md").write_text("agent edit\n", encoding="utf-8")
    (repo / "kept.txt").write_text("keep\n", encoding="utf-8")

    service.delete_session(session.id)

    assert (repo / "README.md").read_text(encoding="utf-8") == "agent edit\n"
    assert (repo / "kept.txt").exists()
    assert (repo / ".git").is_dir()


def test_forking_an_in_place_task_moves_its_changes_into_a_worktree(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    service = service_with(tmp_path, FakeEngine())
    parent = wait_for_status(service, in_place(service, repo).id, SessionStatus.completed)
    (repo / "README.md").write_text("in progress\n", encoding="utf-8")

    child = service.fork_session(parent.id, CREDS)

    assert child.workspace_mode == WorkspaceMode.worktree
    assert child.workspace_kind == WorkspaceKind.git_worktree
    assert Path(child.workspace_path) != repo.resolve()
    assert (Path(child.workspace_path) / "README.md").read_text(encoding="utf-8") == "in progress\n"
    # The parent keeps working in the user's folder.
    assert (repo / "README.md").read_text(encoding="utf-8") == "in progress\n"
    assert service.get_session(parent.id).workspace_kind == WorkspaceKind.direct_folder


def make_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    git(path, "init", "-b", "main")
    git(path, "config", "user.email", "cowork@example.invalid")
    git(path, "config", "user.name", "Cowork Test")
    (path / "README.md").write_text("base\n", encoding="utf-8")
    git(path, "add", ".")
    git(path, "commit", "-m", "base")
    return path


def two_repo_project(service: CodingService, tmp_path: Path, **options):
    app = make_repo(tmp_path / "app")
    api = make_repo(tmp_path / "api")
    project = service.projects.create(ProjectCreateRequest(
        name="Shop",
        folders=[
            ProjectFolder(
                id="app", name="App", path=str(app),
                commands=[ProjectCommand(id="install", label="Install", argv=["touch", "installed.txt"], phase="setup")],
            ),
            ProjectFolder(id="api", name="API", path=str(api)),
        ],
        default_engine_id="fake",
        default_model="fake-model",
        **options,
    ))
    return project, app, api


def start_project_task(service: CodingService, project_id: str, prompt: str = "Change both", **request):
    return service.create_session(
        SessionCreateRequest(project_id=project_id, prompt=prompt, workspace_mode=WorkspaceMode.in_place, **request),
        CREDS,
        "fake",
        "fake-model",
    )


def test_a_project_task_works_in_each_repository_on_its_current_branch(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    project, app, api = two_repo_project(service, tmp_path)
    git(api, "switch", "-c", "feature")

    session = wait_for_status(service, start_project_task(service, project.id).id, SessionStatus.completed)

    assert session.workspace_mode == WorkspaceMode.in_place
    by_folder = {item.folder_id: item for item in session.workspaces}
    assert Path(by_folder["app"].workspace_path) == app.resolve()
    assert Path(by_folder["api"].workspace_path) == api.resolve()
    assert {item.workspace_kind for item in session.workspaces} == {WorkspaceKind.direct_folder}
    # No task branch: each checkout stays on the branch the user had.
    assert by_folder["app"].task_branch is None and by_folder["app"].base_branch == "main"
    assert by_folder["api"].base_branch == "feature"
    assert by_folder["api"].base_revision == head(api)
    # The user's checkout is already set up, so setup commands do not run in it.
    assert not (app / "installed.txt").exists()
    assert "user's own checkouts" in session.developer_instructions
    assert "Never modify the user's source folders" not in session.developer_instructions

    (app / "README.md").write_text("app change\n", encoding="utf-8")
    (api / "new.py").write_text("print(1)\n", encoding="utf-8")
    assert {(item.folder_id, item.path) for item in service.diff(session.id)} == {("app", "README.md"), ("api", "new.py")}


def test_a_project_default_runs_its_tasks_in_place(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    project, app, _ = two_repo_project(service, tmp_path, default_workspace_mode=WorkspaceMode.in_place)

    created = service.create_session(
        SessionCreateRequest(project_id=project.id, prompt="Go"), CREDS, "fake", "fake-model",
    )

    session = wait_for_status(service, created.id, SessionStatus.completed)
    assert session.workspace_mode == WorkspaceMode.in_place
    assert Path(session.workspaces[0].workspace_path) == app.resolve()


def test_committing_an_in_place_project_task_commits_in_the_users_repositories(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    project, app, api = two_repo_project(service, tmp_path)
    session = wait_for_status(service, start_project_task(service, project.id).id, SessionStatus.completed)
    (app / "README.md").write_text("app change\n", encoding="utf-8")
    before_api = head(api)

    service.commit(session.id, "Update the app")

    log = subprocess.run(["git", "log", "-1", "--format=%s"], cwd=app, check=True, capture_output=True, text=True)
    assert log.stdout.strip() == "Update the app"
    assert head(api) == before_api
    with pytest.raises(WorkspaceError, match="already there"):
        service.apply_to_source(session.id)


def test_in_place_project_tasks_are_refused_where_they_cannot_work(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    project, _, _ = two_repo_project(service, tmp_path)
    with pytest.raises(ValueError, match="Branch choices"):
        start_project_task(service, project.id, repository_setup=TaskRepositorySetup(branch="feature/x"))

    plain = tmp_path / "notes"
    plain.mkdir()
    mixed = service.projects.create(ProjectCreateRequest(
        name="Mixed",
        folders=[ProjectFolder(id="notes", name="Notes", path=str(plain))],
        default_engine_id="fake",
        default_model="fake-model",
    ))
    with pytest.raises(WorkspaceError, match="Notes: Working in the original folder needs a Git repository"):
        start_project_task(service, mixed.id)

    remote = service.projects.create(ProjectCreateRequest(
        name="Remote",
        resources=[RepositoryResource(id="lib", name="lib", source_url="https://github.com/example/lib.git")],
        default_engine_id="fake",
        default_model="fake-model",
    ))
    with pytest.raises(WorkspaceError, match="lib has no checkout on this computer"):
        start_project_task(service, remote.id)
    assert service.list_sessions().items == []


def test_a_project_default_of_local_falls_back_where_it_cannot_apply(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    plain = tmp_path / "notes"
    plain.mkdir()
    (plain / "todo.md").write_text("todo\n", encoding="utf-8")
    notes = service.projects.create(ProjectCreateRequest(
        name="Notes",
        folders=[ProjectFolder(id="notes", name="Notes", path=str(plain))],
        default_engine_id="fake",
        default_model="fake-model",
        default_workspace_mode=WorkspaceMode.in_place,
    ))

    options = service.workspace_mode_options(notes.id, None)
    assert options.supported == [WorkspaceMode.worktree]
    assert options.default == WorkspaceMode.worktree
    assert "needs a Git repository" in (options.in_place_unavailable_reason or "")
    created = service.create_session(SessionCreateRequest(project_id=notes.id, prompt="Go"), CREDS, "fake", "fake-model")
    session = wait_for_status(service, created.id, SessionStatus.completed)
    assert session.workspace_mode == WorkspaceMode.worktree
    assert Path(session.workspaces[0].workspace_path) != plain.resolve()

    # A branch choice needs a worktree, so the default gives way to it too.
    shop, _, _ = two_repo_project(service, tmp_path, default_workspace_mode=WorkspaceMode.in_place)
    assert service.workspace_mode_options(shop.id, None).default == WorkspaceMode.in_place
    branched = service.create_session(
        SessionCreateRequest(project_id=shop.id, prompt="Go", repository_setup=TaskRepositorySetup(branch="feature/x")),
        CREDS,
        "fake",
        "fake-model",
    )
    assert wait_for_status(service, branched.id, SessionStatus.completed).workspace_mode == WorkspaceMode.worktree


def test_a_project_task_and_a_folder_task_cannot_both_work_in_one_repository(tmp_path: Path) -> None:
    engine = FakeEngine()
    engine.block_until_release = True
    service = service_with(tmp_path, engine)
    project, _, api = two_repo_project(service, tmp_path)
    try:
        first = start_project_task(service, project.id)
        assert engine.started.wait(timeout=3)
        with pytest.raises(RuntimeError, match="choose Worktree"):
            in_place(service, api)
    finally:
        engine.release_events.set()
    wait_for_status(service, first.id, SessionStatus.completed)


def test_forking_an_in_place_project_task_isolates_the_fork(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    project, app, _ = two_repo_project(service, tmp_path)
    parent = wait_for_status(service, start_project_task(service, project.id).id, SessionStatus.completed)
    (app / "README.md").write_text("in progress\n", encoding="utf-8")

    child = service.fork_session(parent.id, CREDS)

    assert child.workspace_mode == WorkspaceMode.worktree
    assert {item.workspace_kind for item in child.workspaces} == {WorkspaceKind.git_worktree}
    assert all(item.task_branch for item in child.workspaces)
    child_app = next(item for item in child.workspaces if item.folder_id == "app")
    assert (Path(child_app.workspace_path) / "README.md").read_text(encoding="utf-8") == "in progress\n"
    assert "Never modify the user's source folders" in child.developer_instructions
    assert "user's own checkouts" not in child.developer_instructions


def test_deleting_an_in_place_project_task_leaves_every_repository_alone(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeEngine())
    project, app, api = two_repo_project(service, tmp_path)
    session = wait_for_status(service, start_project_task(service, project.id).id, SessionStatus.completed)
    (app / "README.md").write_text("keep me\n", encoding="utf-8")

    service.delete_session(session.id)

    assert (app / "README.md").read_text(encoding="utf-8") == "keep me\n"
    assert (api / ".git").is_dir()
