from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from cowork.coding.check_sandbox import (
    CheckPolicy, GitDirectories, check_environment, filter_storage, run_sandboxed, sandbox_environment, sandboxed_argv,
)
from cowork.coding.contracts import PermissionMode, TaskWorkspace, WorkspaceKind
from cowork.coding.project_models import CodeProject, ProjectCommand, ProjectEnvironment, ProjectFolder
from cowork.coding.project_workspaces import ProjectCommandRunner, ProjectWorkspaceManager
from cowork.coding.workspace import WorkspaceError, WorkspaceManager


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def repository(path: Path) -> Path:
    path.mkdir(parents=True)
    git(path, "init", "-q")
    git(path, "config", "user.email", "cowork@example.invalid")
    git(path, "config", "user.name", "Cowork Test")
    git(path, "config", "maintenance.auto", "false")
    (path / "README.md").write_text("base\n", encoding="utf-8")
    git(path, "add", ".")
    git(path, "commit", "-q", "-m", "base")
    return path


def hook(repo: Path, name: str, body: str) -> None:
    path = repo / ".git" / "hooks" / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_check_environment_inherits_only_selected_platform_variables(tmp_path: Path) -> None:
    environment = check_environment(
        {
            "PATH": "/usr/bin",
            "HOME": "/home/person",
            "JAVA_HOME": "/opt/java",
            "INTERNAL_AUTH_HEADER": "private",
            "COWORK_AUTH_TOKEN": "server-only",
            "GITHUB_CLIENT_SECRET": "bundled",
            "MINDSHUB_CODEX_API_KEY": "loopback",
            "NPM_TOKEN": "user",
            "DATABASE_URI": "sqlite:///cowork.db",
        },
        variables={"NPM_TOKEN": "chosen-for-this-project"},
        ports={"PORT": 4100},
        cache_root=tmp_path / "cache",
    )

    assert "INTERNAL_AUTH_HEADER" not in environment
    assert environment["PATH"] == "/usr/bin"
    assert environment["JAVA_HOME"] == "/opt/java"
    assert {"COWORK_AUTH_TOKEN", "GITHUB_CLIENT_SECRET", "MINDSHUB_CODEX_API_KEY", "DATABASE_URI"}.isdisjoint(environment)
    # A credential the project chose to pass is an explicit choice.
    assert environment["NPM_TOKEN"] == "chosen-for-this-project"
    assert environment["PORT"] == "4100"
    assert environment["UV_CACHE_DIR"] == str(tmp_path / "cache" / "uv")


def test_check_environment_keeps_proxies_only_without_credentials() -> None:
    environment = check_environment({
        "HTTPS_PROXY": "http://proxy.corp:3128",
        "HTTP_PROXY": "http://person:secret@proxy.corp:3128",
        "ALL_PROXY": "person@proxy.corp:1080",
        "NO_PROXY": "localhost",
        "NODE_EXTRA_CA_CERTS": "/etc/corp-ca.pem",
    })

    assert environment["HTTPS_PROXY"] == "http://proxy.corp:3128"
    assert "HTTP_PROXY" not in environment and "ALL_PROXY" not in environment
    assert environment["NO_PROXY"] == "localhost"
    assert environment["NODE_EXTRA_CA_CERTS"] == "/etc/corp-ca.pem"


@pytest.mark.skipif(os.name == "nt", reason="the hook under test is a POSIX shell script")
def test_host_git_runs_no_repository_hook_while_staging_or_reading_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Git runs post-index-change from ordinary staging and status commands,
    # and husky points the hook path into the task's own files.
    repo = repository(tmp_path / "repo")
    marker = tmp_path / "hook-ran"
    husky = repo / ".husky"
    husky.mkdir()
    for name in ("post-index-change", "pre-commit", "post-commit"):
        script = husky / name
        script.write_text(f"#!/bin/sh\necho {name} >> '{marker}'\n", encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
    git(repo, "-c", f"core.hooksPath={os.devnull}", "add", ".husky")
    git(repo, "-c", f"core.hooksPath={os.devnull}", "commit", "-q", "-m", "hooks")
    git(repo, "config", "core.hooksPath", ".husky")
    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    manager = WorkspaceManager(tmp_path / "root")

    manager.git_state(str(repo), str(repo))
    manager.commit(str(repo), "Change", manager.check_policy("project", PermissionMode.full_access, False))

    # Full access runs the supported pre-commit hook outside the sandbox, as
    # the agent itself would; Git must run none of the others.
    assert marker.read_text().split() == ["pre-commit"]
    assert git(repo, "log", "-1", "--format=%s") == "Change"


def test_full_access_checks_run_outside_the_sandbox(tmp_path: Path) -> None:
    policy = CheckPolicy.for_task(PermissionMode.full_access, network_access=False)

    assert sandboxed_argv(["npm", "test"], policy, [tmp_path]) == ["npm", "test"]


@pytest.mark.parametrize("network_access", [False, True])
@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_sandboxed_checks_follow_the_task_network_setting(
    tmp_path: Path, network_access: bool, platform: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    policy = CheckPolicy.for_task(PermissionMode.workspace, network_access=network_access)
    argv = sandboxed_argv(["npm", "test"], policy, [tmp_path], codex=Path("codex"))
    joined = " ".join(argv)

    assert argv[:2] == ["codex", "sandbox"]
    assert argv[-3:] == ["--", "npm", "test"]
    assert ("allow_local_binding=true" in joined) == (platform != "win32")
    uses_proxy = platform != "win32" and not network_access
    assert ("features.network_proxy=true" in argv) == uses_proxy
    assert ("domains={}" in joined) == uses_proxy
    assert ('windows.sandbox="elevated"' in argv) == (platform == "win32")
    windows_offline = platform == "win32" and not network_access
    assert ("permissions.checks.network={enabled=false}" in argv) == windows_offline


def test_windows_offline_checks_drop_proxy_firewall_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    environment = {
        "https_proxy": "http://localhost:3128", "Ws_Proxy": "http://localhost:1234",
        "CODEX_WINDOWS_SANDBOX_PROXY_PORTS": "5555", "CODEX_NETWORK_ALLOW_LOCAL_BINDING": "1", "PATH": "bin",
    }
    offline = CheckPolicy.for_task(PermissionMode.workspace, network_access=False)
    online = CheckPolicy.for_task(PermissionMode.workspace, network_access=True)
    assert sandbox_environment(environment, offline) == {"PATH": "bin"}
    assert sandbox_environment(environment, online) == environment


@pytest.mark.skipif(sys.platform != "win32", reason="exercise the native Windows sandbox")
def test_windows_native_checks_run_with_and_without_network_access(tmp_path: Path) -> None:
    # Network off is the default for a task, so a check has to start under it
    # and still be kept from other computers. The pinned sandbox does not keep
    # its offline identity from services on this computer, so localhost is not
    # asserted. The network-on run is the control: it proves the endpoint is
    # reachable from the sandbox, so a blocked offline run is the policy.
    workspace = tmp_path / "project"
    workspace.mkdir()
    # Outside every folder the policy grants, which includes the temp folder
    # pytest works in. This process can write there, so a refusal inside the
    # sandbox comes from the sandbox.
    outside = Path.home() / f".cowork-check-sandbox-{uuid.uuid4().hex}"
    outside.touch()
    outside.unlink()
    script = (
        "from pathlib import Path\n"
        "import socket, sys\n"
        "Path('ran.txt').write_text('ran')\n"
        "try:\n"
        "    Path(sys.argv[1]).touch()\n"
        "except OSError:\n"
        "    print('confined')\n"
        "else:\n"
        "    print('escaped')\n"
        "try:\n"
        "    socket.create_connection(('github.com', 443), timeout=10).close()\n"
        "except OSError:\n"
        "    print('blocked')\n"
        "else:\n"
        "    print('connected')\n"
    )
    try:
        for network_access, expected in [(True, "connected"), (False, "blocked")]:
            (workspace / "ran.txt").unlink(missing_ok=True)
            policy = CheckPolicy.for_task(PermissionMode.workspace, network_access, sandbox_home=tmp_path / "check-sandbox")
            result = run_sandboxed(
                [sys.executable, "-c", script, str(outside)],
                cwd=workspace, policy=policy, environment=check_environment(os.environ), writable=[workspace],
            )
            assert result.returncode == 0, result.stderr
            assert (workspace / "ran.txt").exists()
            assert result.stdout.split() == ["confined", expected]
            assert not outside.exists()
    finally:
        outside.unlink(missing_ok=True)


def test_hooks_may_stage_but_not_change_what_makes_git_run_programs(tmp_path: Path) -> None:
    repo = repository(tmp_path / "repo")
    worktree = tmp_path / "task"
    git(repo, "worktree", "add", "-q", "-b", "task", str(worktree))
    common = (repo / ".git").resolve()
    private = common / "worktrees" / "task"
    grants = GitDirectories(common=common, private=private).grants()

    assert grants[common] == "read"
    assert grants[private] == "write"
    assert grants[common / "objects"] == "read"
    if sys.platform.startswith("linux"):
        # Linux can only grant the folder; what is already in it stays read-only.
        assert grants[common / "refs"] == "write"
        assert grants[common / "refs" / "heads"] == "read"
    else:
        assert common / "refs" not in grants
        assert grants[common / "refs" / "stash"] == "write"
    for read_only in (private / "HEAD", private / "gitdir", private / "commondir"):
        assert grants[read_only] == "read"
    missing = {path.name for path, access in grants.items() if access == "read" and not path.exists()}
    if sys.platform.startswith("linux"):
        # The Linux sandbox would create a stand-in for a missing path.
        assert missing == set()
    else:
        # Elsewhere a missing control file is listed so nothing can create it.
        assert missing <= {"config", "config.worktree", "commondir", "gitdir", "hooks", "modules"}


def test_commit_without_checks_runs_no_repository_hook(tmp_path: Path) -> None:
    repo = repository(tmp_path / "repo")
    marker = tmp_path / "ran"
    hook(repo, "pre-commit", f"touch '{marker}'\n")
    (repo / "README.md").write_text("change\n", encoding="utf-8")

    WorkspaceManager(tmp_path / "root").commit(str(repo), "Change")

    assert not marker.exists()
    assert git(repo, "log", "-1", "--format=%s") == "Change"


requires_sandbox = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the Codex sandbox profile is verified on macOS; Linux and Windows need their own run",
)


@pytest.fixture
def checks(tmp_path: Path) -> CheckPolicy:
    return WorkspaceManager(tmp_path / "root").check_policy("project", PermissionMode.workspace, False)


@requires_sandbox
def test_project_checks_run_sandboxed_with_localhost_but_no_secrets_or_network(
    tmp_path: Path, checks: CheckPolicy, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = repository(tmp_path / "repo")
    monkeypatch.setenv("COWORK_TEST_SECRET", "server-only")
    script = (
        "import os, socket, urllib.request\n"
        "s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); print('listen ok')\n"
        "print('secret', os.environ.get('COWORK_TEST_SECRET'))\n"
        "try:\n"
        "    urllib.request.urlopen('https://example.com', timeout=5); print('network ok')\n"
        "except Exception:\n"
        "    print('network blocked')\n"
        "try:\n"
        "    open('.git/config', 'a').write('[core]\\n'); print('git config written')\n"
        "except OSError:\n"
        "    print('git config protected')\n"
        "open('build.txt', 'w').write('ok'); print('workspace write ok')\n"
    )
    project = CodeProject(
        id="project",
        name="Checks",
        folders=[ProjectFolder(
            id="app",
            name="App",
            path=str(repo),
            commands=[ProjectCommand(id="check", label="Check", phase="validate", argv=[sys.executable, "-c", script])],
        )],
    )
    workspace = TaskWorkspace(
        folder_id="app", folder_name="App", source_path=str(repo), workspace_path=str(repo),
        workspace_kind=WorkspaceKind.direct_folder,
    )

    [result] = ProjectCommandRunner().run(project, (workspace,), "validate", {}, policy=checks)

    assert result.return_code == 0, result.output
    assert "listen ok" in result.output
    assert "secret None" in result.output
    assert "network blocked" in result.output
    assert "git config protected" in result.output
    assert "workspace write ok" in result.output


@requires_sandbox
def test_read_only_validation_can_read_but_cannot_modify_project_files(tmp_path: Path) -> None:
    repo = repository(tmp_path / "repo")
    original = (repo / "README.md").read_text()
    script = (
        "from pathlib import Path\n"
        "source = Path('README.md')\n"
        "print(source.read_text())\n"
        "try: source.write_text('changed')\n"
        "except PermissionError: print('write denied')\n"
    )
    project = CodeProject(
        id="project", name="Checks",
        folders=[ProjectFolder(id="app", name="App", path=str(repo), commands=[
            ProjectCommand(id="check", label="Check", phase="validate", argv=[sys.executable, "-c", script]),
        ])],
    )
    workspace = TaskWorkspace(
        folder_id="app", folder_name="App", source_path=str(repo), workspace_path=str(repo),
        workspace_kind=WorkspaceKind.direct_folder,
    )
    policy = WorkspaceManager(tmp_path / "root").check_policy("project", PermissionMode.read_only, False)

    [result] = ProjectCommandRunner().run(project, (workspace,), "validate", {}, policy=policy)

    assert result.return_code == 0, result.output
    assert original.strip() in result.output
    assert "write denied" in result.output
    assert (repo / "README.md").read_text() == original


@requires_sandbox
def test_commit_hooks_run_sandboxed_and_may_restage_and_edit_the_message(
    tmp_path: Path, checks: CheckPolicy, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = repository(tmp_path / "repo")
    monkeypatch.setenv("COWORK_TEST_SECRET", "server-only")
    hook(repo, "pre-commit", (
        '[ -z "$COWORK_TEST_SECRET" ] || exit 3\n'
        "git config core.fsmonitor planted 2>/dev/null\n"
        "printf 'formatted\\n' > README.md && git add README.md\n"
        "git stash create >/dev/null\n"
    ))
    hook(repo, "commit-msg", 'printf "\\nChecked-by: hook\\n" >> "$1"\n')
    (repo / "README.md").write_text("unformatted\n", encoding="utf-8")

    WorkspaceManager(tmp_path / "root").commit(str(repo), "Change", checks)

    assert git(repo, "show", "HEAD:README.md") == "formatted"
    assert "Checked-by: hook" in git(repo, "log", "-1", "--format=%B")
    assert subprocess.run(["git", "config", "core.fsmonitor"], cwd=repo, capture_output=True).returncode != 0


@requires_sandbox
def test_staging_is_sandboxed_and_the_commit_is_recorded_on_the_host(
    tmp_path: Path, checks: CheckPolicy, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Staging can run repository filters, so it runs under the task's
    # permissions. Recording the commit reads no working tree, so it runs on
    # the host, where signing keeps the person's environment.
    repo = repository(tmp_path / "repo")
    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    calls = []
    real_run = subprocess.run

    def record(argv, **kwargs):
        calls.append(list(argv))
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", record)
    WorkspaceManager(tmp_path / "root").commit(str(repo), "Change", checks)

    [staging] = [argv for argv in calls if argv[-2:] == ["add", "--all"]]
    assert staging[1] == "sandbox"
    [status] = [argv for argv in calls if "status" in argv]
    assert status[1] == "sandbox"
    [recording] = [argv for argv in calls if "commit-tree" in argv]
    assert recording[0] == "git"
    assert not any(argv[0] == "git" and "commit" in argv for argv in calls)
    assert git(repo, "show", "HEAD:README.md") == "changed"


def test_a_read_only_task_cannot_commit(tmp_path: Path) -> None:
    repo = repository(tmp_path / "repo")
    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    manager = WorkspaceManager(tmp_path / "root")

    with pytest.raises(WorkspaceError, match="read-only"):
        manager.commit(str(repo), "Change", manager.check_policy("project", PermissionMode.read_only, False))

    assert git(repo, "log", "-1", "--format=%s") == "base"


def task_worktree(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = repository(tmp_path / "repo")
    worktree = tmp_path / "task"
    git(repo, "worktree", "add", "-q", "-b", "task", str(worktree))
    (worktree / "README.md").write_text("changed\n", encoding="utf-8")
    return repo, worktree, (repo / ".git" / "worktrees" / "task").resolve()


def plant_then(outcome):
    """A sandboxed step that writes a Git control file, then ends as given."""
    def run(argv, *, cwd, **_kwargs):
        private = Path(subprocess.run(["git", "rev-parse", "--absolute-git-dir"], cwd=cwd, capture_output=True, text=True).stdout.strip())
        (private / "config.worktree").write_text("[core]\n\tfsmonitor = planted\n", encoding="utf-8")
        if isinstance(outcome, BaseException):
            raise outcome
        return subprocess.CompletedProcess(argv, outcome, stdout="", stderr="")
    return run


@pytest.mark.parametrize("outcome", [0, 1, subprocess.TimeoutExpired(["hook"], 3)])
def test_what_a_step_plants_in_a_task_worktree_is_removed_however_it_ends(
    tmp_path: Path, checks: CheckPolicy, monkeypatch: pytest.MonkeyPatch, outcome,
) -> None:
    from cowork.coding import check_sandbox

    _repo, worktree, private = task_worktree(tmp_path)
    monkeypatch.setattr(check_sandbox, "run_sandboxed", plant_then(outcome))

    with pytest.raises(WorkspaceError):
        WorkspaceManager(tmp_path / "root").commit(
            str(worktree), "Change", checks, workspace_kind=WorkspaceKind.git_worktree,
        )

    assert not (private / "config.worktree").exists()
    assert git(worktree, "log", "-1", "--format=%s") == "base"


@pytest.mark.parametrize("outcome", [0, 1, subprocess.TimeoutExpired(["hook"], 3)])
def test_a_control_file_in_a_local_folder_is_reported_and_left_in_place(
    tmp_path: Path, checks: CheckPolicy, monkeypatch: pytest.MonkeyPatch, outcome,
) -> None:
    # A local folder's Git directory is the person's repository; another tool
    # may have written there at the same moment.
    from cowork.coding import check_sandbox

    repo = repository(tmp_path / "repo")
    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    monkeypatch.setattr(check_sandbox, "run_sandboxed", plant_then(outcome))
    manager = WorkspaceManager(tmp_path / "root")
    host_git = []
    real_run = manager.git.run
    monkeypatch.setattr(manager, "_keep_new_objects", lambda *_args: host_git.append("objects"))

    def record(cwd, *args, **kwargs):
        if (repo / ".git" / "config.worktree").exists():
            host_git.append(args)
        return real_run(cwd, *args, **kwargs)

    monkeypatch.setattr(manager.git, "run", record)

    with pytest.raises(WorkspaceError, match="Cowork left them in place"):
        manager.commit(str(repo), "Change", checks)

    # Host Git must not run against a repository holding a file Cowork can't vouch for.
    assert host_git == []
    assert (repo / ".git" / "config.worktree").exists()
    assert git(repo, "log", "-1", "--format=%s") == "base"


def test_local_task_preserves_control_file_in_existing_linked_worktree(
    tmp_path: Path, checks: CheckPolicy, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cowork.coding import check_sandbox

    _, worktree, private = task_worktree(tmp_path)
    manager = ProjectWorkspaceManager(WorkspaceManager(tmp_path / "coding"))
    project = CodeProject(
        id="local", name="Local", folders=[ProjectFolder(id="app", name="App", path=str(worktree))],
        environment=ProjectEnvironment(port_names=[]),
    )
    prepared = manager.prepare_in_place("local-task", project)
    assert prepared.primary.workspace_kind == WorkspaceKind.direct_folder
    control_file = private / "config.worktree"
    original_head = git(worktree, "rev-parse", "HEAD")

    def concurrent_edit(argv, **kwargs):
        if "status" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=" M README.md\n", stderr="")
        control_file.write_text("# user's configuration\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    host_run = manager.workspaces.git.run

    def require_safe_metadata(*args, **kwargs):
        assert not control_file.exists(), "host Git ran after the metadata check failed"
        return host_run(*args, **kwargs)

    monkeypatch.setattr(manager.workspaces.git, "run", require_safe_metadata)
    monkeypatch.setattr(check_sandbox, "run_sandboxed", concurrent_edit)
    with pytest.raises(WorkspaceError, match="Git files appeared"):
        manager.commit(list(prepared.workspaces), "Local change", checks)

    assert control_file.read_text(encoding="utf-8") == "# user's configuration\n"
    assert git(worktree, "rev-parse", "HEAD") == original_head


@requires_sandbox
@pytest.mark.parametrize("local_folder", [True, False])
def test_a_hook_cannot_create_git_control_files_on_macos(tmp_path: Path, checks: CheckPolicy, local_folder: bool) -> None:
    if local_folder:
        repo = folder = repository(tmp_path / "repo")
        (folder / "README.md").write_text("changed\n", encoding="utf-8")
        private = (repo / ".git").resolve()
    else:
        repo, folder, private = task_worktree(tmp_path)
    hook(repo, "pre-commit", f"printf '[core]\\n' > '{private}/config.worktree' && touch '{private}/commondir'\nexit 0\n")

    WorkspaceManager(tmp_path / "root").commit(str(folder), "Change", checks)

    assert not (private / "config.worktree").exists()
    assert not (private / "commondir").exists() or not local_folder
    assert git(folder, "log", "-1", "--format=%s") == "Change"


@requires_sandbox
def test_a_hook_may_stash_but_not_add_or_move_other_references(tmp_path: Path, checks: CheckPolicy) -> None:
    repo = repository(tmp_path / "repo")
    common = (repo / ".git").resolve()
    hook(repo, "pre-commit", (
        "git stash create >/dev/null\n"
        "git update-ref refs/heads/main HEAD~0 2>/dev/null && git update-ref refs/heads/moved HEAD 2>/dev/null\n"
        f"mkdir -p '{common}/refs/replace' 2>/dev/null && printf 'x' > '{common}/refs/replace/planted'\n"
        "exit 0\n"
    ))
    (repo / "README.md").write_text("changed\n", encoding="utf-8")

    WorkspaceManager(tmp_path / "root").commit(str(repo), "Change", checks)

    assert git(repo, "log", "-1", "--format=%s") == "Change"
    assert not (common / "refs" / "replace").exists()
    assert not (common / "refs" / "heads" / "moved").exists()


@requires_sandbox
def test_a_failing_commit_hook_stops_the_commit(tmp_path: Path, checks: CheckPolicy) -> None:
    repo = repository(tmp_path / "repo")
    hook(repo, "pre-commit", "echo 'lint failed: src/app.ts' >&2\nexit 1\n")
    (repo / "README.md").write_text("change\n", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="pre-commit hook failed:\nlint failed"):
        WorkspaceManager(tmp_path / "root").commit(str(repo), "Change", checks)

    assert git(repo, "log", "-1", "--format=%s") == "base"


@requires_sandbox
def test_commit_hooks_in_a_task_worktree_can_stage(tmp_path: Path, checks: CheckPolicy) -> None:
    repo = repository(tmp_path / "repo")
    worktree = tmp_path / "task"
    git(repo, "worktree", "add", "-q", "-b", "task", str(worktree))
    hook(repo, "pre-commit", "printf 'formatted\\n' > README.md && git add README.md\n")
    (worktree / "README.md").write_text("unformatted\n", encoding="utf-8")

    WorkspaceManager(tmp_path / "root").commit(str(worktree), "Change", checks)

    assert git(worktree, "show", "HEAD:README.md") == "formatted"
    assert os.access(repo / ".git" / "config", os.R_OK)


def test_an_unattributed_new_reference_is_reported_but_never_deleted(
    tmp_path: Path, checks: CheckPolicy, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On Linux a hook can create a refs folder, but so can a person working in
    # another worktree at the same moment, so Cowork stops without deleting.
    from cowork.coding import check_sandbox

    repo = repository(tmp_path / "repo")
    notes = (repo / ".git" / "refs" / "notes").resolve()

    def another_worktree_adds_a_note(*_args, **_kwargs):
        notes.mkdir()
        (notes / "commits").write_text("0" * 40 + "\n", encoding="utf-8")
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(check_sandbox, "run_sandboxed", another_worktree_adds_a_note)
    (repo / "README.md").write_text("changed\n", encoding="utf-8")

    with pytest.raises(WorkspaceError, match=r"Git files appeared .*notes"):
        WorkspaceManager(tmp_path / "root").commit(str(repo), "Change", checks)

    assert (notes / "commits").exists()
    assert git(repo, "log", "-1", "--format=%s") == "base"


def test_nothing_to_commit_is_reported(tmp_path: Path) -> None:
    repo = repository(tmp_path / "repo")

    with pytest.raises(WorkspaceError, match="no changes to commit"):
        WorkspaceManager(tmp_path / "root").commit(str(repo), "Nothing")


requires_ssh_signing = pytest.mark.skipif(
    os.name == "nt" or shutil.which("ssh-keygen") is None, reason="needs ssh-keygen for SSH commit signing",
)


def signing_repository(tmp_path: Path) -> Path:
    repo = repository(tmp_path / "repo")
    key = tmp_path / "signing-key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    git(repo, "config", "gpg.format", "ssh")
    git(repo, "config", "user.signingkey", str(key))
    git(repo, "config", "commit.gpgsign", "true")
    return repo


@requires_ssh_signing
def test_the_commit_is_signed_with_the_persons_own_signing_setup(tmp_path: Path) -> None:
    repo = signing_repository(tmp_path)
    (repo / "README.md").write_text("changed\n", encoding="utf-8")

    WorkspaceManager(tmp_path / "root").commit(str(repo), "Signed change\n\n  with a body  \n")

    raw = git(repo, "cat-file", "commit", "HEAD")
    assert "gpgsig -----BEGIN SSH SIGNATURE-----" in raw
    assert raw.endswith("Signed change\n\n  with a body") or raw.endswith("Signed change\n\n  with a body\n")
    assert git(repo, "show", "HEAD:README.md") == "changed"
    assert "commit: Signed change" in git(repo, "reflog", "-1")


@requires_ssh_signing
def test_cowork_will_not_sign_with_a_program_inside_the_task(tmp_path: Path) -> None:
    repo = signing_repository(tmp_path)
    marker = tmp_path / "signer-ran"
    signer = repo / "tools" / "sign.sh"
    signer.parent.mkdir()
    signer.write_text(f"#!/bin/sh\ntouch '{marker}'\nexec ssh-keygen \"$@\"\n", encoding="utf-8")
    signer.chmod(signer.stat().st_mode | stat.S_IXUSR)
    git(repo, "config", "gpg.ssh.program", "./tools/sign.sh")
    (repo / "README.md").write_text("changed\n", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="won't sign"):
        WorkspaceManager(tmp_path / "root").commit(str(repo), "Change")

    assert not marker.exists()
    assert git(repo, "log", "-1", "--format=%s") == "base"


@requires_sandbox
def test_staging_runs_a_repository_filter_under_the_task_permissions(
    tmp_path: Path, checks: CheckPolicy, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The filter is configured by the person, but its script is in the
    # checkout, where the agent can edit it.
    repo = repository(tmp_path / "repo")
    marker = tmp_path / "filter-environment"
    script = repo / "tools" / "clean.sh"
    script.parent.mkdir()
    script.write_text(f"#!/bin/sh\nenv > '{marker}'\ntr a-z A-Z\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    (repo / ".gitattributes").write_text("*.txt filter=upper\n", encoding="utf-8")
    git(repo, "config", "filter.upper.clean", "./tools/clean.sh")
    (repo / "note.txt").write_text("staged through the filter\n", encoding="utf-8")
    monkeypatch.setenv("COWORK_TEST_SECRET", "server-only")

    WorkspaceManager(tmp_path / "root").commit(str(repo), "Change", checks)

    assert git(repo, "show", "HEAD:note.txt") == "STAGED THROUGH THE FILTER"
    assert "COWORK_TEST_SECRET" not in marker.read_text(encoding="utf-8")


def test_only_commands_that_run_a_file_in_the_task_count_as_task_code(tmp_path: Path) -> None:
    from cowork.coding.workspace import names_task_file

    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    (root / "tools" / "clean.sh").write_text("", encoding="utf-8")
    (root / "clean.py").write_text("", encoding="utf-8")

    assert names_task_file("./tools/clean.sh", root)
    assert names_task_file("sh tools/clean.sh %f", root)
    assert names_task_file("python clean.py", root)
    assert names_task_file(str(root / "tools" / "clean.sh"), root)
    assert not names_task_file("git-lfs clean -- %f", root)
    assert not names_task_file("gpg", root)


@pytest.mark.skipif(os.name == "nt", reason="the filter under test is a POSIX shell script")
def test_host_status_never_runs_a_filter_script_from_the_task(tmp_path: Path) -> None:
    repo = repository(tmp_path / "repo")
    marker = tmp_path / "filter-ran"
    script = repo / "tools" / "clean.sh"
    script.parent.mkdir()
    script.write_text(f"#!/bin/sh\ntouch '{marker}'\ncat\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    (repo / ".gitattributes").write_text("*.txt filter=task diff=task\n", encoding="utf-8")
    git(repo, "config", "filter.task.clean", "./tools/clean.sh")
    git(repo, "config", "diff.task.textconv", "./tools/clean.sh")
    (repo / "note.txt").write_text("one\n", encoding="utf-8")
    git(repo, "-c", "filter.task.clean=", "add", ".")
    git(repo, "-c", "filter.task.clean=", "commit", "-q", "-m", "filtered")
    (repo / "note.txt").write_text("two\n", encoding="utf-8")
    manager = WorkspaceManager(tmp_path / "root")

    state = manager.git_state(str(repo), str(repo))
    manager.diff(str(repo), git(repo, "rev-parse", "HEAD~1"))

    assert state.dirty
    assert not marker.exists()


def test_an_unfinished_merge_is_left_for_git_to_finish(tmp_path: Path) -> None:
    repo = repository(tmp_path / "repo")
    git(repo, "switch", "-q", "-c", "side")
    (repo / "side.txt").write_text("side\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "side")
    git(repo, "switch", "-q", "-")
    git(repo, "merge", "--no-ff", "--no-commit", "side")
    head = git(repo, "rev-parse", "HEAD")

    with pytest.raises(WorkspaceError, match="merge is in progress"):
        WorkspaceManager(tmp_path / "root").commit(str(repo), "Merge side")

    assert git(repo, "rev-parse", "HEAD") == head
    assert (repo / ".git" / "MERGE_HEAD").exists()


@requires_ssh_signing
def test_the_signing_program_runs_outside_the_task_folder(tmp_path: Path) -> None:
    repo = signing_repository(tmp_path)
    marker = tmp_path / "signer-cwd"
    signer = tmp_path / "bin" / "sign"
    signer.parent.mkdir()
    signer.write_text(f"#!/bin/sh\npwd -P > '{marker}'\nexec ssh-keygen \"$@\"\n", encoding="utf-8")
    signer.chmod(signer.stat().st_mode | stat.S_IXUSR)
    git(repo, "config", "gpg.ssh.program", str(signer))
    (repo / "README.md").write_text("changed\n", encoding="utf-8")

    WorkspaceManager(tmp_path / "root").commit(str(repo), "Change")

    ran_in = Path(marker.read_text(encoding="utf-8").strip())
    assert repo.resolve() != ran_in and repo.resolve() not in ran_in.parents
    assert "gpgsig" in git(repo, "cat-file", "commit", "HEAD")


@pytest.mark.skipif(os.name == "nt", reason="the filter is a POSIX shell script")
def test_project_commit_preserves_filters_and_runs_both_status_checks_with_task_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = repository(tmp_path / "repo")
    manager = ProjectWorkspaceManager(WorkspaceManager(tmp_path / "coding"))
    project = CodeProject(id="project", name="Project", folders=[ProjectFolder(id="app", name="App", path=str(repo))])
    prepared = manager.prepare_in_place("task", project)
    checks = manager.check_policy("project", PermissionMode.full_access, False)
    script = repo / "normalize"
    script.write_text("#!/bin/sh\ntr a-z A-Z\n", encoding="utf-8")
    script.chmod(0o755)
    git(repo, "config", "filter.normalize.clean", "./normalize")
    (repo / ".gitattributes").write_text("*.txt filter=normalize\n", encoding="utf-8")
    (repo / "note.txt").write_text("lowercase\n", encoding="utf-8")
    monkeypatch.setenv("COWORK_TEST_SECRET", "server-only")
    calls = []
    real_run = subprocess.run

    def record(argv, **kwargs):
        if "status" in argv or argv[-2:] == ["add", "--all"]:
            calls.append("status" if "status" in argv else "add")
            assert "COWORK_TEST_SECRET" not in kwargs["env"]
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", record)
    manager.commit(list(prepared.workspaces), "Normalize", checks)
    assert calls == ["status", "add", "status"]
    assert git(repo, "show", "HEAD:note.txt") == "LOWERCASE"

    hook(repo, "pre-commit", "echo 'README.md needs formatting' >&2\nexit 1\n")
    (repo / "README.md").write_text("change\n", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="README.md needs formatting"):
        manager.commit(list(prepared.workspaces), "Rejected", checks)
    assert git(repo, "log", "-1", "--format=%s") == "Normalize"


def test_validation_in_plain_folder_without_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    workspace = TaskWorkspace(
        folder_id="app", folder_name="App", source_path=str(tmp_path), workspace_path=str(tmp_path),
        workspace_kind=WorkspaceKind.direct_folder,
    )
    command = ProjectCommand(id="check", label="Check", phase="validate", argv=[sys.executable, "-c", "print('ok')"])
    policy = CheckPolicy.for_task(PermissionMode.full_access, False)
    result = ProjectCommandRunner._run_one(command, workspace, {}, policy)
    assert result.return_code == 0, result.output


@requires_sandbox
def test_filtered_git_reads_use_read_only_sandbox_and_can_export_patch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = repository(tmp_path / "repo")
    git(repo, "config", "filter.normalize.clean", "tr a-z A-Z")
    (repo / ".gitattributes").write_text("README.md filter=normalize\n", encoding="utf-8")
    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    manager = WorkspaceManager(tmp_path / "manager")
    monkeypatch.setenv("COWORK_TEST_SECRET", "server-only")
    real_run = subprocess.run
    reads = []

    def record(argv, **kwargs):
        if "status" in argv or "diff" in argv:
            assert argv[1] == "sandbox"
            assert "COWORK_TEST_SECRET" not in kwargs["env"]
            reads.append(argv)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", record)
    assert manager.git_state(str(repo), str(repo)).dirty
    patch = tmp_path / "review.patch"
    manager.git.run(repo, "diff", f"--output={patch}", "HEAD", "--", "README.md", environment={"COWORK_TEST_SECRET": "private"})
    assert "+CHANGED" in patch.read_text()
    assert len(reads) == 2


def test_windows_filtered_git_reads_reuse_the_cowork_sandbox_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cowork.coding import check_sandbox
    from cowork.common.settings.app_settings import get_app_settings

    repo = repository(tmp_path / "repo")
    git(repo, "config", "filter.normalize.clean", "cat")
    manager = WorkspaceManager(tmp_path / "coding")
    monkeypatch.setattr(get_app_settings().coding, "root_dir", str(manager.root))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(check_sandbox, "sandbox_codex", lambda: Path("codex"))
    real_run = subprocess.run
    homes = []

    def record(argv, **kwargs):
        if argv[:2] == ["codex", "sandbox"]:
            homes.append(Path(kwargs["env"]["CODEX_HOME"]))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", record)
    for operation in ("status", "diff"):
        manager.git.run(repo, operation)
    home = manager.root / "check-sandbox"
    assert homes == [home, home]
    assert home.is_dir()


def stateful_filter_repository(tmp_path: Path) -> Path:
    """A repository whose installed filter stores content in the Git directory, as Git LFS does."""
    program = tmp_path / "bin" / "store-filter"
    program.parent.mkdir()
    program.write_text(
        "#!/bin/sh\n"
        'store="$(cd "$(git rev-parse --git-common-dir)" && pwd)/lfs/objects"\n'
        'mkdir -p "$store" && cat > "$store/tmp.$$" || exit 1\n'
        'oid=$(shasum -a 256 "$store/tmp.$$" | cut -d" " -f1)\n'
        'mv "$store/tmp.$$" "$store/$oid" && echo "pointer $oid"\n',
        encoding="utf-8",
    )
    program.chmod(program.stat().st_mode | stat.S_IXUSR)
    repo = repository(tmp_path / "repo")
    git(repo, "config", "filter.lfs.clean", str(program))
    git(repo, "config", "filter.lfs.smudge", "cat")
    git(repo, "config", "filter.lfs.required", "true")
    (repo / ".gitattributes").write_text("*.bin filter=lfs\n", encoding="utf-8")
    (repo / "asset.bin").write_text("one\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "asset")
    return repo


@requires_sandbox
def test_a_filter_that_stores_content_in_the_git_directory_still_commits_and_diffs(
    tmp_path: Path, checks: CheckPolicy,
) -> None:
    repo = stateful_filter_repository(tmp_path)
    worktree = tmp_path / "task"
    git(repo, "worktree", "add", "-q", "-b", "task", str(worktree))
    manager = WorkspaceManager(tmp_path / "root")

    (repo / "asset.bin").write_text("changed in the folder\n", encoding="utf-8")
    assert "asset.bin" in manager.git.run(repo, "diff", "--no-ext-diff", "HEAD").stdout
    folder = manager.commit(str(repo), "Folder", checks, workspace_kind=WorkspaceKind.direct_folder)

    (worktree / "asset.bin").write_text("changed in the task\n", encoding="utf-8")
    task = manager.commit(str(worktree), "Task", checks, workspace_kind=WorkspaceKind.git_worktree)

    assert not folder.dirty and git(repo, "log", "-1", "--format=%s") == "Folder"
    assert not task.dirty and git(worktree, "log", "-1", "--format=%s") == "Task"


@requires_sandbox
@pytest.mark.parametrize("kind", [WorkspaceKind.git_worktree, WorkspaceKind.direct_folder])
def test_a_filters_first_stored_file_commits_before_its_storage_folder_exists(
    tmp_path: Path, checks: CheckPolicy, kind: WorkspaceKind,
) -> None:
    repo = stateful_filter_repository(tmp_path)
    # As in a repository that tracks files with Git LFS but has stored none yet.
    shutil.rmtree(repo / ".git" / "lfs")
    target = repo
    if kind == WorkspaceKind.git_worktree:
        target = tmp_path / "task"
        git(repo, "worktree", "add", "-q", "-b", "task", str(target))
    manager = WorkspaceManager(tmp_path / "root")

    (target / "asset.bin").write_text("changed\n", encoding="utf-8")
    state = manager.commit(str(target), "Change", checks, workspace_kind=kind)

    assert not state.dirty and git(target, "log", "-1", "--format=%s") == "Change"
    assert any((repo / ".git" / "lfs" / "objects").iterdir())


def lfs_installed_repository(tmp_path: Path) -> tuple[Path, WorkspaceManager]:
    """A repository on a computer with Git LFS installed, which configures its filter everywhere."""
    repo = repository(tmp_path / "repo")
    git(repo, "config", "filter.lfs.clean", "git-lfs clean -- %f")
    git(repo, "config", "filter.lfs.process", "git-lfs filter-process")
    (repo / "asset.bin").write_text("content\n", encoding="utf-8")
    return repo, WorkspaceManager(tmp_path / "root")


def test_a_repository_that_tracks_nothing_with_git_lfs_gets_no_storage_folder(tmp_path: Path) -> None:
    repo, manager = lfs_installed_repository(tmp_path)
    # A comment that mentions the filter assigns it to nothing.
    (repo / ".gitattributes").write_text("# was: *.bin filter=lfs -text\n*.txt text\n", encoding="utf-8")

    assert git(repo, "check-attr", "filter", "asset.bin").endswith("unspecified")
    assert manager.git.filter_storage(repo, repo / ".git") == []
    assert not (repo / ".git" / "lfs").exists()


@pytest.mark.parametrize("source", ["nested", "outside"])
def test_git_lfs_storage_is_made_wherever_git_takes_the_rule_from(tmp_path: Path, source: str) -> None:
    repo, manager = lfs_installed_repository(tmp_path)
    if source == "nested":
        (repo / "assets").mkdir()
        (repo / "asset.bin").rename(repo / "assets" / "asset.bin")
        (repo / "assets" / ".gitattributes").write_text("*.bin filter=lfs -text\n", encoding="utf-8")
    else:
        # The person's own attributes file, outside the checkout.
        attributes = tmp_path / "attributes"
        attributes.write_text("*.bin filter=lfs -text\n", encoding="utf-8")
        git(repo, "config", "core.attributesFile", str(attributes))

    assert manager.git.filter_storage(repo, repo / ".git") == [repo / ".git" / "lfs"]


def test_a_recorded_commit_is_reported_even_when_status_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = repository(tmp_path / "repo")
    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    manager = WorkspaceManager(tmp_path / "root")

    def unreadable(*_args, **_kwargs):
        raise WorkspaceError("Could not check Git status")

    monkeypatch.setattr(manager, "_status_lines", unreadable)

    state = manager.commit(str(repo), "Change")

    assert state.revision == git(repo, "rev-parse", "HEAD")
    assert not state.status_checked
    assert git(repo, "log", "-1", "--format=%s") == "Change"


def test_redirected_filter_storage_is_never_granted(tmp_path: Path) -> None:
    repo = repository(tmp_path / "repo")
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / ".git" / "lfs").symlink_to(outside, target_is_directory=True)
    policy = CheckPolicy.for_task(PermissionMode.read_only, False)

    storage = filter_storage((repo / ".git").resolve())
    profile = " ".join(sandboxed_argv(["true"], policy, (), storage=storage, codex=Path("codex")))

    assert storage == []
    assert str(outside.resolve()) not in profile


def test_an_invalid_signing_setting_stops_the_commit(tmp_path: Path) -> None:
    repo = repository(tmp_path / "repo")
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "config", "commit.gpgSign", "maybe")
    (repo / "README.md").write_text("changed\n", encoding="utf-8")

    with pytest.raises(WorkspaceError, match="commit.gpgsign"):
        WorkspaceManager(tmp_path / "root").commit(str(repo), "Change")

    assert git(repo, "rev-parse", "HEAD") == head


@requires_sandbox
def test_a_required_filter_in_the_task_still_produces_a_patch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = repository(tmp_path / "repo")
    script = repo / "tools" / "clean.sh"
    script.parent.mkdir()
    script.write_text('#!/bin/sh\nprintf "%s" "${COWORK_TEST_SECRET:-hidden}" >&2\ntr a-z A-Z\n', encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    (repo / ".gitattributes").write_text("*.txt filter=upper\n", encoding="utf-8")
    git(repo, "config", "filter.upper.clean", "./tools/clean.sh")
    git(repo, "config", "filter.upper.required", "true")
    (repo / "note.txt").write_text("one\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "filtered")
    (repo / "note.txt").write_text("two\n", encoding="utf-8")
    monkeypatch.setenv("COWORK_TEST_SECRET", "server-only")

    result = WorkspaceManager(tmp_path / "root").git.run(repo, "diff", "--no-ext-diff", "HEAD")

    assert "+TWO" in result.stdout
    assert "server-only" not in result.stderr


@requires_sandbox
def test_a_hook_cannot_remove_history_other_checkouts_share(tmp_path: Path, checks: CheckPolicy) -> None:
    repo = repository(tmp_path / "repo")
    git(repo, "gc", "-q")
    worktree = tmp_path / "task"
    git(repo, "worktree", "add", "-q", "-b", "task", str(worktree))
    objects = repo / ".git" / "objects"
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(f"#!/bin/sh\nrm -rf '{objects}'/pack '{objects}'/?? 2>/dev/null\nexit 0\n", encoding="utf-8")
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR)
    before = sorted(path.name for path in (objects / "pack").iterdir())
    (worktree / "README.md").write_text("changed\n", encoding="utf-8")

    state = WorkspaceManager(tmp_path / "root").commit(
        str(worktree), "Change", checks, workspace_kind=WorkspaceKind.git_worktree,
    )

    assert sorted(path.name for path in (objects / "pack").iterdir()) == before
    assert not state.dirty and git(worktree, "log", "-1", "--format=%s") == "Change"
    git(repo, "fsck", "--no-dangling")


@requires_sandbox
def test_a_required_filter_in_the_task_can_stage_and_be_saved_for_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = repository(tmp_path / "repo")
    script = repo / "tools" / "clean.sh"
    script.parent.mkdir()
    script.write_text('#!/bin/sh\nprintf "%s" "${COWORK_TEST_SECRET:-hidden}" > "$(dirname "$0")/seen"\ntr a-z A-Z\n', encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    (repo / ".gitattributes").write_text("*.txt filter=upper\n", encoding="utf-8")
    git(repo, "config", "filter.upper.clean", "./tools/clean.sh")
    git(repo, "config", "filter.upper.required", "true")
    (repo / "note.txt").write_text("one\n", encoding="utf-8")
    monkeypatch.setenv("COWORK_TEST_SECRET", "server-only")
    manager = WorkspaceManager(tmp_path / "root")

    manager.review_file_action(str(repo), "note.txt", "stage")
    _index, files = manager._save_worktree_state(repo, git(repo, "rev-parse", "HEAD"))

    assert git(repo, "show", ":note.txt") == "ONE"
    assert git(repo, "show", f"{files}:note.txt") == "ONE"
    assert (script.parent / "seen").read_text(encoding="utf-8") == "hidden"


def test_an_edit_made_right_after_staging_is_still_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Same size, same second: Git can only tell by re-reading the file, which
    # it stops doing if recording the commit rewrites the index.
    repo = repository(tmp_path / "repo")
    (repo / "README.md").write_text("task\n", encoding="utf-8")
    manager = WorkspaceManager(tmp_path / "root")
    record = manager._record_commit

    def edit_then_record(root, message):
        (repo / "README.md").write_text("late\n", encoding="utf-8")
        return record(root, message)

    monkeypatch.setattr(manager, "_record_commit", edit_then_record)

    state = manager.commit(str(repo), "Change")

    assert git(repo, "show", "HEAD:README.md") == "task"
    assert state.dirty


def test_windows_hook_uses_git_interpreter_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cowork.coding import check_sandbox

    repo = repository(tmp_path / "repo")
    hooks = repo / "project hooks"
    hooks.mkdir()
    script = hooks / "commit-msg"
    script.write_text(
        '#!/usr/bin/env python\nimport sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text("checked")\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    message = tmp_path / "message with spaces"
    with monkeypatch.context() as platform:
        platform.setattr(sys, "platform", "win32")
        argv = check_sandbox.hook_argv(script, str(message))
    # Use Git's real interpreter dispatch, including the environment override
    # that normally prevents implicit hooks in a sandboxed Git step.
    env = {**os.environ, "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.hooksPath", "GIT_CONFIG_VALUE_0": os.devnull}
    subprocess.run(argv, cwd=repo, env=env, check=True)
    assert message.read_text() == "checked"
