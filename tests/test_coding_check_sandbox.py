from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from cowork.coding.check_sandbox import CheckPolicy, GitDirectories, check_environment, sandboxed_argv
from cowork.coding.contracts import PermissionMode, TaskWorkspace, WorkspaceKind
from cowork.coding.project_models import CodeProject, ProjectCommand, ProjectFolder
from cowork.coding.project_workspaces import ProjectCommandRunner
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


def test_full_access_checks_run_outside_the_sandbox(tmp_path: Path) -> None:
    policy = CheckPolicy.for_task(PermissionMode.full_access, network_access=False)

    assert sandboxed_argv(["npm", "test"], policy, [tmp_path]) == ["npm", "test"]


@pytest.mark.parametrize("network_access", [False, True])
def test_sandboxed_checks_follow_the_task_network_setting(tmp_path: Path, network_access: bool) -> None:
    policy = CheckPolicy.for_task(PermissionMode.workspace, network_access=network_access)
    argv = sandboxed_argv(["npm", "test"], policy, [tmp_path], codex=Path("/bin/codex"))
    joined = " ".join(argv)

    assert argv[:2] == ["/bin/codex", "sandbox"]
    assert argv[-3:] == ["--", "npm", "test"]
    assert "allow_local_binding=true" in joined
    assert ("features.network_proxy=true" in argv) is not network_access
    assert ("domains={}" in joined) is not network_access


def test_hooks_may_stage_but_not_change_what_makes_git_run_programs(tmp_path: Path) -> None:
    common = tmp_path / "repo" / ".git"
    private = common / "worktrees" / "task"
    dirs = GitDirectories(common=common, private=private, branch_ref="refs/heads/task")
    grants = dirs.grants()
    assert grants[common] == "read"
    assert grants[private] == "write"
    assert grants[private / "HEAD"] == "read"
    assert grants[private / "gitdir"] == "read"
    assert common / "refs/heads/task" not in grants
    assert dirs.grants(committing=True)[common / "refs/heads/task"] == "write"


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
def test_staging_and_commit_use_the_task_sandbox(tmp_path: Path, checks: CheckPolicy, monkeypatch) -> None:
    repo = repository(tmp_path / "repo")
    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    monkeypatch.setenv("INTERNAL_AUTH_HEADER", "test-only")
    calls = []
    real_run = subprocess.run

    def record(argv, **kwargs):
        if "add" in argv or "commit" in argv:
            calls.append((argv, kwargs["env"]))
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", record)
    WorkspaceManager(tmp_path / "root").commit(str(repo), "Change", checks)
    assert len(calls) == 2
    for argv, env in calls:
        assert argv[1] == "sandbox"
        assert f"core.hooksPath={os.devnull}" in argv
        assert "INTERNAL_AUTH_HEADER" not in env


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
