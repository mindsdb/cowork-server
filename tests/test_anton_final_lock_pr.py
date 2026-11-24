"""Exercise refresh decisions against real Git trees, and the PR lifecycle."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SPEC = importlib.util.spec_from_file_location(
    "anton_final_lock_pr", Path(__file__).resolve().parents[1] / "scripts/anton_final_lock_pr.py"
)
pr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pr)
TITLE = "fix(deps): lock anton-agent's final 1.0 instead of 1.0rc1 (ENG-4355)"
FLOOR = '[project]\ndependencies = ["anton-agent>={},<3"]\n'


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pr.run("git", "init", "-q")
    pr.run("git", "config", "user.name", "Test")
    pr.run("git", "config", "user.email", "test@example.com")
    (tmp_path / "cowork").mkdir()
    write_lock("1.0rc1", "1")
    Path("pyproject.toml").write_text(FLOOR.format("1.0rc1"))
    pr.run("git", "add", ".")
    pr.run("git", "commit", "-qm", "base")
    return pr.run("git", "rev-parse", "HEAD")


def write_lock(anton, other):
    text = f"anton-agent=={anton}\nother=={other}\n"
    Path("uv.lock").write_text(text)
    Path("cowork/release-requirements.txt").write_text(text)


def test_identical_proposal_does_not_restart_ci(repo):
    write_lock("1.0", "1")
    assert pr.prepare_commit(repo, "", "bump")
    previous = pr.run("git", "rev-parse", "HEAD")
    pr.run("git", "checkout", "--detach", repo)
    write_lock("1.0", "1")
    assert not pr.prepare_commit(repo, previous, "bump")
    assert pr.run("git", "rev-parse", "HEAD") == repo


def test_moved_staging_refreshes_same_version_and_preserves_other_changes(repo):
    write_lock("1.0", "1")
    pr.prepare_commit(repo, "", "same title")
    previous = pr.run("git", "rev-parse", "HEAD")
    pr.run("git", "checkout", "--detach", repo)
    write_lock("1.0rc1", "2")
    pr.run("git", "commit", "-qam", "another dependency")
    base = pr.run("git", "rev-parse", "HEAD")
    write_lock("1.0", "2")
    assert pr.prepare_commit(base, previous, "same title")
    assert pr.run("git", "rev-parse", "HEAD^") == base
    assert "other==2" in pr.run("git", "show", "HEAD:uv.lock")


def test_same_parent_but_changed_resolution_refreshes(repo):
    write_lock("1.0", "1")
    pr.prepare_commit(repo, "", "bump")
    previous = pr.run("git", "rev-parse", "HEAD")
    pr.run("git", "checkout", "--detach", repo)
    write_lock("1.0.post1", "1")
    assert pr.prepare_commit(repo, previous, "new final")


def test_empty_relock_is_refused(repo):
    with pytest.raises(RuntimeError, match="no change"):
        pr.prepare_commit(repo, "", "empty")


@pytest.mark.parametrize("final, reason", [
    ("true", "already locks a final"), ("false", "newer than any final"),
])
def test_an_open_pr_with_nothing_to_propose_is_closed(monkeypatch, final, reason):
    calls = []
    def run(*args):
        calls.append(args)
        if args[:3] == ("git", "rev-parse", "HEAD"):
            return "base"
        if args[:3] == ("gh", "pr", "list"):
            return '[{"number": 42}]'
        return ""
    monkeypatch.setattr(pr, "run", run)
    monkeypatch.setattr(pr, "remote_head", lambda branch: "base")
    for key, value in {"REPO": "mindsdb/cowork-server", "BRANCH": "bot/anton-final-lock",
                       "LOCKED": "1.0", "VERSION": "", "LOCKED_FINAL": final}.items():
        monkeypatch.setenv(key, value)
    pr.main()
    closes = [c for c in calls if c[:3] == ("gh", "pr", "close")]
    assert len(closes) == 1 and reason in closes[0][-1]


def test_staging_race_stops_before_any_pr_mutation(monkeypatch):
    monkeypatch.setenv("REPO", "mindsdb/cowork-server")
    monkeypatch.setenv("BRANCH", "bot/anton-final-lock")
    monkeypatch.setenv("LOCKED", "1.0rc1")
    monkeypatch.setattr(pr, "run", lambda *args: "old-base")
    monkeypatch.setattr(pr, "remote_head", lambda branch: "new-base")
    with pytest.raises(RuntimeError, match="Staging moved"):
        pr.main()


def test_explicit_lease_rejects_concurrent_branch_edit(repo, tmp_path):
    remote = tmp_path / "remote.git"
    pr.run("git", "init", "--bare", "-q", str(remote))
    pr.run("git", "remote", "add", "origin", str(remote))
    pr.run("git", "push", "origin", "HEAD:refs/heads/bot/anton-final-lock")
    write_lock("1.0", "1")
    pr.prepare_commit(repo, "", "bump")
    pr.run("git", "push", "origin", "HEAD:refs/heads/bot/anton-final-lock")
    with pytest.raises(subprocess.CalledProcessError):
        pr.run("git", "push", f"--force-with-lease=refs/heads/bot/anton-final-lock:{repo}",
               "origin", f"{repo}:refs/heads/bot/anton-final-lock")


def test_reconcile_pushes_once_then_refreshes_from_new_staging(repo, tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    pr.run("git", "init", "--bare", "-q", str(remote))
    pr.run("git", "remote", "add", "origin", str(remote))
    pr.run("git", "push", "origin", "HEAD:refs/heads/staging")
    real_run, calls = pr.run, []
    def run(*args):
        if args[0] != "gh":
            return real_run(*args)
        calls.append(args)
        if args[1:3] == ("pr", "list"):
            if "closed" in args:
                return "[]"
            return '[{"number": 42}]' if any(c[1:3] == ("pr", "create") for c in calls) else '[]'
        if args[1] == "api":
            return "123"
        return ""
    monkeypatch.setattr(pr, "run", run)
    for key, value in {"REPO": "mindsdb/cowork-server", "BRANCH": "bot/anton-final-lock",
                       "LOCKED": "1.0rc1", "VERSION": "1.0", "APP_SLUG": "test",
                       "RUNNER_TEMP": str(tmp_path)}.items():
        monkeypatch.setenv(key, value)
    write_lock("1.0", "1")
    pr.main()
    first = pr.remote_head("bot/anton-final-lock")
    pr.run("git", "checkout", "--detach", repo)
    write_lock("1.0", "1")
    pr.main()
    assert pr.remote_head("bot/anton-final-lock") == first
    assert sum(c[1:3] == ("pr", "create") for c in calls) == 1
    pr.run("git", "reset", "--hard", repo)
    write_lock("1.0rc1", "2")
    pr.run("git", "commit", "-qam", "staging moves")
    base = pr.run("git", "rev-parse", "HEAD")
    pr.run("git", "push", "origin", "HEAD:refs/heads/staging")
    write_lock("1.0", "2")
    pr.main()
    assert pr.remote_head("bot/anton-final-lock") != first
    assert pr.run("git", "rev-parse", "HEAD^") == base
    assert "other==2" in pr.run("git", "show", "HEAD:uv.lock")


@pytest.mark.parametrize("closed, proposes", [
    # Closed unmerged with this exact title: declined, so not reopened.
    ([{"number": 7, "title": TITLE, "mergedAt": None}], False),
    # Merged earlier, or a different final or candidate: still proposed.
    ([{"number": 7, "title": TITLE, "mergedAt": "2026-11-16T00:00:00Z"}], True),
    ([{"number": 7, "title": TITLE.replace("1.0rc1", "0.9rc1"), "mergedAt": None}], True),
    ([], True),
])
@pytest.mark.parametrize("closer, active", [("reviewer", False), ("test[bot]", False), ("reviewer", True)])
def test_a_declined_proposal_is_not_reopened(repo, tmp_path, monkeypatch, closed, proposes, closer, active):
    remote = tmp_path / "remote.git"
    pr.run("git", "init", "--bare", "-q", str(remote))
    pr.run("git", "remote", "add", "origin", str(remote))
    pr.run("git", "push", "origin", "HEAD:refs/heads/staging")
    real_run, calls = pr.run, []
    def run(*args):
        if args[0] != "gh":
            return real_run(*args)
        calls.append(args)
        if args[1:3] == ("pr", "list"):
            return json.dumps(closed) if "closed" in args else ('[{"number": 42}]' if active else "[]")
        if args[1] == "api" and args[2].startswith("repos/"):
            return closer
        return "123" if args[1] == "api" else ""
    monkeypatch.setattr(pr, "run", run)
    for key, value in {"REPO": "mindsdb/cowork-server", "BRANCH": "bot/anton-final-lock",
                       "LOCKED": "1.0rc1", "VERSION": "1.0", "APP_SLUG": "test",
                       "RUNNER_TEMP": str(tmp_path)}.items():
        monkeypatch.setenv(key, value)
    write_lock("1.0", "1")
    pr.main()
    expected = proposes or closer == "test[bot]" or active
    action = "edit" if active else "create"
    assert any(c[1:3] == ("pr", action) for c in calls) == expected
    assert bool(pr.remote_head("bot/anton-final-lock")) == expected
    if active:
        assert not any(c[1:3] == ("pr", "list") and "closed" in c for c in calls)


@pytest.mark.parametrize("new_locked, expected", [("1.1rc1", "1.1"), ("1.1", ""), ("1.2rc1", "")])
def test_retry_uses_fresh_staging_lock_and_replaces_old_decision(repo, tmp_path, monkeypatch, new_locked, expected):
    remote = tmp_path / "remote.git"
    pr.run("git", "init", "--bare", "-q", str(remote))
    pr.run("git", "remote", "add", "origin", str(remote))
    # A merge lands while the original attempt waits for PyPI.
    write_lock(new_locked, "2")
    pr.run("git", "commit", "-qam", "staging advances during wait")
    fresh = pr.run("git", "rev-parse", "HEAD")
    pr.run("git", "push", "origin", "HEAD:refs/heads/staging")
    pr.run("git", "checkout", "--detach", repo)
    write_lock("1.0", "1")
    # Also cover a race detected after prepare_commit has created a local commit.
    pr.prepare_commit(repo, "", "obsolete generated bump")
    monkeypatch.setattr(pr.selection, "CONTRACT", tmp_path / "cowork/release-requirements.txt")
    announcements = []
    def releases(announced):
        announcements.append(announced)
        return {"1.0": [{"yanked": False}], "1.1": [{"yanked": False}]}
    monkeypatch.setattr(pr.selection, "releases_listing", releases)
    real_run, calls = pr.run, []
    def run(*args):
        calls.append(args)
        if args[0] == "git":
            return real_run(*args)
        if args[0] == "uv":
            assert args == ("uv", "lock", "--upgrade-package", "anton-agent==1.1")
            assert pr.run("git", "rev-parse", "HEAD") == fresh
            write_lock("1.1", "2")
            return ""
        if args[0] == sys.executable:
            return ""
        if args[:3] == ("gh", "pr", "list"):
            return "[]" if expected or "closed" in args else '[{"number": 42}]'
        return "123" if args[1] == "api" else ""
    monkeypatch.setattr(pr, "run", run)
    for key, value in {"REPO": "mindsdb/cowork-server", "BRANCH": "bot/anton-final-lock",
                       "LOCKED": "1.0rc1", "VERSION": "1.0", "APP_SLUG": "test",
                       "LOCKED_FINAL": "false", "ANNOUNCED_VERSION": "1.1",
                       "RUNNER_TEMP": str(tmp_path)}.items():
        monkeypatch.setenv(key, value)
    pr.retry_staging()
    assert pr.os.environ["LOCKED"] == new_locked
    assert pr.os.environ["VERSION"] == expected
    assert pr.os.environ["LOCKED_FINAL"] == str(new_locked == "1.1").lower()
    assert announcements == ([] if new_locked == "1.1" else ["1.1"])
    if expected:
        assert pr.remote_head("bot/anton-final-lock") == pr.run("git", "rev-parse", "HEAD")
        assert pr.run("git", "rev-parse", "HEAD^") == fresh
        assert "other==2" in pr.run("git", "show", "HEAD:uv.lock")
        assert pr.run("git", "show", "HEAD:pyproject.toml") == FLOOR.format("1.1").strip()
        assert (sys.executable, "scripts/release_contract.py") in calls
        assert (sys.executable, "scripts/release_contract.py", "--check") in calls
    else:
        assert not pr.remote_head("bot/anton-final-lock")
        assert any(c[:3] == ("gh", "pr", "close") for c in calls)
        assert not any(c[0] == "uv" for c in calls)


def test_repeated_staging_changes_stop_after_three_attempts(monkeypatch):
    calls, refreshes = [], []
    def reconcile():
        calls.append(1)
        raise pr.StagingMoved("Staging moved")
    monkeypatch.setattr(pr, "main", reconcile)
    monkeypatch.setattr(pr, "refresh_staging", lambda: refreshes.append(1))
    with pytest.raises(pr.StagingMoved):
        pr.retry_staging()
    assert len(calls) == 3
    assert len(refreshes) == 2


def test_relock_raises_a_candidate_floor_before_locking(repo, monkeypatch):
    calls = []
    def run(*args):
        calls.append(args)
        if args[0] == "uv":
            assert Path("pyproject.toml").read_text() == FLOOR.format("1.0")
        return ""
    monkeypatch.setattr(pr, "run", run)
    pr.relock("1.0")
    assert calls == [
        ("uv", "lock", "--upgrade-package", "anton-agent==1.0"),
        (sys.executable, "scripts/release_contract.py"),
        (sys.executable, "scripts/release_contract.py", "--check"),
    ]


def test_relock_drops_anton_agents_allowance_before_locking(repo, monkeypatch):
    allowance = '\n[tool.cowork.release.allow-prerelease]\nanton-agent = "1.0rc1"\n'
    Path("pyproject.toml").write_text(FLOOR.format("1.0rc1") + allowance)
    def run(*args):
        if args[0] == "uv":
            assert Path("pyproject.toml").read_text() == FLOOR.format("1.0") + "\n[tool.cowork.release.allow-prerelease]\n"
        return ""
    monkeypatch.setattr(pr, "run", run)
    pr.relock("1.0")


def test_the_raised_floor_is_committed(repo):
    Path("pyproject.toml").write_text(FLOOR.format("1.0"))
    assert pr.prepare_commit(repo, "", "floor")
    assert pr.run("git", "show", "HEAD:pyproject.toml") == FLOOR.format("1.0").strip()


def test_retry_does_not_reset_checkout_for_non_staging_failure(monkeypatch):
    def reconcile():
        raise subprocess.CalledProcessError(1, ["git", "push", "--force-with-lease"])
    def refresh():
        pytest.fail("A rejected push lease must not trigger a reset/retry")
    monkeypatch.setattr(pr, "main", reconcile)
    monkeypatch.setattr(pr, "refresh_staging", refresh)
    with pytest.raises(subprocess.CalledProcessError):
        pr.retry_staging()
