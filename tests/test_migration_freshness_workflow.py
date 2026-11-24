"""Contract tests for the `migration-freshness` required check.

The check is the job in migration-freshness.yml, which runs on every pull
request event that can change its answer, a retarget (`edited`) included.
migration-freshness-refresh.yml marks open pull requests stale when their base
moves. Both must protect the same paths, or one of them misses a stale PR.
"""

import re
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
PR_CALLER = WORKFLOWS / "migration-freshness.yml"
REFRESH = WORKFLOWS / "migration-freshness-refresh.yml"

ACTION = "mindsdb/github-actions/migration-freshness@"
# A commit of the action until it merges, then main.
PINNED_ACTION = re.compile(
    r"mindsdb/github-actions/migration-freshness@(?:main|[0-9a-f]{40})"
)
NOTIFY = "mindsdb/github-actions/.github/workflows/notify-main-failure.yml@"
SCOPE_INPUTS = ("protected-paths", "constant-file", "constant-pattern")
PROTECTED_FOLDERS = ("cowork/db/alembic/versions/", "cowork/skills_builtin/")
ON_PROTECTED = "steps.scope.outputs.protected == 'true'"
# What runs on the head once the scope step finds a protected change.
GRAPH_RUNS = {
    "Check that the migration graph has one head": (
        "uv run --locked python -m pytest"
        " tests/test_schema_migrations.py::test_migration_graph_has_single_head"
    ),
    "Check the builtin skills manifest": (
        "uv run --locked python scripts/builtin_skills_shipped.py --check"
    ),
}


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _pr_job() -> dict:
    return _load(PR_CALLER)["jobs"]["migration-freshness"]


def _refresh_jobs() -> dict:
    return _load(REFRESH)["jobs"]


def _action_step(job: dict) -> dict:
    return next(
        step for step in job["steps"] if str(step.get("uses", "")).startswith(ACTION)
    )


def _graph_steps(job: dict) -> dict:
    return {step["name"]: step for step in job["steps"] if "run" in step}


def test_the_pr_workflow_runs_on_every_pull_request_and_on_a_retarget():
    # `edited` re-runs the check when a pull request's base changes. A
    # branches or paths filter would leave a pull request with no run, and its
    # required check waiting forever.
    workflow = _load(PR_CALLER)

    assert workflow[True] == {
        "pull_request": {"types": ["opened", "reopened", "synchronize", "edited"]}
    }
    assert workflow["permissions"] == {}
    # One group per pull request: a shared group would cancel the check of
    # another pull request.
    assert workflow["concurrency"] == {
        "group": "${{ github.workflow_ref }}",
        "cancel-in-progress": True,
    }


def test_only_the_pr_workflow_reports_the_check():
    owners = [
        path.name
        for path in sorted(WORKFLOWS.glob("*.y*ml"))
        if "migration-freshness" in (_load(path).get("jobs") or {})
    ]

    assert owners == [PR_CALLER.name]


def test_the_pr_job_always_runs_so_its_check_is_never_skipped():
    # GitHub counts a skipped job as passing, so a `needs:` that fails or an
    # `if:` that is false would turn the required check green. The job id is
    # the check's name.
    job = _pr_job()

    assert "needs" not in job
    assert "if" not in job
    assert "name" not in job
    assert job["runs-on"] == "ubuntu-latest"
    assert job["permissions"] == {"contents": "read", "pull-requests": "read"}


def test_the_pr_job_runs_the_action_in_pr_mode_and_checks_out_the_head():
    job = _pr_job()
    scope = _action_step(job)
    checkout = next(
        step
        for step in job["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )

    assert job["steps"][0] is scope
    # A skipped scope step leaves `protected` unset, which skips every check
    # after it and turns the job green.
    assert "if" not in scope
    assert scope["id"] == "scope"
    assert scope["with"]["mode"] == "pr"
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.head.sha }}"
    assert checkout["with"]["persist-credentials"] is False
    assert all(step["if"] == ON_PROTECTED for step in job["steps"][1:])


def test_no_step_of_the_pr_job_can_fail_and_leave_the_check_green():
    job = _pr_job()

    assert "continue-on-error" not in job
    assert [step for step in job["steps"] if "continue-on-error" in step] == []
    for step in _graph_steps(job).values():
        assert "|| true" not in step["run"]


def test_the_pr_job_runs_both_graph_checks_on_the_head():
    steps = _graph_steps(_pr_job())

    assert {name: step["run"] for name, step in steps.items()} == GRAPH_RUNS
    # The head contains the base tip, so the tip is the manifest's baseline.
    assert steps["Check the builtin skills manifest"]["env"] == {
        "BUILTIN_SKILLS_BASELINE_REF": "${{ steps.scope.outputs.tip }}"
    }


def test_the_refresher_runs_on_every_push_to_a_gated_base_and_by_hand():
    # The pr job re-runs itself on a retarget, so the refresher needs no
    # pull_request_target, which GitHub blocks in a public repository.
    workflow = _load(REFRESH)

    assert set(workflow[True]) == {"push", "workflow_dispatch"}
    assert workflow[True]["push"] == {"branches": ["staging", "main"]}
    assert workflow["permissions"] == {}
    assert set(workflow["jobs"]) == {"refresh", "notify"}


def test_the_refresher_writes_statuses_on_every_run():
    # A skipped or forgiven refresh leaves stale pull requests green.
    job = _refresh_jobs()["refresh"]

    assert "if" not in job
    assert "continue-on-error" not in job
    assert job["runs-on"] == "ubuntu-latest"
    assert job["permissions"] == {
        "contents": "read",
        "pull-requests": "read",
        "statuses": "write",
    }
    assert [step["uses"].split("@")[0] for step in job["steps"]] == [ACTION.rstrip("@")]
    assert "if" not in job["steps"][0]
    assert "continue-on-error" not in job["steps"][0]
    assert (
        job["steps"][0]["with"]["mode"]
        == "${{ github.event_name == 'workflow_dispatch' && inputs.mode || 'refresh' }}"
    )


def test_a_failed_refresh_is_reported_on_any_day():
    # A failed staging refresh leaves stale pull requests mergeable on any day,
    # not only during the freeze, so the alert is not freeze-scoped.
    notify = _refresh_jobs()["notify"]

    assert notify["needs"] == ["refresh"]
    assert (
        notify["if"] == "${{ !cancelled() && !contains(needs.*.result, 'cancelled') }}"
    )
    assert notify["permissions"] == {"contents": "read", "actions": "read"}
    assert notify["uses"].startswith(NOTIFY)
    assert (
        notify["with"]["status"]
        == "${{ contains(needs.*.result, 'failure') && 'failed' || 'recovered' }}"
    )
    assert notify["with"].get("freeze-scoped", False) is False
    assert notify["with"].get("runs-on", "ubuntu-latest") == "ubuntu-latest"
    assert notify["secrets"] == "inherit"


def test_both_workflows_protect_the_same_paths_with_the_same_action():
    pr_step = _action_step(_pr_job())
    refresh_step = _action_step(_refresh_jobs()["refresh"])

    # No other input, such as gated-bases or token, overrides the action's
    # defaults: a narrower gated-bases skips the check or the refresh.
    assert set(pr_step["with"]) == {"mode", *SCOPE_INPUTS}
    assert set(refresh_step["with"]) == {"mode", *SCOPE_INPUTS}
    for name in SCOPE_INPUTS:
        assert refresh_step["with"][name] == pr_step["with"][name], name
    assert refresh_step["uses"] == pr_step["uses"]
    assert PINNED_ACTION.fullmatch(pr_step["uses"])


def test_the_protected_paths_match_every_tracked_file_in_the_protected_folders():
    # The action matches with fnmatch, where `*` also matches `/`.
    patterns = _action_step(_pr_job())["with"]["protected-paths"].split()
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    in_folders = {path for path in tracked if path.startswith(PROTECTED_FOLDERS)}
    matched = {
        path
        for path in tracked
        if any(fnmatchcase(path, pattern) for pattern in patterns)
    }

    # A moved folder would leave both sets empty and the patterns protecting nothing.
    for folder in PROTECTED_FOLDERS:
        assert any(path.startswith(folder) for path in tracked), folder
    assert matched == in_folders


def test_the_constant_pattern_matches_only_the_two_version_definitions():
    # The action searches each changed line of the constant file, so a pattern
    # that also matched a use or a comment would protect unrelated edits.
    inputs = _action_step(_pr_job())["with"]
    pattern = re.compile(inputs["constant-pattern"])
    source = (ROOT / inputs["constant-file"]).read_text(encoding="utf-8")

    matched = [line for line in source.splitlines() if pattern.search(line)]

    assert inputs["constant-file"] == "cowork/services/skills.py"
    assert [line.split("=")[0].strip() for line in matched] == [
        "BUILTIN_SKILLS_VERSION",
        "CODE_BUILTIN_SKILLS_VERSION",
    ]
    assert all(
        re.fullmatch(r"(CODE_)?BUILTIN_SKILLS_VERSION = \d+", line) for line in matched
    )
