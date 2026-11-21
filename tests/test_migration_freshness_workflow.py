"""Contract tests for the `migration-freshness` required check.

The check is a job in dev-build-deploy.yml, which runs on every pull request.
migration-freshness-refresh.yml marks open pull requests stale when their base
moves. Both must protect the same paths, or one of them misses a stale PR.
"""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PR_CALLER = ROOT / ".github/workflows/dev-build-deploy.yml"
REFRESH = ROOT / ".github/workflows/migration-freshness-refresh.yml"

ACTION = "mindsdb/github-actions/migration-freshness@"
SCOPE_INPUTS = ("protected-paths", "constant-file", "constant-pattern")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _pr_job() -> dict:
    return _load(PR_CALLER)["jobs"]["migration-freshness"]


def _action_step(job: dict) -> dict:
    return next(step for step in job["steps"] if str(step.get("uses", "")).startswith(ACTION))


def test_the_pr_job_always_runs_so_its_check_is_never_skipped():
    # GitHub counts a skipped job as passing, so a `needs:` that fails or an
    # `if:` that is false would turn the required check green.
    job = _pr_job()

    assert "needs" not in job
    assert "if" not in job
    assert "name" not in job
    assert "pull_request" in _load(PR_CALLER)[True]


def test_the_pr_job_runs_the_action_in_pr_mode_and_checks_out_the_head():
    job = _pr_job()
    scope = _action_step(job)
    checkout = next(step for step in job["steps"] if str(step.get("uses", "")).startswith("actions/checkout@"))

    assert job["steps"][0] is scope
    # A skipped or forgiven scope step leaves `protected` unset, which skips
    # every check after it and turns the job green.
    assert "if" not in scope
    assert "continue-on-error" not in scope
    assert "continue-on-error" not in job
    assert scope["id"] == "scope"
    assert scope["with"]["mode"] == "pr"
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.head.sha }}"
    assert checkout["with"]["persist-credentials"] is False
    assert all(step["if"] == "steps.scope.outputs.protected == 'true'" for step in job["steps"][1:])


def test_the_refresher_writes_statuses_on_every_push_to_a_gated_base():
    workflow = _load(REFRESH)
    jobs = workflow["jobs"]

    assert "migration-freshness" not in jobs
    assert set(workflow[True]["push"]["branches"]) == {"staging", "main"}
    assert workflow["permissions"] == {}
    for job in jobs.values():
        assert job["permissions"]["statuses"] == "write"


def test_the_refresher_rechecks_a_pull_request_retargeted_onto_a_gated_base():
    # A stacked pull request whose base merged is retargeted onto staging and
    # keeps the check it passed against its old base. pull_request_target runs
    # with the base repository's token, so the job must run nothing but the action.
    workflow = _load(REFRESH)
    trigger = workflow[True]["pull_request_target"]

    assert trigger["types"] == ["edited"]
    assert set(trigger["branches"]) == {"staging", "main"}
    for job in workflow["jobs"].values():
        assert "github.event.changes.base" in job["if"]
        assert [step["uses"].split("@")[0] for step in job["steps"]] == [ACTION.rstrip("@")]


def test_both_workflows_protect_the_same_paths():
    pr_inputs = _action_step(_pr_job())["with"]
    for job in _load(REFRESH)["jobs"].values():
        refresh_inputs = _action_step(job)["with"]
        for name in SCOPE_INPUTS:
            assert refresh_inputs[name] == pr_inputs[name], name
