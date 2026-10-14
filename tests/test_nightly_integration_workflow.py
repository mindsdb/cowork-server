from pathlib import Path

import pytest
import yaml

from tests.integration.prereq import missing_prerequisite


ROOT = Path(__file__).resolve().parents[1]
NIGHTLY_WORKFLOW = ROOT / ".github/workflows/nightly-staging-integration.yml"
INTEGRATION_SUITE = ROOT / ".github/workflows/tests-integration.yml"


def test_nightly_workflow_calls_staging_suite_and_reports_its_result():
    workflow = NIGHTLY_WORKFLOW.read_text(encoding="utf-8")
    parsed_workflow = yaml.safe_load(workflow)

    assert parsed_workflow[True] == {
        "schedule": [{"cron": "41 6 * * *"}],
        "workflow_dispatch": None,
    }
    assert "permissions: {}" in workflow
    assert (
        """\
  integration:
    permissions:
      contents: read
    uses: ./.github/workflows/tests-integration.yml
    with:
      deploy-env: staging
      runner: mdb-dev
      # A scheduled run starts on the default branch. Without this, the job
      # would check main's copy of the tests against staging's pods.
      ref: staging
    secrets: inherit
"""
        in workflow
    )
    assert (
        """\
  notify:
    needs: [integration]
    if: ${{ !cancelled() && !contains(needs.*.result, 'cancelled') }}
    permissions:
      contents: read
      actions: read
    uses: mindsdb/github-actions/.github/workflows/notify-main-failure.yml@main
"""
        in workflow
    )
    assert (
        "status: ${{ contains(needs.*.result, 'failure') && 'failed' || 'recovered' }}"
        in workflow
    )


def test_integration_callers_supply_the_suite_input_contract():
    # New required inputs must reach every caller, including scheduled runs.
    # `yaml.safe_load` reads the `on:` key as the boolean True.
    suite = yaml.safe_load(INTEGRATION_SUITE.read_text(encoding="utf-8"))
    inputs = suite[True]["workflow_call"]["inputs"]
    defined = set(inputs)
    required = {name for name, spec in inputs.items() if spec.get("required")}
    callers = []

    for path in sorted(INTEGRATION_SUITE.parent.glob("*.y*ml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_id, job in workflow.get("jobs", {}).items():
            if job.get("uses") != "./.github/workflows/tests-integration.yml":
                continue
            caller = f"{path.name}:{job_id}"
            callers.append(caller)
            passed = set(job.get("with", {}))
            missing = required - passed
            unknown = passed - defined
            assert not missing, f"{caller} is missing required inputs {sorted(missing)}"
            assert not unknown, f"{caller} passes undefined inputs {sorted(unknown)}"

    assert callers, "no callers of tests-integration.yml found"


def test_the_suite_checks_out_the_ref_its_caller_names():
    suite = yaml.safe_load(INTEGRATION_SUITE.read_text(encoding="utf-8"))
    steps = suite["jobs"]["integration-tests"]["steps"]
    own_checkouts = [
        step
        for step in steps
        if step.get("uses", "").startswith("actions/checkout@")
        and "repository" not in step.get("with", {})
    ]

    # Without the input, the checkout takes the ref that triggered the run,
    # which for a scheduled caller is always the default branch.
    assert [step.get("with", {}).get("ref") for step in own_checkouts] == [
        "${{ inputs.ref }}"
    ]
    # Empty by default, so the deploy callers keep testing the commit they
    # just deployed without naming it.
    ref_input = suite[True]["workflow_call"]["inputs"].get("ref", {})
    assert ref_input.get("default") == ""
    assert ref_input.get("required") is False


def test_every_scheduled_caller_pins_the_ref_matching_its_deploy_env():
    """A scheduled run starts on the default branch, whatever environment it tests.

    So a scheduled caller has to name the branch its target deployment was
    built from. Otherwise it checks main's copy of the tests against code main
    does not have yet, and every unreleased behavior change turns it red.
    """
    checked = []
    for path in sorted(INTEGRATION_SUITE.parent.glob("*.y*ml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        triggers = workflow.get(True)
        if not isinstance(triggers, dict) or "schedule" not in triggers:
            continue

        for job_id, job in workflow.get("jobs", {}).items():
            if job.get("uses") != "./.github/workflows/tests-integration.yml":
                continue

            given = job.get("with", {})
            deploy_env = given.get("deploy-env")
            assert given.get("ref") == deploy_env, (
                f"{path.name}:{job_id} tests the {deploy_env} deployment from "
                f"ref={given.get('ref')!r}; a scheduled run checks out the default "
                f"branch unless it names {deploy_env!r}"
            )
            checked.append(f"{path.name}:{job_id}")

    # A parsing slip that finds no caller would otherwise pass this vacuously.
    assert "nightly-staging-integration.yml:integration" in checked


def test_required_integration_prerequisite_fails_in_staging(monkeypatch):
    monkeypatch.setenv("COWORK_REQUIRE_INTEGRATION", "true")
    monkeypatch.setattr(
        pytest,
        "skip",
        lambda reason: pytest.fail(f"unexpected green skip: {reason}"),
    )

    with pytest.raises(pytest.fail.Exception, match="absence is a defect"):
        missing_prerequisite("missing target")
