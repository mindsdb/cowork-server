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


def test_required_integration_prerequisite_fails_in_staging(monkeypatch):
    monkeypatch.setenv("COWORK_REQUIRE_INTEGRATION", "true")
    monkeypatch.setattr(
        pytest,
        "skip",
        lambda reason: pytest.fail(f"unexpected green skip: {reason}"),
    )

    with pytest.raises(pytest.fail.Exception, match="absence is a defect"):
        missing_prerequisite("missing target")
