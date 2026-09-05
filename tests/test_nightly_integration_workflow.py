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


def test_nightly_workflow_passes_only_inputs_the_suite_defines():
    # actionlint rejects the whole workflow-lint job over an input the callee
    # does not declare, so the caller's `with:` block is checked here too.
    # `yaml.safe_load` reads the `on:` key as the boolean True.
    caller = yaml.safe_load(NIGHTLY_WORKFLOW.read_text(encoding="utf-8"))
    suite = yaml.safe_load(INTEGRATION_SUITE.read_text(encoding="utf-8"))

    passed = set(caller["jobs"]["integration"]["with"])
    defined = set(suite[True]["workflow_call"]["inputs"])

    assert not passed - defined, (
        f"nightly caller passes {sorted(passed - defined)}, "
        f"which tests-integration.yml does not define"
    )


def test_required_integration_prerequisite_fails_in_staging(monkeypatch):
    monkeypatch.setenv("COWORK_REQUIRE_INTEGRATION", "true")
    monkeypatch.setattr(
        pytest,
        "skip",
        lambda reason: pytest.fail(f"unexpected green skip: {reason}"),
    )

    with pytest.raises(pytest.fail.Exception, match="absence is a defect"):
        missing_prerequisite("missing target")
