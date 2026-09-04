from pathlib import Path

import pytest
import yaml

from tests.integration.prereq import missing_prerequisite


ROOT = Path(__file__).resolve().parents[1]
NIGHTLY_WORKFLOW = ROOT / ".github/workflows/nightly-staging-integration.yml"


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


def test_required_integration_prerequisite_fails_in_staging(monkeypatch):
    monkeypatch.setenv("COWORK_REQUIRE_INTEGRATION", "true")
    monkeypatch.setattr(
        pytest,
        "skip",
        lambda reason: pytest.fail(f"unexpected green skip: {reason}"),
    )

    with pytest.raises(pytest.fail.Exception, match="absence is a defect"):
        missing_prerequisite("missing target")
