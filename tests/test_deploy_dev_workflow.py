"""Contract tests for the `deploy-dev` rollout in build-deploy.yml.

The shared `dev` namespace serves cowork.dev.mindshub.ai. Only a PR labeled
`deploy-to-dev` rolls into it: `deploy` takes staging and production builds,
and `deploy-pr-env` gives each PR a namespace of its own.
"""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILD_DEPLOY = ROOT / ".github/workflows/build-deploy.yml"
PR_CALLER = ROOT / ".github/workflows/dev-build-deploy.yml"
DEV_VALUES = ROOT / "deployment/cowork-server/values-dev.yaml"

HELM_ACTION = "DevOps-Nirvana/aws-helm-multi-deploy-nodocker@v8"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _jobs() -> dict:
    return _load(BUILD_DEPLOY)["jobs"]


def _helm_step(job: dict) -> dict:
    return next(step for step in job["steps"] if step.get("uses") == HELM_ACTION)


def test_only_a_pr_build_with_the_deploy_to_dev_label_rolls_into_dev():
    condition = _jobs()["deploy-dev"]["if"]

    assert "inputs.pr-environment" in condition
    assert "contains(github.event.pull_request.labels.*.name, 'deploy-to-dev')" in condition


def test_a_fork_pr_never_rolls_into_dev():
    condition = _jobs()["deploy-dev"]["if"]
    caller = _load(PR_CALLER)["jobs"]
    gate_script = caller["gate"]["steps"][0]["run"]

    assert "github.event.pull_request.head.repo.full_name == github.repository" in condition
    assert caller["build-deploy"]["needs"] == ["gate"]
    assert caller["build-deploy"]["if"] == "needs.gate.outputs.run == 'true'"
    assert 'if [ "$HEAD_REPO" != "$THIS_REPO" ]; then\n  echo "run=false"' in gate_script


def test_the_rollout_targets_the_dev_namespace_with_the_dev_values():
    job = _jobs()["deploy-dev"]
    helm = _helm_step(job)["with"]

    assert job["environment"]["name"] == "dev"
    assert helm["environment-slug"] == "dev"
    assert helm["k8s-namespace"] == "dev"
    assert DEV_VALUES.is_file()


def test_the_rollout_deploys_the_image_this_run_built_after_its_scan():
    jobs = _jobs()
    dev = jobs["deploy-dev"]

    assert set(dev["needs"]) == {"build", "scan"}
    assert _helm_step(dev)["with"]["image-tag"] == _helm_step(jobs["deploy"])["with"]["image-tag"]
    assert any(step.get("uses", "").startswith("mindsdb/github-actions/setup-env@") for step in dev["steps"])


def test_rollouts_into_dev_queue_instead_of_cancelling_each_other():
    concurrency = _jobs()["deploy-dev"]["concurrency"]

    assert "github.event" not in concurrency["group"]
    assert concurrency["cancel-in-progress"] is False


def test_adding_the_label_starts_a_run():
    triggers = _load(PR_CALLER)[True]["pull_request"]["types"]

    assert "labeled" in triggers


def test_the_staging_and_production_rollout_still_skips_development_builds():
    condition = _jobs()["deploy"]["if"]

    assert "inputs.build-environment == 'staging'" in condition
    assert "inputs.build-environment == 'production'" in condition
    assert "development" not in condition
