"""Contract tests for the `deploy-to-dev` rollout in build-deploy.yml.

The shared `dev` namespace serves cowork.dev.mindshub.ai. Only a PR labeled
`deploy-to-dev` rolls into it, through `deploy-argocd`: `deploy` takes
production builds with Helm, and `deploy-pr-env` gives each PR a namespace of
its own.
"""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILD_DEPLOY = ROOT / ".github/workflows/build-deploy.yml"
PR_CALLER = ROOT / ".github/workflows/dev-build-deploy.yml"
DEV_VALUES = ROOT / "deployment/cowork-server/values-dev.yaml"

WRITE_ENV_STATE = "mindsdb/github-actions/write-env-state@"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _jobs() -> dict:
    return _load(BUILD_DEPLOY)["jobs"]


def _write_step(job: dict) -> dict:
    return next(step for step in job["steps"] if str(step.get("uses", "")).startswith(WRITE_ENV_STATE))


def test_only_a_pr_build_with_the_deploy_to_dev_label_rolls_into_dev():
    condition = _jobs()["deploy-argocd"]["if"]

    assert "inputs.pr-environment" in condition
    assert "contains(github.event.pull_request.labels.*.name, 'deploy-to-dev')" in condition
    assert "inputs.build-environment == 'staging'" in condition


def test_a_fork_pr_never_rolls_into_dev():
    condition = _jobs()["deploy-argocd"]["if"]
    caller = _load(PR_CALLER)["jobs"]
    gate_script = caller["gate"]["steps"][0]["run"]

    assert "github.event.pull_request.head.repo.full_name == github.repository" in condition
    assert caller["build-deploy"]["needs"] == ["gate"]
    assert caller["build-deploy"]["if"] == "needs.gate.outputs.run == 'true'"
    assert caller["build-deploy"]["with"]["deploy-through-argocd"] is True
    assert 'if [ "$HEAD_REPO" != "$THIS_REPO" ]; then\n  echo "run=false"' in gate_script


def test_the_rollout_records_the_dev_environment_for_argo_cd():
    job = _jobs()["deploy-argocd"]
    write = _write_step(job)["with"]

    assert job["env"]["DEPLOY_ENV"] == "${{ inputs.pr-environment && 'dev' || 'staging' }}"
    assert write["env"] == "${{ env.DEPLOY_ENV }}"
    assert write["service"] == "cowork-server"
    assert DEV_VALUES.is_file()


def test_the_rollout_deploys_the_image_this_run_built_after_its_scan():
    job = _jobs()["deploy-argocd"]
    write = _write_step(job)["with"]

    assert set(job["needs"]) == {"build", "scan"}
    # The PR head, not the merge commit GitHub drops once the PR closes, and
    # the head tag build-image pushes for it.
    assert job["env"]["DEPLOY_SHA"] == "${{ github.event.pull_request.head.sha || github.sha }}"
    assert write["sha"] == "${{ env.DEPLOY_SHA }}"
    assert write["tag"] == "${{ env.IMAGE_TAG }}"
    assert "head-" in job["env"]["IMAGE_TAG"]


def test_rollouts_into_dev_queue_instead_of_cancelling_each_other():
    concurrency = _jobs()["deploy-argocd"]["concurrency"]

    assert "github.event" not in concurrency["group"]
    assert concurrency["cancel-in-progress"] is False


def test_adding_the_label_starts_a_run():
    triggers = _load(PR_CALLER)[True]["pull_request"]["types"]

    assert "labeled" in triggers


def test_the_helm_rollout_is_left_to_production():
    jobs = _jobs()
    condition = jobs["deploy"]["if"]

    assert "deploy-dev" not in jobs
    assert "!inputs.deploy-through-argocd" in condition
    assert "inputs.build-environment == 'production'" in condition
    assert "development" not in condition
