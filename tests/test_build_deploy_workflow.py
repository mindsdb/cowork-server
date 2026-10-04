"""Contracts for the image build and deploy pipeline in build-deploy.yml.

Every job runs on a GitHub-hosted runner. The build pushes to an ECR tier
through GitHub OIDC, PR and staging images to the dev tier and main's to the
prod tier, and the staging and prod rollouts happen in mindsdb/deployer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
CHART = ROOT / "deployment/cowork-server"
BUILD_DEPLOY_TEXT = (WORKFLOWS / "build-deploy.yml").read_text()
BUILD_DEPLOY = yaml.safe_load(BUILD_DEPLOY_TEXT)
JOBS = BUILD_DEPLOY["jobs"]

REGISTRY = "168681354662.dkr.ecr.us-east-1.amazonaws.com"
DEV_TIER = "mindsdb-cowork-server-dev"
PROD_TIER = "mindsdb-cowork-server"
DEV_ROLE = "arn:aws:iam::168681354662:role/gha-cowork-server-ecr-dev"
PROD_ROLE = "arn:aws:iam::168681354662:role/gha-cowork-server-ecr-prod"
CONFIGURE_AWS = (
    "aws-actions/configure-aws-credentials@e1253824e5c10ff9df46874f81ed3ec929e19cfd"
)
# An empty name means no Environment, which is what gives a PR build the
# `pull_request` OIDC subject.
TIER_ENVIRONMENT = (
    "${{ inputs.build-environment == 'production' && 'prod' || "
    "inputs.build-environment == 'staging' && 'staging' || '' }}"
)


def _step_index(job: dict, uses_prefix: str) -> int:
    return next(
        index
        for index, step in enumerate(job["steps"])
        if step.get("uses", "").startswith(uses_prefix)
    )


def test_only_main_builds_reach_the_prod_tier() -> None:
    assert BUILD_DEPLOY["env"] == {
        "ECR_REPOSITORY": (
            "${{ inputs.build-environment == 'production' && "
            f"'{PROD_TIER}' || '{DEV_TIER}' }}}}"
        ),
        "ECR_ROLE": (
            "${{ inputs.build-environment == 'production' && "
            f"'{PROD_ROLE}' || '{DEV_ROLE}' }}}}"
        ),
    }


@pytest.mark.parametrize("job_id", ["build", "scan"])
def test_build_and_scan_assume_the_tier_role_through_oidc(job_id: str) -> None:
    job = JOBS[job_id]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["environment"] == TIER_ENVIRONMENT
    assert job["permissions"] == {"contents": "read", "id-token": "write"}
    configure = job["steps"][_step_index(job, "aws-actions/configure-aws-credentials@")]
    assert configure == {
        "uses": CONFIGURE_AWS,
        "with": {"role-to-assume": "${{ env.ECR_ROLE }}", "aws-region": "${{ env.AWS_REGION }}"},
    }
    action = "build-push-ecr" if job_id == "build" else "snyk-docker-scan"
    assert _step_index(job, "aws-actions/configure-aws-credentials@") < _step_index(
        job, f"mindsdb/github-actions/{action}@"
    )
    for step in job["steps"]:
        if step.get("uses", "").startswith("actions/checkout@"):
            assert step["with"]["persist-credentials"] is False


def test_the_build_pushes_with_a_local_builder_to_the_tier_repository() -> None:
    build = JOBS["build"]
    # This job can mint the OIDC token for the prod-tier role, so every step it
    # runs is listed here.
    assert [step["uses"].partition("@")[0] for step in build["steps"]] == [
        "actions/checkout",
        "aws-actions/configure-aws-credentials",
        "mindsdb/github-actions/build-push-ecr",
    ]
    push = build["steps"][_step_index(build, "mindsdb/github-actions/build-push-ecr@")]
    assert push["with"] == {
        "module-name": "${{ env.ECR_REPOSITORY }}",
        "build-for-environment": "${{ inputs.build-environment }}",
        "builder": "local",
    }


def test_staging_and_prod_deploy_through_the_deployer() -> None:
    deploy = JOBS["deploy"]
    assert deploy["runs-on"] == "ubuntu-latest"
    assert deploy["needs"] == ["build", "scan"]
    assert deploy["permissions"] == {}
    assert deploy["timeout-minutes"] == 45
    assert deploy["environment"]["name"] == (
        "${{ inputs.build-environment == 'production' && 'prod' || 'staging' }}"
    )
    token, dispatch = deploy["steps"]
    assert token["uses"] == (
        "actions/create-github-app-token@bcd2ba49218906704ab6c1aa796996da409d3eb1"
    )
    assert token["with"] == {
        "client-id": "${{ vars.DEPLOYER_APP_CLIENT_ID }}",
        "private-key": "${{ secrets.DEPLOYER_APP_PRIVATE_KEY }}",
        "owner": "mindsdb",
        "repositories": "deployer",
        "permission-actions": "write",
    }
    assert dispatch["env"] == {"GH_TOKEN": "${{ steps.app.outputs.token }}"}
    assert (
        'url=$(gh workflow run deploy.yml -R mindsdb/deployer --ref main '
        '-f repo=cowork-server -f env="$DEPLOY_ENV" -f sha="$GITHUB_SHA")'
    ) in dispatch["run"]
    assert 'gh run watch "${url##*/}" -R mindsdb/deployer --exit-status' in dispatch["run"]


def test_this_repository_no_longer_deploys_anything_itself() -> None:
    """No Helm, no Argo CD, no cluster credential in the public repository."""
    assert "deploy-pr-env" not in JOBS
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        text = path.read_text()
        for needle in ("ARGOCD_AUTH_TOKEN", "argocd-pr-env-deploy", "aws-helm-multi-deploy"):
            assert needle not in text, f"{path.name} still mentions {needle}"


def test_the_pr_environment_is_tested_with_no_secrets() -> None:
    pr_tests = JOBS["integration-tests-pr-env"]
    assert pr_tests["needs"] == ["build"]
    assert pr_tests["if"] == (
        "inputs.pr-environment && "
        "contains(github.event.pull_request.labels.*.name, 'deploy')"
    )
    assert pr_tests["permissions"] == {"contents": "read"}
    assert "secrets" not in pr_tests
    assert pr_tests["with"] == {
        "deploy-env": "pr-cowork-server-${{ github.event.pull_request.number }}"
    }
    # The PR comment says "up" only once the environment served this build.
    assert BUILD_DEPLOY[True]["workflow_call"]["outputs"]["pr-env-deployed"]["value"] == (
        "${{ jobs.integration-tests-pr-env.outputs.pr-env-serving }}"
    )


@pytest.mark.parametrize(
    ("workflow", "condition"),
    [
        ("dev-build-deploy.yml", "needs.gate.outputs.run == 'true'"),
        ("publish-staging.yml", "github.ref == 'refs/heads/staging'"),
        ("publish.yml", "github.ref == 'refs/heads/main'"),
    ],
)
def test_every_caller_grants_the_oidc_token_and_guards_its_ref(
    workflow: str, condition: str
) -> None:
    caller = yaml.safe_load((WORKFLOWS / workflow).read_text())["jobs"]["build-deploy"]
    assert caller["uses"] == "./.github/workflows/build-deploy.yml"
    assert caller["permissions"] == {"contents": "read", "id-token": "write"}
    assert caller["if"] == condition


def test_the_chart_pulls_each_environment_from_its_own_tier() -> None:
    """PR environments, dev and staging take the base value; prod overrides it."""

    def repository(values_file: str) -> str | None:
        values = yaml.safe_load((CHART / values_file).read_text()) or {}
        return values.get("deployment", {}).get("image", {}).get("repository")

    assert repository("values.yaml") == f"{REGISTRY}/{DEV_TIER}"
    assert repository("values-dev.yaml") is None
    assert repository("values-staging.yaml") is None
    assert repository("values-prod.yaml") == f"{REGISTRY}/{PROD_TIER}"
