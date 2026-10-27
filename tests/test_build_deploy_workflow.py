"""Contracts for the image build and deploy pipeline in build-deploy.yml.

Every job runs on a GitHub-hosted runner. The build pushes to an ECR tier
through GitHub OIDC, PR and staging images to the dev tier and main's to the
prod tier, and the staging and prod rollouts happen in a private deploy
repository.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
CHART = ROOT / "deployment/cowork-server"
BUILD_DEPLOY_TEXT = (WORKFLOWS / "build-deploy.yml").read_text()
BUILD_DEPLOY = yaml.safe_load(BUILD_DEPLOY_TEXT)
JOBS = BUILD_DEPLOY["jobs"]

# Helm values cannot read a GitHub secret, so the chart names the registry.
REGISTRY = "168681354662.dkr.ecr.us-east-1.amazonaws.com"
DEV_TIER = "mindsdb-cowork-server-dev"
PROD_TIER = "mindsdb-cowork-server"
DEV_ROLE = "gha-cowork-server-ecr-dev"
PROD_ROLE = "gha-cowork-server-ecr-prod"
# The workflows name the account only through this organization secret, so it
# stays out of the public repository and GitHub masks it in the log.
ROLE_TO_ASSUME = "arn:aws:iam::${{ secrets.AWS_DEV_ACCOUNT_ID }}:role/${{ env.ECR_ROLE }}"
PULL_ONLY = {
    "ecr:GetAuthorizationToken",
    "ecr:BatchGetImage",
    "ecr:GetDownloadUrlForLayer",
    "ecr:BatchCheckLayerAvailability",
}
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
    assert configure["uses"] == CONFIGURE_AWS
    session_policy = configure["with"].pop("inline-session-policy", None)
    assert configure["with"] == {"role-to-assume": ROLE_TO_ASSUME, "aws-region": "${{ env.AWS_REGION }}"}
    # The scan runs the Snyk CLI with these credentials, so its session pulls only.
    if job_id == "scan":
        [statement] = json.loads(session_policy)["Statement"]
        assert (statement["Effect"], set(statement["Action"]), statement["Resource"]) == ("Allow", PULL_ONLY, "*")
    else:
        assert session_policy is None
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
    assert [step.get("uses", "").partition("@")[0] or f"run:{step['id']}" for step in build["steps"]] == [
        "actions/checkout",
        "aws-actions/configure-aws-credentials",
        "mindsdb/github-actions/build-push-ecr",
        "run:pushed",
    ]
    push = build["steps"][_step_index(build, "mindsdb/github-actions/build-push-ecr@")]
    assert push["with"] == {
        "module-name": "${{ env.ECR_REPOSITORY }}",
        "build-for-environment": "${{ inputs.build-environment }}",
        "builder": "local",
    }


def test_staging_and_prod_deploy_through_the_deploy_repository() -> None:
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
    # The deploy repository is private, so its name comes from a secret too.
    assert token["with"] == {
        "client-id": "${{ vars.DEPLOYER_APP_CLIENT_ID }}",
        "private-key": "${{ secrets.DEPLOYER_APP_PRIVATE_KEY }}",
        "owner": "mindsdb",
        "repositories": "${{ secrets.DEPLOYER_REPOSITORY }}",
        "permission-actions": "write",
    }
    assert dispatch["env"] == {
        "GH_TOKEN": "${{ steps.app.outputs.token }}",
        "DEPLOY_REPOSITORY": "mindsdb/${{ secrets.DEPLOYER_REPOSITORY }}",
        "DIGEST": "${{ needs.build.outputs.digest }}",
    }


def test_the_build_hands_on_the_tag_and_digest_it_pushed() -> None:
    """GitHub drops a job output that holds a secret, and the full image
    reference starts with the account id, so the scan rebuilds it from the tag."""
    assert JOBS["build"]["outputs"] == {
        "tag": "${{ steps.pushed.outputs.tag }}",
        "digest": "${{ steps.pushed.outputs.digest }}",
    }
    scan = JOBS["scan"]
    snyk = scan["steps"][_step_index(scan, "mindsdb/github-actions/snyk-docker-scan@")]
    assert snyk["with"]["image"] == (
        "${{ secrets.AWS_DEV_ACCOUNT_ID }}.dkr.ecr.us-east-1.amazonaws.com/"
        "${{ env.ECR_REPOSITORY }}:${{ needs.build.outputs.tag }}"
    )


SHA = "a" * 40
NEWER = "b" * 40
DIGEST = "sha256:" + "e5" * 32
RUN_URL = "https://github.com/mindsdb/example-deploy/actions/runs/42"
GH_STUB = """
case "$1 $2" in
  "workflow run") printf '%s\\n' "$STUB_DISPATCH" ;;
  "run watch") exit "$STUB_WATCH_STATUS" ;;
  "run view") echo failure ;;
  *) echo "unexpected: gh $*" >&2; exit 2 ;;
esac
"""
GIT_STUB = """
[ "$1" = ls-remote ] || exit 2
printf '%s\\trefs/heads/staging\\n' "$STUB_HEAD"
"""


def _roll_out(tmp_path: Path, *, dispatch_prints: str, watch_status: int, head: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the dispatch step's real script for a staging build of SHA, with
    `gh` and `git` faked and the branch at `head`."""
    log = tmp_path / "calls.log"
    for name, body in (("gh", GH_STUB), ("git", GIT_STUB)):
        stub = tmp_path / name
        stub.write_text(f'#!/usr/bin/env bash\nprintf \'%s\\n\' "{name} $*" >> "$STUB_LOG"\n{body}')
        stub.chmod(0o755)
    script = tmp_path / "step.sh"
    script.write_text(JOBS["deploy"]["steps"][1]["run"])
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)],
        env={
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "STUB_LOG": str(log),
            "STUB_DISPATCH": dispatch_prints,
            "STUB_WATCH_STATUS": str(watch_status),
            "STUB_HEAD": head,
            "GH_TOKEN": "app-token",
            "DEPLOY_REPOSITORY": "mindsdb/example-deploy",
            "DIGEST": DIGEST,
            "DEPLOY_ENV": "staging",
            "BRANCH": "staging",
            "GITHUB_SHA": SHA,
            "GITHUB_REPOSITORY": "mindsdb/cowork-server",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


def test_the_dispatch_hands_the_deploy_repository_the_built_digest(tmp_path: Path) -> None:
    result, calls = _roll_out(tmp_path, dispatch_prints=RUN_URL, watch_status=0, head=SHA)

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls[0] == (
        "gh workflow run deploy.yml -R mindsdb/example-deploy --ref main "
        f"-f repo=cowork-server -f env=staging -f sha={SHA} -f digest={DIGEST}"
    )


def test_a_failed_rollout_of_the_branch_head_fails_the_job(tmp_path: Path) -> None:
    result, _ = _roll_out(tmp_path, dispatch_prints=RUN_URL, watch_status=1, head=SHA)

    assert result.returncode == 1
    assert "::notice::" not in result.stdout


def test_a_rollout_a_newer_commit_superseded_passes_with_a_notice(tmp_path: Path) -> None:
    """Two merges inside one build window: the older one finishes after the
    newer one is the head, and the deploy repository refuses it."""
    result, calls = _roll_out(tmp_path, dispatch_prints=RUN_URL, watch_status=1, head=NEWER)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "git ls-remote https://github.com/mindsdb/cowork-server.git refs/heads/staging" in calls
    assert f"::notice::staging moved to {NEWER}" in result.stdout


def test_a_dispatch_that_reports_no_run_fails_with_its_own_message(tmp_path: Path) -> None:
    result, calls = _roll_out(tmp_path, dispatch_prints="", watch_status=0, head=SHA)

    assert result.returncode == 1
    assert "::error::gh workflow run did not report the deploy run it created" in result.stdout
    assert not any(call.startswith("gh run watch") for call in calls)


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.y*ml")), ids=lambda path: path.name)
def test_no_workflow_names_an_aws_account(path: Path) -> None:
    assert not re.search(r"arn:aws:[a-z0-9-]*:[a-z0-9-]*:[0-9]{12}:", path.read_text())


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
