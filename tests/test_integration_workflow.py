"""Contract tests for the integration workflow's wiring, runner and identities."""

import subprocess
from pathlib import Path

import pytest
import yaml

from tests.integration import test_post_deploy, test_two_replicas


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
BUILD_DEPLOY = (WORKFLOWS / "build-deploy.yml").read_text()
INTEGRATION = (WORKFLOWS / "tests-integration.yml").read_text()
PUBLISH = (WORKFLOWS / "publish.yml").read_text()
README = (ROOT / "README.md").read_text()
SUITE = yaml.safe_load(INTEGRATION)
SUITE_JOB = SUITE["jobs"]["integration-tests"]
CROSS_REPLICA_TEST = "test_reconnect_works_on_the_other_replica"


def _suite_step(name: str) -> dict:
    return next(step for step in SUITE_JOB["steps"] if step.get("name") == name)


@pytest.fixture(autouse=True)
def _canonical_prod_cowork_target(monkeypatch) -> None:
    """Standing-identity tests start from the one permitted key destination."""
    monkeypatch.setenv("COWORK_BASE_URL", test_post_deploy.PROD_COWORK_BASE_URL)


def test_no_job_defined_here_requests_a_self_hosted_runner() -> None:
    """Every job this repository's workflow files define runs GitHub-hosted.

    So does every `runs-on` these files pass to a reusable workflow. GitHub
    recommends GitHub-hosted runners for public repositories. A reusable
    workflow called from mindsdb/github-actions picks its runner in its own
    file, which this test does not read.
    """
    checked = []
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        text = path.read_text()
        assert "mdb-dev" not in text and "mdb-prod" not in text, path.name
        for job_id, job in yaml.safe_load(text).get("jobs", {}).items():
            runs_on = job.get("runs-on", job.get("with", {}).get("runs-on"))
            if runs_on is not None:
                assert runs_on == "ubuntu-latest", f"{path.name}:{job_id} {runs_on!r}"
                checked.append(f"{path.name}:{job_id}")
    assert "tests-integration.yml:integration-tests" in checked
    assert "build-deploy.yml:deploy" in checked
    assert yaml.safe_load((ROOT / ".github/actionlint.yaml").read_text()) == {
        "self-hosted-runner": {"labels": []}
    }


def test_the_suite_needs_no_cluster_access() -> None:
    """No runner input, no kubectl, no port-forward: every target is a public host."""
    assert SUITE_JOB["runs-on"] == "ubuntu-latest"
    assert set(SUITE[True]["workflow_call"]["inputs"]) == {"deploy-env", "ref"}
    assert "build-runner" not in BUILD_DEPLOY
    assert "build-runner" not in PUBLISH
    for step in SUITE_JOB["steps"]:
        assert "kubectl" not in step.get("run", ""), step.get("name")
        assert "_work/_tool" not in str(step.get("with", {})), step.get("name")


def test_the_two_replica_test_gets_a_real_redis() -> None:
    redis = SUITE_JOB["services"]["redis"]
    image, _, digest = redis["image"].partition("@")
    assert image.startswith("redis:")
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64
    assert redis["ports"] == ["6379:6379"]
    assert "redis-cli ping" in redis["options"]
    assert SUITE_JOB["env"]["COWORK_TEST_REDIS_URL"] == "redis://localhost:6379/1"


class _UnreachableRedis:
    async def ping(self) -> None:
        raise ConnectionError("Connection refused")


@pytest.mark.parametrize(
    ("named_url", "required", "outcome"),
    [
        ("redis://localhost:6379/1", "true", pytest.fail.Exception),
        ("redis://localhost:6379/1", "false", pytest.skip.Exception),
        (None, "true", pytest.skip.Exception),
    ],
    ids=["named-and-required-fails", "named-best-effort-skips", "unnamed-skips"],
)
async def test_an_unreachable_redis_fails_only_a_run_that_named_it(
    monkeypatch, named_url: str | None, required: str, outcome: type[BaseException]
) -> None:
    """A run that names no Redis promised none, so the two-replica tests skip.

    The staging nightly runs main's copy of this workflow against staging's
    tests, so its copy can predate the Redis service. A run that names its
    Redis, as this workflow does, fails on a missing one in staging and prod.
    """
    monkeypatch.setenv("COWORK_REQUIRE_INTEGRATION", required)
    if named_url is None:
        monkeypatch.delenv("COWORK_TEST_REDIS_URL", raising=False)
    else:
        monkeypatch.setenv("COWORK_TEST_REDIS_URL", named_url)
    monkeypatch.setattr(
        test_two_replicas.aioredis, "from_url", lambda url, **_: _UnreachableRedis()
    )
    fixture = test_two_replicas.redis_url.__wrapped__(monkeypatch)

    # Catch both outcomes and compare after: an uncaught skip would report this
    # test as skipped instead of failing it.
    with pytest.raises(
        (pytest.fail.Exception, pytest.skip.Exception), match="no Redis reachable"
    ) as raised:
        await anext(fixture)
    assert raised.type is outcome


def test_only_the_pod_level_cross_replica_test_is_left_out() -> None:
    """It needs port-forwards into the cluster, which a hosted runner cannot open."""
    assert SUITE_JOB["env"]["PYTEST_ADDOPTS"] == (
        f"--deselect tests/integration/test_post_deploy.py::{CROSS_REPLICA_TEST}"
    )
    # A renamed test would slip past a stale --deselect and fail every run.
    assert callable(getattr(test_post_deploy, CROSS_REPLICA_TEST))


def test_a_pr_environment_is_tested_only_once_it_serves_this_build() -> None:
    wait = _suite_step("Wait until the PR environment serves this build")
    assert wait["id"] == "serving"
    assert wait["if"] == "startsWith(inputs.deploy-env, 'pr-')"
    assert wait["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "PR_HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
    }
    assert wait["run"] == (
        "uv run python -m tests.integration.wait_for_build\n"
        'echo "serving=true" >> "$GITHUB_OUTPUT"\n'
    )
    steps = [step.get("name") for step in SUITE_JOB["steps"]]
    assert steps.index("Resolve the target and the identity source") < steps.index(
        wait["name"]
    ) < steps.index("Run integration tests")
    assert SUITE_JOB["outputs"] == {
        "pr-env-serving": "${{ steps.serving.outputs.serving }}"
    }
    assert SUITE[True]["workflow_call"]["outputs"]["pr-env-serving"]["value"] == (
        "${{ jobs.integration-tests.outputs.pr-env-serving }}"
    )


def test_prod_build_deploy_refuses_a_non_main_dispatch() -> None:
    """A branch-selected manual run must not enter the production workflow."""
    build_deploy = PUBLISH.split("  build-deploy:\n", 1)[1].split("\n  release:\n", 1)[
        0
    ]

    assert "push:\n    branches: [main]" in PUBLISH
    assert "build-environment: production" in build_deploy
    assert "if: github.ref == 'refs/heads/main'" in build_deploy


def test_prod_standing_key_docs_describe_the_environment_guards_in_force() -> None:
    """The workflow guard must not be presented as protection for the key.

    The `prod` Environment is that protection: `main` only, no admin bypass,
    and no required reviewer, so a release never waits for an approval.
    """
    normalized_readme = " ".join(README.split())
    for required in (
        "allows deployments only from the `main` branch",
        "`can_admins_bypass` is `false`",
        "It has no required reviewers",
        "Workflow code is therefore defense in depth, not the authority",
        "protected_branches: false",
        "custom_branch_policies: true",
        "deployment-branch-policies",
    ):
        assert required in normalized_readme
    for retired in (
        "Do not store or use `COWORK_TEST_API_KEY` yet",
        "nonempty required-reviewer rule",
        "required-reviewer gate",
    ):
        assert retired not in normalized_readme


def test_all_permanent_environments_fail_on_missing_prerequisites() -> None:
    """Production must not turn nine skipped post-deploy tests into green CI."""
    assert "dev|staging|prod) enforce=true ;;" in INTEGRATION


def test_ci_identities_come_from_the_target_environment_never_the_provisioner() -> None:
    """Staging reads a standing user, prod the guarded standing identity.

    Neither calls auth's internal provisioner, which mutates the fixed
    emailsink user and needs cluster DNS a hosted runner does not have.
    """
    assert '[[ "$DEPLOY_ENV" == "prod" ]]' in INTEGRATION
    assert "COWORK_TEST_IDENTITY_MODE=standing" in INTEGRATION
    assert "TEST_USER_PROVISION" not in INTEGRATION
    assert _suite_step("Run integration tests") == {
        "name": "Run integration tests",
        "env": {
            "COWORK_TEST_API_KEY": "${{ secrets.COWORK_TEST_API_KEY }}",
            "COWORK_TEST_USER_ID": "${{ vars.COWORK_TEST_USER_ID }}",
            "COWORK_TEST_USER_EMAIL": "${{ vars.COWORK_TEST_USER_EMAIL }}",
            "COWORK_TEST_ORG_ID": "${{ vars.COWORK_TEST_ORG_ID }}",
        },
        "run": "make test/integration",
    }
    # Values resolve from the Environment the job names, one per target.
    assert SUITE_JOB["environment"] == {"name": "${{ inputs.deploy-env }}"}


def test_staging_identity_is_taken_as_given_without_a_network_call(
    monkeypatch,
) -> None:
    """Identity source 1: no provisioning, no auth lookup, no mint."""
    monkeypatch.setenv("COWORK_BASE_URL", "https://cowork.staging.mindshub.ai")
    monkeypatch.delenv("COWORK_TEST_IDENTITY_MODE", raising=False)
    monkeypatch.setenv("COWORK_TEST_API_KEY", "mdb_staging.standing-key")
    monkeypatch.setenv("COWORK_TEST_USER_ID", "staging-user-id")
    monkeypatch.setenv("COWORK_TEST_ORG_ID", "staging-org-id")
    monkeypatch.delenv("COWORK_TEST_USER_EMAIL", raising=False)
    monkeypatch.setenv("TEST_USER_MINT_URL", "https://auth.example/dev/mint-test-user/")
    monkeypatch.setattr(
        test_post_deploy.httpx,
        "post",
        lambda *_args, **_kwargs: pytest.fail("staging minted or provisioned a user"),
    )
    monkeypatch.setattr(
        test_post_deploy.httpx,
        "get",
        lambda *_args, **_kwargs: pytest.fail("staging looked its identity up"),
    )

    identity = test_post_deploy._provision_identity()

    assert identity == {
        "api_key": "mdb_staging.standing-key",
        "user_id": "staging-user-id",
        "organization_id": "staging-org-id",
        "email": "postdeploy@example.com",
    }


def _run_identity_preflight(deploy_env: str, inputs: dict[str, str]):
    step = _suite_step("Check the standing-identity inputs")
    env = {"PATH": "/usr/bin:/bin", "DEPLOY_ENV": deploy_env, **inputs}
    # The callers assert the exit status, so a failing preflight must not raise.
    return subprocess.run(
        ["bash", "-c", step["run"]],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_identity_inputs_are_named_once_before_the_suite_runs() -> None:
    """A missing credential is named once, not once per test.

    Without this the first signal is seven identical session-fixture errors,
    which reads as a broken suite rather than as unset configuration. That is
    how prod run 34790453832 reported it.
    """
    step = _suite_step("Check the standing-identity inputs")
    assert step["if"] == "${{ !startsWith(inputs.deploy-env, 'pr-') }}"
    assert step["env"] == {
        "DEPLOY_ENV": "${{ inputs.deploy-env }}",
        "COWORK_TEST_API_KEY": "${{ secrets.COWORK_TEST_API_KEY }}",
        "COWORK_TEST_USER_ID": "${{ vars.COWORK_TEST_USER_ID }}",
        "COWORK_TEST_USER_EMAIL": "${{ vars.COWORK_TEST_USER_EMAIL }}",
        "COWORK_TEST_ORG_ID": "${{ vars.COWORK_TEST_ORG_ID }}",
    }
    steps = [each.get("name") for each in SUITE_JOB["steps"]]
    assert steps.index(step["name"]) < steps.index("Run integration tests")


@pytest.mark.parametrize(
    ("deploy_env", "inputs", "missing"),
    [
        (
            "staging",
            {
                "COWORK_TEST_API_KEY": "mdb_never-printed",
                "COWORK_TEST_USER_EMAIL": "never-printed@mindshub.ai",
            },
            "COWORK_TEST_USER_ID, COWORK_TEST_ORG_ID",
        ),
        (
            "prod",
            {
                "COWORK_TEST_USER_ID": "never-printed-user-id",
                "COWORK_TEST_ORG_ID": "never-printed-org-id",
            },
            "COWORK_TEST_API_KEY, COWORK_TEST_USER_EMAIL",
        ),
    ],
)
def test_the_preflight_names_exactly_what_the_target_is_missing(
    deploy_env, inputs, missing
) -> None:
    result = _run_identity_preflight(deploy_env, inputs)

    assert result.returncode == 1
    assert (
        f"::error::Required {deploy_env} integration inputs are empty: {missing}."
        in result.stdout
    )
    # The values themselves are never echoed.
    for value in inputs.values():
        assert value not in result.stdout


@pytest.mark.parametrize(
    ("deploy_env", "inputs"),
    [
        (
            "staging",
            {
                "COWORK_TEST_API_KEY": "k",
                "COWORK_TEST_USER_ID": "u",
                "COWORK_TEST_ORG_ID": "o",
            },
        ),
        (
            "prod",
            {
                "COWORK_TEST_API_KEY": "k",
                "COWORK_TEST_USER_EMAIL": "e@mindshub.ai",
                "COWORK_TEST_ORG_ID": "o",
            },
        ),
    ],
)
def test_the_preflight_passes_with_the_targets_own_inputs(deploy_env, inputs) -> None:
    result = _run_identity_preflight(deploy_env, inputs)

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"All required {deploy_env} integration inputs are present." in result.stdout


def test_standing_identity_mode_fails_before_a_configured_provisioner(
    monkeypatch,
) -> None:
    monkeypatch.setenv("COWORK_REQUIRE_INTEGRATION", "true")
    monkeypatch.setenv("COWORK_TEST_IDENTITY_MODE", "standing")
    monkeypatch.delenv("COWORK_TEST_API_KEY", raising=False)
    monkeypatch.delenv("COWORK_TEST_USER_EMAIL", raising=False)
    monkeypatch.delenv("COWORK_TEST_ORG_ID", raising=False)
    monkeypatch.setenv(
        "TEST_USER_PROVISION_URL",
        "http://auth.prod.svc.cluster.local/v1/internal/test-users/",
    )
    monkeypatch.setenv("TEST_USER_PROVISION_SECRET", "must-not-be-sent")
    monkeypatch.setattr(
        test_post_deploy.httpx,
        "post",
        lambda *_args, **_kwargs: pytest.fail("prod fell back to provisioning"),
    )
    monkeypatch.setattr(
        test_post_deploy.httpx,
        "get",
        lambda *_args, **_kwargs: pytest.fail("incomplete identity reached auth"),
    )

    with pytest.raises(pytest.fail.Exception, match="standing requires"):
        test_post_deploy._provision_identity()


@pytest.mark.parametrize(
    "base_url",
    [
        "https://attacker.example",
        "http://cowork.mindshub.ai",
        "https://cowork.mindshub.ai.evil.example",
        "https://cowork.mindshub.ai/other",
    ],
)
def test_prod_standing_identity_rejects_a_different_target_before_network(
    monkeypatch, base_url
) -> None:
    monkeypatch.setenv("COWORK_REQUIRE_INTEGRATION", "true")
    monkeypatch.setenv("COWORK_TEST_IDENTITY_MODE", "standing")
    monkeypatch.setenv("COWORK_BASE_URL", base_url)
    monkeypatch.setenv("COWORK_TEST_API_KEY", "mdb_deadbeef.must-not-be-sent")
    monkeypatch.setenv("COWORK_TEST_USER_EMAIL", "cowork-ci@mindshub.ai")
    monkeypatch.setenv("COWORK_TEST_ORG_ID", "org-id")
    monkeypatch.setattr(
        test_post_deploy.httpx,
        "get",
        lambda *_args, **_kwargs: pytest.fail(
            "network reached before production target validation"
        ),
    )

    with pytest.raises(pytest.fail.Exception, match="refusing to send"):
        test_post_deploy._provision_identity()


@pytest.mark.parametrize(
    "email",
    [
        "cowork-postdeploy@emailsink.dev",
        "cowork-ci@mindsdb.com",
        "attacker@example.com",
    ],
)
def test_prod_standing_identity_rejects_an_uncontrolled_email_before_network(
    monkeypatch, email
) -> None:
    monkeypatch.setenv("COWORK_REQUIRE_INTEGRATION", "true")
    monkeypatch.setenv("COWORK_TEST_IDENTITY_MODE", "standing")
    monkeypatch.setenv("COWORK_TEST_API_KEY", "mdb_deadbeef.must-not-be-sent")
    monkeypatch.setenv("COWORK_TEST_USER_EMAIL", email)
    monkeypatch.setenv("COWORK_TEST_ORG_ID", "org-id")
    monkeypatch.setattr(
        test_post_deploy.httpx,
        "get",
        lambda *_args, **_kwargs: pytest.fail(
            "network reached before configured-email validation"
        ),
    )

    with pytest.raises(pytest.fail.Exception) as failure:
        test_post_deploy._provision_identity()
    assert str(failure.value) == (
        "COWORK_TEST_USER_EMAIL must use a controlled, non-staff @mindshub.ai "
        "account; @emailsink.dev and the staff @mindsdb.com domain are not "
        "permitted in prod. COWORK_REQUIRE_INTEGRATION is set, so this "
        "environment is supposed to have it and the absence is a defect rather "
        "than a skip."
    )


def test_prod_standing_identity_is_resolved_live_without_a_mutating_post(
    monkeypatch,
) -> None:
    monkeypatch.setenv("COWORK_REQUIRE_INTEGRATION", "true")
    monkeypatch.setenv("COWORK_TEST_IDENTITY_MODE", "standing")
    monkeypatch.setenv("COWORK_TEST_API_KEY", "mdb_deadbeef.dedicated-key")
    monkeypatch.setenv("COWORK_TEST_USER_EMAIL", "cowork-ci@mindshub.ai")
    monkeypatch.setenv("COWORK_TEST_ORG_ID", "org-id")
    monkeypatch.setenv(
        "TEST_USER_PROVISION_URL",
        "http://auth.prod.svc.cluster.local/v1/internal/test-users/",
    )
    monkeypatch.setenv("TEST_USER_PROVISION_SECRET", "must-not-be-sent")
    calls = []

    class Response:
        status_code = 200
        headers = {"X-Billing-Segment": "free"}

        @staticmethod
        def json():
            return {
                "valid": True,
                "auth_method": "api_key",
                "key_type": "user",
                "key_prefix": "mdb_deadbeef",
                "email": "cowork-ci@mindshub.ai",
                "user_id": "user-id",
                "organization_id": "org-id",
                "entitlements": {"permissions": {"admin": {"hub": False}}},
            }

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(test_post_deploy.httpx, "get", get)
    monkeypatch.setattr(
        test_post_deploy.httpx,
        "post",
        lambda *_args, **_kwargs: pytest.fail("prod fell back to provisioning"),
    )

    identity = test_post_deploy._provision_identity()

    assert identity == {
        "api_key": "mdb_deadbeef.dedicated-key",
        "email": "cowork-ci@mindshub.ai",
        "user_id": "user-id",
        "organization_id": "org-id",
    }
    assert len(calls) == 1
    assert calls[0][0] == test_post_deploy.PROD_AUTHENTICATE_URL
    assert calls[0][1]["follow_redirects"] is False
    assert (
        calls[0][1]["headers"]["Authorization"] == "Bearer mdb_deadbeef.dedicated-key"
    )


@pytest.mark.parametrize(
    ("hub_admin", "billing_segment", "message"),
    [
        (True, "free", "Hub admin"),
        (False, "employee", "non-employee"),
    ],
)
def test_prod_standing_identity_rejects_privileged_principal(
    monkeypatch,
    hub_admin,
    billing_segment,
    message,
) -> None:
    monkeypatch.setenv("COWORK_REQUIRE_INTEGRATION", "true")
    monkeypatch.setenv("COWORK_TEST_IDENTITY_MODE", "standing")
    monkeypatch.setenv("COWORK_TEST_API_KEY", "mdb_deadbeef.dedicated-key")
    monkeypatch.setenv("COWORK_TEST_USER_EMAIL", "cowork-ci@mindshub.ai")
    monkeypatch.setenv("COWORK_TEST_ORG_ID", "org-id")

    class Response:
        status_code = 200
        headers = {"X-Billing-Segment": billing_segment}

        @staticmethod
        def json():
            return {
                "valid": True,
                "auth_method": "api_key",
                "key_type": "user",
                "key_prefix": "mdb_deadbeef",
                "email": "cowork-ci@mindshub.ai",
                "user_id": "user-id",
                "organization_id": "org-id",
                "entitlements": {"permissions": {"admin": {"hub": hub_admin}}},
            }

    monkeypatch.setattr(
        test_post_deploy.httpx, "get", lambda *_args, **_kwargs: Response()
    )
    monkeypatch.setattr(
        test_post_deploy.httpx,
        "post",
        lambda *_args, **_kwargs: pytest.fail("prod fell back to provisioning"),
    )

    with pytest.raises(pytest.fail.Exception, match=message):
        test_post_deploy._provision_identity()


@pytest.mark.parametrize(
    ("auth_method", "key_type", "key_prefix"),
    [
        ("api_key", None, "mdb_deadbeef"),
        ("api_key", "instance", "mdb_deadbeef"),
        ("api_key", "turn", "mdb_deadbeef"),
        ("jwt", "user", None),
        ("api_key", "user", None),
        ("api_key", "user", "mdb_different"),
    ],
)
def test_prod_standing_identity_rejects_a_non_user_api_key(
    monkeypatch, auth_method, key_type, key_prefix
) -> None:
    monkeypatch.setenv("COWORK_REQUIRE_INTEGRATION", "true")
    monkeypatch.setenv("COWORK_TEST_IDENTITY_MODE", "standing")
    monkeypatch.setenv("COWORK_TEST_API_KEY", "mdb_deadbeef.dedicated-key")
    monkeypatch.setenv("COWORK_TEST_USER_EMAIL", "cowork-ci@mindshub.ai")
    monkeypatch.setenv("COWORK_TEST_ORG_ID", "org-id")

    class Response:
        status_code = 200
        headers = {"X-Billing-Segment": "free"}

        @staticmethod
        def json():
            return {
                "valid": True,
                "auth_method": auth_method,
                "key_type": key_type,
                "key_prefix": key_prefix,
                "email": "cowork-ci@mindshub.ai",
                "user_id": "user-id",
                "organization_id": "org-id",
                "entitlements": {"permissions": {"admin": {"hub": False}}},
            }

    monkeypatch.setattr(
        test_post_deploy.httpx,
        "get",
        lambda *_args, **_kwargs: Response(),
    )

    with pytest.raises(pytest.fail.Exception, match="standing user API key"):
        test_post_deploy._provision_identity()


def test_prod_standing_identity_rejects_a_non_mindsdb_key_before_network(
    monkeypatch,
) -> None:
    monkeypatch.setenv("COWORK_REQUIRE_INTEGRATION", "true")
    monkeypatch.setenv("COWORK_TEST_IDENTITY_MODE", "standing")
    monkeypatch.setenv("COWORK_TEST_API_KEY", "not-a-mindsdb-key")
    monkeypatch.setenv("COWORK_TEST_USER_EMAIL", "cowork-ci@mindshub.ai")
    monkeypatch.setenv("COWORK_TEST_ORG_ID", "org-id")
    monkeypatch.setattr(
        test_post_deploy.httpx,
        "get",
        lambda *_args, **_kwargs: pytest.fail("invalid key reached auth"),
    )

    with pytest.raises(pytest.fail.Exception, match="beginning with mdb_"):
        test_post_deploy._provision_identity()


def test_prod_standing_identity_rejects_a_different_organization(monkeypatch) -> None:
    monkeypatch.setenv("COWORK_REQUIRE_INTEGRATION", "true")
    monkeypatch.setenv("COWORK_TEST_IDENTITY_MODE", "standing")
    monkeypatch.setenv("COWORK_TEST_API_KEY", "mdb_deadbeef.dedicated-key")
    monkeypatch.setenv("COWORK_TEST_USER_EMAIL", "cowork-ci@mindshub.ai")
    monkeypatch.setenv("COWORK_TEST_ORG_ID", "dedicated-org-id")

    class Response:
        status_code = 200
        headers = {"X-Billing-Segment": "free"}

        @staticmethod
        def json():
            return {
                "valid": True,
                "auth_method": "api_key",
                "key_type": "user",
                "key_prefix": "mdb_deadbeef",
                "email": "cowork-ci@mindshub.ai",
                "user_id": "user-id",
                "organization_id": "different-live-org-id",
                "entitlements": {"permissions": {"admin": {"hub": False}}},
            }

    monkeypatch.setattr(
        test_post_deploy.httpx,
        "get",
        lambda *_args, **_kwargs: Response(),
    )

    with pytest.raises(pytest.fail.Exception, match="different organization"):
        test_post_deploy._provision_identity()
