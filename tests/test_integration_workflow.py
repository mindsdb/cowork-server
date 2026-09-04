"""Contract tests for permanent-environment integration workflow wiring."""

from pathlib import Path

import pytest

from tests.integration import test_post_deploy


ROOT = Path(__file__).resolve().parents[1]
BUILD_DEPLOY = (ROOT / ".github/workflows/build-deploy.yml").read_text()
INTEGRATION = (ROOT / ".github/workflows/tests-integration.yml").read_text()
PUBLISH = (ROOT / ".github/workflows/publish.yml").read_text()


def test_integration_uses_callers_target_cluster_runner() -> None:
    """Both reusable-workflow calls pass through the deployment runner."""
    assert INTEGRATION.count("runs-on: ${{ inputs.runner }}") == 1
    assert BUILD_DEPLOY.count("runner: ${{ inputs.build-runner }}") == 2
    assert "build-runner: mdb-prod" in PUBLISH


def test_all_permanent_environments_fail_on_missing_prerequisites() -> None:
    """Production must not turn nine skipped post-deploy tests into green CI."""
    assert "dev|staging|prod) enforce=true ;;" in INTEGRATION


def test_prod_uses_a_standing_identity_without_provisioner_fallback() -> None:
    """The prod branch must never mutate the fixed emailsink user."""
    assert '[[ "$DEPLOY_ENV" == "prod" ]]' in INTEGRATION
    assert "COWORK_TEST_IDENTITY_MODE=standing" in INTEGRATION
    non_prod_step, prod_step = INTEGRATION.split(
        "- name: Run integration tests (non-prod)", 1
    )[1].split("- name: Run integration tests (prod)", 1)
    assert "TEST_USER_PROVISION_SECRET" in non_prod_step
    assert "COWORK_TEST_API_KEY" not in non_prod_step
    assert "TEST_USER_PROVISION_SECRET" not in prod_step
    assert "COWORK_TEST_API_KEY: ${{ secrets.COWORK_TEST_API_KEY }}" in prod_step
    assert "COWORK_TEST_USER_EMAIL: ${{ vars.COWORK_TEST_USER_EMAIL }}" in prod_step


def test_standing_identity_mode_fails_before_a_configured_provisioner(
    monkeypatch,
) -> None:
    monkeypatch.setenv("COWORK_REQUIRE_INTEGRATION", "true")
    monkeypatch.setenv("COWORK_TEST_IDENTITY_MODE", "standing")
    monkeypatch.delenv("COWORK_TEST_API_KEY", raising=False)
    monkeypatch.delenv("COWORK_TEST_USER_EMAIL", raising=False)
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
    monkeypatch.setenv("COWORK_TEST_API_KEY", "must-not-be-sent")
    monkeypatch.setenv("COWORK_TEST_USER_EMAIL", email)
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
    monkeypatch.setenv("COWORK_TEST_API_KEY", "dedicated-key")
    monkeypatch.setenv("COWORK_TEST_USER_EMAIL", "cowork-ci@mindshub.ai")
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
        "api_key": "dedicated-key",
        "email": "cowork-ci@mindshub.ai",
        "user_id": "user-id",
        "organization_id": "org-id",
    }
    assert len(calls) == 1
    assert calls[0][0] == test_post_deploy.PROD_AUTHENTICATE_URL
    assert calls[0][1]["follow_redirects"] is False
    assert calls[0][1]["headers"]["Authorization"] == "Bearer dedicated-key"


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
    monkeypatch.setenv("COWORK_TEST_API_KEY", "dedicated-key")
    monkeypatch.setenv("COWORK_TEST_USER_EMAIL", "cowork-ci@mindshub.ai")

    class Response:
        status_code = 200
        headers = {"X-Billing-Segment": billing_segment}

        @staticmethod
        def json():
            return {
                "valid": True,
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
