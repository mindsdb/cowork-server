"""The /health endpoint echoes the per-install owner token (ENG-439).

The desktop app only adopts a running server whose /health `owner` matches
the token it generated, so it can never drive another OS user's sidecar on a
shared loopback port.
"""

from types import SimpleNamespace
from unittest.mock import patch


def test_health_reports_owner_from_app_settings():
    from cowork.api.v1.endpoints import health

    with (
        patch.object(health, "get_app_settings", return_value=SimpleNamespace(owner="install-token-abc", tenancy_mode="local")),
        patch.object(health, "get_user_settings", return_value=SimpleNamespace(config_status={})),
    ):
        body = health.health()

    assert body["owner"] == "install-token-abc"
    assert body["status"] == "ok"


def test_health_owner_empty_when_unset():
    from cowork.api.v1.endpoints import health

    with (
        patch.object(health, "get_app_settings", return_value=SimpleNamespace(owner="", tenancy_mode="local")),
        patch.object(health, "get_user_settings", return_value=SimpleNamespace(config_status={})),
    ):
        body = health.health()

    # Empty owner = server advertises no identity → the app must not adopt it.
    assert body["owner"] == ""


def _health_with(tenancy_mode: str) -> dict:
    from cowork.api.v1.endpoints import health

    with (
        patch.object(
            health, "get_app_settings", return_value=SimpleNamespace(owner="", tenancy_mode=tenancy_mode)
        ),
        patch.object(health, "get_user_settings", return_value=SimpleNamespace(config_status={})),
    ):
        return health.health()


def test_health_reports_org_mode():
    """The client can't tell a hosted org deployment from an authenticated
    standalone one from ``config_ready`` alone — that says the deployment can
    run, not who may configure it. Onboarding needs the difference: in org mode
    provider config is admin-owned, so finalizing by writing it 403s a member.
    """
    assert _health_with("org")["org_mode"] is True


def test_health_org_mode_false_on_desktop():
    assert _health_with("local")["org_mode"] is False
