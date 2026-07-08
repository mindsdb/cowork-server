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
        patch.object(health, "get_app_settings", return_value=SimpleNamespace(owner="install-token-abc")),
        patch.object(health, "get_user_settings", return_value=SimpleNamespace(config_status={})),
    ):
        body = health.health()

    assert body["owner"] == "install-token-abc"
    assert body["status"] == "ok"


def test_health_owner_empty_when_unset():
    from cowork.api.v1.endpoints import health

    with (
        patch.object(health, "get_app_settings", return_value=SimpleNamespace(owner="")),
        patch.object(health, "get_user_settings", return_value=SimpleNamespace(config_status={})),
    ):
        body = health.health()

    # Empty owner = server advertises no identity → the app must not adopt it.
    assert body["owner"] == ""
