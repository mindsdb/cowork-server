"""Code mode's model listing keeps the session token; its inference bills the pinned turn key.

The gateway's /v1/models checks entitlements, which refuse turn keys.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from cowork.api.v1.endpoints import coding, coding_runtime
from cowork.common.settings import runtime_credential
from cowork.common.settings.user_settings import UserSettings


@pytest.fixture(autouse=True)
def pinned_desktop(monkeypatch):
    monkeypatch.setattr(runtime_credential, "_minds_credential", "session-jwt")
    monkeypatch.setattr(runtime_credential, "_organization_id", "org-a")
    monkeypatch.setattr(
        runtime_credential,
        "_inference_key",
        runtime_credential.InferenceKey(
            value="mdb_turn",
            organization_id="org-a",
            instance_id="desktop-1",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        ),
    )


def _settings():
    return UserSettings(minds_api_key="session-jwt", minds_url="https://api.example/v1")


def test_model_listing_sends_the_session_token():
    assert (
        coding._credentials(_settings(), model_listing=True).minds_api_key
        == "session-jwt"
    )


def test_inference_sends_the_turn_key():
    assert coding._credentials(_settings()).minds_api_key == "mdb_turn"


@pytest.mark.parametrize(
    ("path", "expected"), [("models", "session-jwt"), ("responses", "mdb_turn")]
)
def test_the_runtime_proxy_picks_the_key_by_path(monkeypatch, path, expected):
    sent = []

    async def upstream(request, path, credentials):
        sent.append(credentials.minds_api_key)

    monkeypatch.setattr(coding_runtime, "proxy_inference", upstream)
    monkeypatch.setattr(
        coding_runtime,
        "_control",
        lambda: SimpleNamespace(authenticate_run_token=lambda *a: None),
    )
    monkeypatch.setattr(coding_runtime, "_runtime_token", lambda request: "run-token")
    monkeypatch.setattr(
        coding_runtime,
        "SettingService",
        lambda session, scope: SimpleNamespace(load=_settings),
    )

    asyncio.run(
        coding_runtime.runtime_inference_proxy(
            "computer", "run", path, request=None, scope=None, session=None
        )
    )

    assert sent == [expected]
