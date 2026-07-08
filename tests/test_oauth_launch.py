from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.handlers.probe import ProbeHandler
from cowork.schemas.connectors import OAuthStartResponse
from cowork.services.connectors.submissions import store as submission_store


def _seed_submission(
    connector_id: str = "gmail",
    method: str = "browser_oauth_builtin",
    values: dict | None = None,
) -> str:
    return submission_store.stage(
        form_id=f"{connector_id}-connector",
        connector_id=connector_id,
        conversation_id=None,
        values=values or {},
        skipped=[],
        form_spec=None,
    )


async def _collect(
    handler: ProbeHandler,
    submission_id: str,
    connector_id: str = "gmail",
) -> list[dict]:
    events: list[dict] = []
    async for event_str in handler.run(
        submission_id=submission_id,
        connector_id=connector_id,
        method="browser_oauth_builtin",
        name="test-gmail",
        conversation_id=None,
    ):
        for line in event_str.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def _find_patch(events: list[dict]) -> dict | None:
    for e in events:
        t = e.get("type", "")
        d = e.get("delta", "")
        if "data-vault-form-patch" in d:
            start = d.index("```data-vault-form-patch\n") + len("```data-vault-form-patch\n")
            end = d.index("\n```", start)
            return json.loads(d[start:end])
    return None


def _find_completed(events: list[dict]) -> dict | None:
    for e in events:
        if e.get("type") == "response.completed":
            return e
    return None

def _find_last_patch(events: list[dict]) -> dict | None:
    result = None
    for e in events:
        d = e.get("delta", "")
        if "data-vault-form-patch" in d:
            start = d.index("```data-vault-form-patch\n") + len("```data-vault-form-patch\n")
            end = d.index("\n```", start)
            result = json.loads(d[start:end])
    return result


@pytest.mark.anyio
async def test_oauth_launch_missing_credentials():
    sub_id = _seed_submission()
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        handler = ProbeHandler(session)
        events = await _collect(handler, sub_id)

    found = _find_patch(events)
    assert found is not None
    err = found.get("form_error", "")
    assert "isn't unlocked yet" in err
    assert "Google sign-in" in err

    completed = _find_completed(events)
    assert completed is not None
    assert completed.get("response", {}).get("status") == "retry"


@pytest.mark.anyio
async def test_oauth_launch_with_creds_success():
    sub_id = _seed_submission(values={
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
    })

    fake_start = OAuthStartResponse(
        auth_url="https://accounts.google.com/o/oauth2/v2/auth?test=1",
        redirect_uri="http://127.0.0.1:26866/api/v1/connectors/oauth/gmail/callback",
        started_at="2026-01-01T00:00:00Z",
    )

    fake_state_data = {
        "gmail": {
            "pending": {},
            "lastSuccessAt": "2026-01-01T00:00:01Z",
            "lastError": "",
            "lastErrorAt": "",
            "connectionName": "test@gmail.com",
        }
    }

    with (
        patch("cowork.services.connectors.oauth.google.google_service.start", return_value=fake_start),
        patch("cowork.services.connectors.oauth.state.OAuthStateStore._load", return_value=fake_state_data),
    ):
        engine = get_engine(get_app_settings().database.uri)
        with Session(engine) as session:
            handler = ProbeHandler(session)
            events = await _collect(handler, sub_id)

    found = _find_patch(events)
    assert found is not None
    assert found.get("_oauth_url") == fake_start.auth_url
    assert found.get("_is_probing") is True

    completed = _find_completed(events)
    assert completed is not None
    assert completed.get("response", {}).get("status") == "success"


@pytest.mark.anyio
async def test_oauth_launch_with_creds_error():
    sub_id = _seed_submission(values={
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
    })

    fake_start = OAuthStartResponse(
        auth_url="https://accounts.google.com/o/oauth2/v2/auth?test=1",
        redirect_uri="http://127.0.0.1:26866/api/v1/connectors/oauth/gmail/callback",
        started_at="2026-01-01T00:00:00Z",
    )

    fake_state_data = {
        "gmail": {
            "pending": {},
            "lastSuccessAt": "",
            "lastError": "User cancelled the sign-in",
            "lastErrorAt": "2026-01-01T00:00:01Z",
        }
    }

    with (
        patch("cowork.services.connectors.oauth.google.google_service.start", return_value=fake_start),
        patch("cowork.services.connectors.oauth.state.OAuthStateStore._load", return_value=fake_state_data),
    ):
        engine = get_engine(get_app_settings().database.uri)
        with Session(engine) as session:
            handler = ProbeHandler(session)
            events = await _collect(handler, sub_id)

    # First patch should be the oauth_url (launch)
    found = _find_patch(events)
    assert found is not None
    assert found.get("_oauth_url") == fake_start.auth_url

    # Second patch (from polling) should carry the error
    last = _find_last_patch(events)
    assert last is not None
    assert "User cancelled" in last.get("form_error", "")

    completed = _find_completed(events)
    assert completed is not None
    assert completed.get("response", {}).get("status") == "retry"


@pytest.mark.anyio
async def test_oauth_launch_start_raises_400():
    sub_id = _seed_submission()

    with patch(
        "cowork.services.connectors.oauth.google.google_service.start",
        side_effect=HTTPException(status_code=400, detail="Google OAuth credentials are not configured."),
    ):
        engine = get_engine(get_app_settings().database.uri)
        with Session(engine) as session:
            handler = ProbeHandler(session)
            events = await _collect(handler, sub_id)

    found = _find_patch(events)
    assert found is not None
    err = found.get("form_error", "")
    assert "isn't unlocked yet" in err

    completed = _find_completed(events)
    assert completed is not None
    assert completed.get("response", {}).get("status") == "retry"
