"""Call sites outside the agent loop that must send a pinned desktop's turn key, not its session token."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import SecretStr

from cowork.common.settings import runtime_credential
from cowork.common.settings.user_settings import Provider, UserSettings


@pytest.fixture
def pinned(monkeypatch):
    """A desktop that sent its mounted organization and holds a key for it."""
    monkeypatch.setattr(runtime_credential, "_minds_credential", "session-jwt")
    monkeypatch.setattr(runtime_credential, "_organization_id", "org-a")
    monkeypatch.setattr(
        runtime_credential,
        "_inference_key",
        runtime_credential.InferenceKey(
            value="mdb_turn_a",
            organization_id="org-a",
            instance_id="desktop-1",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=50),
        ),
    )


def _settings():
    return UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("session-jwt"),
        minds_url="https://api.mindshub.ai",
    )


def test_code_mode_engines_get_the_turn_key(pinned):
    from cowork.api.v1.endpoints import coding

    assert coding._credentials(_settings()).minds_api_key == "mdb_turn_a"


def test_code_mode_model_levels_are_listed_with_the_turn_key(pinned, monkeypatch):
    from cowork.api.v1.endpoints import coding

    seen = {}
    monkeypatch.setattr(coding, "cached_minds_models", lambda url, *, api_key: seen.setdefault("key", api_key))

    coding._model_levels(_settings())

    assert seen["key"] == "mdb_turn_a"


def test_the_scratchpad_gets_the_turn_key(pinned, monkeypatch):
    from cowork.common.settings import user_settings
    from cowork.services import scratchpad_runtime

    monkeypatch.setattr(user_settings, "get_user_settings", lambda *_a, **_k: _settings())

    resolved = scratchpad_runtime._resolve_coding(
        coding_provider="minds-cloud", coding_model="m", coding_api_key="", coding_base_url=""
    )

    assert "mdb_turn_a" in resolved
    assert "session-jwt" not in resolved


def test_the_settings_test_button_pings_with_the_turn_key(pinned):
    from cowork.services.providers import resolve_stored_key

    assert resolve_stored_key(_settings(), "minds-cloud") == "mdb_turn_a"


def test_the_test_button_fails_when_a_pinned_desktop_has_no_live_key(pinned, monkeypatch):
    from cowork.services.providers import resolve_stored_key

    monkeypatch.setattr(runtime_credential, "_inference_key", None)

    assert resolve_stored_key(_settings(), "minds-cloud") == ""


def test_code_mode_says_reconnect_when_a_pinned_desktop_has_no_key(pinned, monkeypatch):
    monkeypatch.setattr(runtime_credential, "_inference_key", None)

    assert "Reconnect" in runtime_credential.missing_minds_key_message()


def test_code_mode_says_sign_in_when_signed_out(monkeypatch):
    monkeypatch.setattr(runtime_credential, "_minds_credential", None)
    monkeypatch.setattr(runtime_credential, "_organization_id", None)

    assert "Sign in" in runtime_credential.missing_minds_key_message()
