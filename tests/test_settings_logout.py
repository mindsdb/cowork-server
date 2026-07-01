"""Regression tests for the logout / clear_credentials flow (ENG-475).

Verifies that:
- POST /api/v1/settings/logout clears all credential keys from the DB
- /health returns config_ready: false after logout
- Provider/model preferences survive logout
- Provider UI/status state is cleared
"""
from __future__ import annotations

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.services.settings import SettingService
from cowork.common.settings.user_settings import UserSettings


@pytest.fixture()
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def _seed_all_settings(session: Session) -> None:
    """Seed credentials, provider prefs, and UI state into the DB."""
    svc = SettingService(session)

    # Credentials (sensitive fields)
    svc.upsert_setting("minds_api_key", "mdb_test_key_123")
    svc.upsert_setting("anthropic_api_key", "sk-ant-test-123")
    svc.upsert_setting("openai_api_key", "sk-test-123")
    svc.upsert_setting("gemini_api_key", "gemini-test-123")
    svc.upsert_setting("openai_compatible_api_key", "oc-test-123")

    # URLs cleared by logout
    svc.upsert_setting("minds_url", "https://api.mindshub.ai")
    svc.upsert_setting("openai_base_url", "https://api.openai.com")

    # Provider/model preferences (should survive logout)
    svc.upsert_setting("planning_provider", "minds_cloud")
    svc.upsert_setting("coding_provider", "minds_cloud")
    svc.upsert_setting("planning_model", "latest:sonnet")
    svc.upsert_setting("coding_model", "latest:haiku")

    # Provider UI state (should be cleared)
    svc.upsert_setting("providers_json", '[{"type":"minds-cloud"}]')
    svc.upsert_setting("provider_status", '{"minds-cloud":"ok"}')
    svc.upsert_setting("provider_status_details", '{"minds-cloud":"connected"}')


def _cleanup(session: Session) -> None:
    """Remove all seeded keys so tests don't leak into each other."""
    svc = SettingService(session)
    for key in (
        "minds_api_key", "anthropic_api_key", "openai_api_key",
        "gemini_api_key", "openai_compatible_api_key",
        "minds_url", "openai_base_url",
        "planning_provider", "coding_provider",
        "planning_model", "coding_model",
        "providers_json", "provider_status", "provider_status_details",
    ):
        svc.delete_setting(key)


def test_clear_credentials_removes_sensitive_keys(session: Session):
    """All sensitive (SecretStr) fields must be deleted."""
    _seed_all_settings(session)
    svc = SettingService(session)
    try:
        deleted = svc.clear_credentials()

        # All API key fields should be in the deleted list
        for key in ("minds_api_key", "anthropic_api_key", "openai_api_key",
                    "gemini_api_key", "openai_compatible_api_key"):
            assert key in deleted, f"Expected '{key}' to be deleted"

        # Verify they're actually gone from the DB
        settings = svc.load()
        assert settings.minds_api_key is None
        assert settings.anthropic_api_key is None
        assert settings.openai_api_key is None
        assert settings.gemini_api_key is None
        assert settings.openai_compatible_api_key is None
    finally:
        _cleanup(session)


def test_clear_credentials_removes_provider_ui_state(session: Session):
    """Provider connectivity and UI card state must be cleared."""
    _seed_all_settings(session)
    svc = SettingService(session)
    try:
        deleted = svc.clear_credentials()

        for key in ("providers_json", "provider_status", "provider_status_details",
                    "minds_url", "openai_base_url"):
            assert key in deleted, f"Expected '{key}' to be deleted"
    finally:
        _cleanup(session)


def test_clear_credentials_preserves_model_preferences(session: Session):
    """Provider/model preferences must survive logout."""
    _seed_all_settings(session)
    svc = SettingService(session)
    try:
        svc.clear_credentials()

        settings = svc.load()
        assert settings.planning_provider.value == "minds_cloud"
        assert settings.coding_provider.value == "minds_cloud"
        assert settings.planning_model == "latest:sonnet"
        assert settings.coding_model == "latest:haiku"
    finally:
        _cleanup(session)


def test_config_ready_false_after_logout(session: Session):
    """After clearing credentials, config_status must report not ready."""
    _seed_all_settings(session)
    svc = SettingService(session)
    try:
        # Before logout, should be configured
        settings_before = svc.load()
        assert settings_before.config_status["config_ready"] is True

        svc.clear_credentials()

        # After logout, should not be configured
        settings_after = svc.load()
        assert settings_after.config_status["config_ready"] is False
    finally:
        _cleanup(session)


def test_clear_credentials_idempotent(session: Session):
    """Calling clear_credentials twice should not error."""
    _seed_all_settings(session)
    svc = SettingService(session)
    try:
        first = svc.clear_credentials()
        assert len(first) > 0

        second = svc.clear_credentials()
        assert len(second) == 0
    finally:
        _cleanup(session)
