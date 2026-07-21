import pytest
from pydantic import ValidationError

from cowork.common.settings.app_settings import AppSettings


def test_app_settings_ignores_generic_server_port(monkeypatch):
    monkeypatch.delenv("COWORK_LISTEN_PORT", raising=False)
    monkeypatch.setenv("SERVER_PORT", "invalid")

    settings = AppSettings(_env_file=None)

    assert settings.port == 26866


def test_app_settings_reads_cowork_listen_port(monkeypatch):
    monkeypatch.setenv("COWORK_LISTEN_PORT", "9999")
    monkeypatch.setenv("SERVER_PORT", "invalid")

    settings = AppSettings(_env_file=None)

    assert settings.port == 9999


def test_app_settings_rejects_invalid_cowork_listen_port(monkeypatch):
    monkeypatch.setenv("COWORK_LISTEN_PORT", "invalid")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_app_settings_ignores_generic_server_host(monkeypatch):
    monkeypatch.delenv("COWORK_SERVER_HOST", raising=False)
    monkeypatch.setenv("SERVER_HOST", "0.0.0.0")

    settings = AppSettings(_env_file=None)

    assert settings.host == "127.0.0.1"


def test_app_settings_tenancy_mode_defaults_to_local(monkeypatch):
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)

    settings = AppSettings(_env_file=None)

    assert settings.tenancy_mode == "local"


def test_app_settings_reads_tenancy_mode_org(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")

    settings = AppSettings(_env_file=None)

    assert settings.tenancy_mode == "org"


def test_app_settings_rejects_invalid_tenancy_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "multi")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_hermes_hidden_from_harness_options_in_org_mode(monkeypatch):
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.common.settings.user_settings import _harness_options
    import cowork.harnesses.anton_harness.harness  # noqa: F401  register anton
    import cowork.harnesses.hermes_harness.harness  # noqa: F401  register hermes

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    try:
        options = _harness_options()
        assert "anton" in options
        assert "hermes" not in options
    finally:
        get_app_settings.cache_clear()


def test_hermes_available_in_local_mode(monkeypatch):
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.common.settings.user_settings import _harness_options
    import cowork.harnesses.anton_harness.harness  # noqa: F401
    import cowork.harnesses.hermes_harness.harness  # noqa: F401

    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    get_app_settings.cache_clear()
    try:
        options = _harness_options()
        assert "anton" in options
        assert "hermes" in options
    finally:
        get_app_settings.cache_clear()


def test_app_settings_identity_enforce_defaults_to_audit(monkeypatch):
    monkeypatch.delenv("COWORK_IDENTITY_ENFORCE", raising=False)

    settings = AppSettings(_env_file=None)

    assert settings.identity_enforce == "audit"


def test_app_settings_rejects_invalid_identity_enforce(monkeypatch):
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "strict")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)
