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


def test_app_settings_reads_legacy_cowork_server_port(monkeypatch):
    # The desktop app spawns the sidecar with COWORK_SERVER_PORT; shipped
    # Electron builds cannot be hot-updated, so the legacy name must work.
    monkeypatch.delenv("COWORK_LISTEN_PORT", raising=False)
    monkeypatch.setenv("COWORK_SERVER_PORT", "9999")

    settings = AppSettings(_env_file=None)

    assert settings.port == 9999


def test_app_settings_listen_port_wins_over_legacy_alias(monkeypatch):
    monkeypatch.setenv("COWORK_LISTEN_PORT", "8888")
    monkeypatch.setenv("COWORK_SERVER_PORT", "9999")

    settings = AppSettings(_env_file=None)

    assert settings.port == 8888


def test_app_settings_ignores_k8s_service_link_port(monkeypatch):
    # K8s injects COWORK_SERVER_PORT=tcp://<ip>:<port> on pods colocated
    # with a `cowork-server` Service; it must not break startup.
    monkeypatch.delenv("COWORK_LISTEN_PORT", raising=False)
    monkeypatch.setenv("COWORK_SERVER_PORT", "tcp://10.3.0.12:26866")

    settings = AppSettings(_env_file=None)

    assert settings.port == 26866


def test_app_settings_listen_port_wins_over_service_link_uri(monkeypatch):
    monkeypatch.setenv("COWORK_LISTEN_PORT", "9010")
    monkeypatch.setenv("COWORK_SERVER_PORT", "tcp://10.3.0.12:26866")

    settings = AppSettings(_env_file=None)

    assert settings.port == 9010


def test_app_settings_rejects_invalid_legacy_server_port(monkeypatch):
    monkeypatch.delenv("COWORK_LISTEN_PORT", raising=False)
    monkeypatch.setenv("COWORK_SERVER_PORT", "invalid")

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


def test_app_settings_identity_enforce_defaults_to_audit(monkeypatch):
    monkeypatch.delenv("COWORK_IDENTITY_ENFORCE", raising=False)

    settings = AppSettings(_env_file=None)

    assert settings.identity_enforce == "audit"


def test_app_settings_rejects_invalid_identity_enforce(monkeypatch):
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "strict")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)
