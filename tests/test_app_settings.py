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


def test_app_settings_reads_desktop_cowork_server_port(monkeypatch):
    # The desktop app hands the derived per-user port to the sidecar as
    # COWORK_SERVER_PORT (ENG-439). Regression: dropping this alias made the
    # server bind :26866 while the app health-polled the derived port.
    monkeypatch.delenv("COWORK_LISTEN_PORT", raising=False)
    monkeypatch.setenv("COWORK_SERVER_PORT", "27735")

    settings = AppSettings(_env_file=None)

    assert settings.port == 27735


def test_app_settings_listen_port_wins_over_server_port(monkeypatch):
    monkeypatch.setenv("COWORK_LISTEN_PORT", "9999")
    monkeypatch.setenv("COWORK_SERVER_PORT", "27735")

    settings = AppSettings(_env_file=None)

    assert settings.port == 9999


def test_app_settings_ignores_k8s_injected_server_port_uri(monkeypatch):
    # K8s auto-injects COWORK_SERVER_PORT=tcp://<ip>:<port> on any pod
    # colocated with a `cowork-server` Service — must fall back to the
    # default, not fail int parsing.
    monkeypatch.delenv("COWORK_LISTEN_PORT", raising=False)
    monkeypatch.setenv("COWORK_SERVER_PORT", "tcp://10.0.0.5:26866")

    settings = AppSettings(_env_file=None)

    assert settings.port == 26866


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
