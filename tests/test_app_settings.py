import pytest
from pydantic import ValidationError

from cowork.common.settings.app_settings import AppSettings


def test_app_settings_ignores_generic_server_port(monkeypatch):
    monkeypatch.delenv("COWORK_SERVER_PORT", raising=False)
    monkeypatch.setenv("SERVER_PORT", "invalid")

    settings = AppSettings(_env_file=None)

    assert settings.port == 26866


def test_app_settings_reads_cowork_server_port(monkeypatch):
    monkeypatch.setenv("COWORK_SERVER_PORT", "9999")
    monkeypatch.setenv("SERVER_PORT", "invalid")

    settings = AppSettings(_env_file=None)

    assert settings.port == 9999


def test_app_settings_rejects_invalid_cowork_server_port(monkeypatch):
    monkeypatch.setenv("COWORK_SERVER_PORT", "invalid")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)


def test_app_settings_ignores_generic_server_host(monkeypatch):
    monkeypatch.delenv("COWORK_SERVER_HOST", raising=False)
    monkeypatch.setenv("SERVER_HOST", "0.0.0.0")

    settings = AppSettings(_env_file=None)

    assert settings.host == "127.0.0.1"
