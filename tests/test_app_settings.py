import pytest
from pydantic import ValidationError

from cowork.common.settings.app_settings import AppSettings, OAuthSettings


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


def test_server_origin_does_not_leak_into_public_base_url(monkeypatch):
    """COWORK_SERVER_ORIGIN feeds OAuth redirect URIs only — it must NOT also
    populate public_base_url. The desktop app sets it to the loopback origin so
    OAuth "Allow" lands on the live port; if that also set public_base_url,
    channels/ingress.py would treat the server as publicly reachable and stop
    every polling adapter (Telegram) in favour of a nonexistent webhook. (ENG-632)
    """
    monkeypatch.delenv("COWORK_PUBLIC_BASE_URL", raising=False)
    monkeypatch.setenv("COWORK_SERVER_ORIGIN", "http://127.0.0.1:51234")

    oauth = OAuthSettings(_env_file=None)
    app = AppSettings(_env_file=None)

    # OAuth origin picks it up (redirect URIs must carry the live port)...
    assert oauth.server_origin == "http://127.0.0.1:51234"
    # ...but the webhook base URL stays empty (server not publicly reachable).
    assert app.public_base_url == ""


def test_public_base_url_still_reads_its_own_env(monkeypatch):
    monkeypatch.delenv("COWORK_SERVER_ORIGIN", raising=False)
    monkeypatch.setenv("COWORK_PUBLIC_BASE_URL", "https://hooks.example.com")

    settings = AppSettings(_env_file=None)

    assert settings.public_base_url == "https://hooks.example.com"
