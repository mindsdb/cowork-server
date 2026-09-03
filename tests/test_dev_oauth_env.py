from pathlib import Path

from cowork.common.settings.app_settings import (
    OAuthSettings,
    _dev_oauth_env_path,
    _dev_oauth_settings_source,
)


def test_dev_oauth_file_is_absent_without_an_explicit_pointer(monkeypatch):
    monkeypatch.delenv("COWORK_DEV_OAUTH_ENV_FILE", raising=False)

    assert _dev_oauth_env_path() is None
    assert _dev_oauth_settings_source() == {}


def test_dev_oauth_file_accepts_only_the_fixed_developer_location(monkeypatch):
    expected = Path.home() / ".cowork-dev" / ".env"
    monkeypatch.setenv("COWORK_DEV_OAUTH_ENV_FILE", str(expected))

    assert _dev_oauth_env_path() == expected

    monkeypatch.setenv("COWORK_DEV_OAUTH_ENV_FILE", str(Path.home() / "other.env"))
    assert _dev_oauth_env_path() is None


def test_dev_oauth_source_admits_only_coding_connector_fields(monkeypatch):
    expected = Path.home() / ".cowork-dev" / ".env"
    monkeypatch.setenv("COWORK_DEV_OAUTH_ENV_FILE", str(expected))
    monkeypatch.setattr(Path, "is_file", lambda self: self == expected)
    monkeypatch.setattr(
        "cowork.common.settings.app_settings.dotenv_values",
        lambda _path: {
            "GITHUB_CLIENT_ID": "github-id",
            "GITHUB_CLIENT_SECRET": "github-secret",
            "LINEAR_CLIENT_ID": "linear-id",
            "LINEAR_CLIENT_SECRET": "linear-secret",
            "DATABASE_URI": "must-not-cross",
            "ANTON_MINDS_API_KEY": "must-not-cross",
        },
    )

    assert _dev_oauth_settings_source() == {
        "GITHUB_CLIENT_ID": "github-id",
        "GITHUB_CLIENT_SECRET": "github-secret",
        "LINEAR_CLIENT_ID": "linear-id",
        "LINEAR_CLIENT_SECRET": "linear-secret",
    }
    settings = OAuthSettings(_env_file=None)
    assert settings.github_client_id == "github-id"
    assert settings.linear_client_id == "linear-id"
