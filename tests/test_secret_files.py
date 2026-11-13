"""Credentials read from the files the process environment names."""

from pathlib import Path

import pytest

from cowork.common.settings.app_settings import AppSettings, get_app_settings
from cowork.common.settings.secret_files import (
    SECRET_FILE_NAMES,
    SecretFileError,
    process_secret,
)
from cowork.common.settings.user_settings import UserSettings

URI = "postgresql+psycopg://cowork:from-the-file@postgres:5432/cowork"


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No credential or credential file in the environment, and cowork-home at the test's folder."""
    for name in SECRET_FILE_NAMES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(f"{name}_FILE", raising=False)
    for name in ("COWORK_DEPLOYMENT_KIND", "COWORK_TENANCY_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    get_app_settings.cache_clear()
    yield monkeypatch
    get_app_settings.cache_clear()


def _write(*, folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def _point(*, env, folder: Path, name: str, text: str) -> Path:
    path = _write(folder=folder, name=name, text=text)
    env.setenv(f"{name}_FILE", str(path))
    return path


def _all_four_from_files(*, env, folder: Path) -> None:
    _point(env=env, folder=folder, name="DATABASE_URI", text=URI + "\n")
    _point(env=env, folder=folder, name="COWORK_AUTH_TOKEN", text="token-from-the-file")
    _point(env=env, folder=folder, name="COWORK_PUBLISH_TOKEN", text="mdb_publish_from_the_file")
    _point(env=env, folder=folder, name="MINDS_API_KEY", text="mdb_publish_from_the_file")


def test_app_settings_read_each_credential_from_its_file(clean_env, tmp_path):
    _all_four_from_files(env=clean_env, folder=tmp_path / "run")

    settings = AppSettings(_env_file=None)

    assert settings.database.uri == URI
    assert settings.auth_token == "token-from-the-file"
    assert settings.publish_token.get_secret_value() == "mdb_publish_from_the_file"


def test_user_settings_read_the_minds_key_from_its_file(clean_env, tmp_path):
    _point(env=clean_env, folder=tmp_path / "run", name="MINDS_API_KEY", text="key-from-the-file\n")

    assert UserSettings(_env_file=None).minds_api_key.get_secret_value() == "key-from-the-file"


def test_org_user_settings_read_no_credential_file(clean_env, tmp_path):
    """Org tenancy builds user settings from stored rows alone, files included."""
    clean_env.setenv("COWORK_TENANCY_MODE", "org")
    _point(env=clean_env, folder=tmp_path / "run", name="MINDS_API_KEY", text="key-from-the-file")
    get_app_settings.cache_clear()

    assert UserSettings(_env_file=None).minds_api_key is None


def test_the_live_apps_launch_token_comes_from_its_file(clean_env, tmp_path):
    clean_env.delenv("COWORK_LIVE_APPS_LAUNCH_TOKEN", raising=False)
    token = "l" * 40
    _point(env=clean_env, folder=tmp_path / "run", name="COWORK_LIVE_APPS_LAUNCH_TOKEN", text=token + "\n")

    assert AppSettings(_env_file=None).live_apps_launch_token.get_secret_value() == token


def test_self_hosted_settings_dont_require_the_live_apps_token(clean_env, tmp_path):
    clean_env.setenv("COWORK_DEPLOYMENT_KIND", "self-hosted")
    clean_env.delenv("COWORK_LIVE_APPS_LAUNCH_TOKEN", raising=False)
    _all_four_from_files(env=clean_env, folder=tmp_path / "run")

    assert AppSettings(_env_file=None).database.uri == URI


def test_user_settings_read_the_endpoint_key_from_its_file(clean_env, tmp_path):
    _point(env=clean_env, folder=tmp_path / "run", name="OPENAI_COMPATIBLE_API_KEY", text="endpoint-key\n")

    assert UserSettings(_env_file=None).openai_compatible_api_key.get_secret_value() == "endpoint-key"


def test_only_one_trailing_newline_is_dropped(clean_env, tmp_path):
    _point(env=clean_env, folder=tmp_path / "run", name="COWORK_AUTH_TOKEN", text="token\n\n")

    assert AppSettings(_env_file=None).auth_token == "token\n"


def test_the_file_wins_over_cowork_homes_env(clean_env, tmp_path):
    dotenv = _write(folder=tmp_path / "home", name=".env", text=f"DATABASE_URI=sqlite:///{tmp_path}/planted.db\n")
    _point(env=clean_env, folder=tmp_path / "run", name="DATABASE_URI", text=URI)

    assert AppSettings(_env_file=str(dotenv)).database.uri == URI


def test_a_file_variable_in_cowork_homes_env_names_nothing(clean_env, tmp_path):
    """Only the process environment names a credential file, so a line written
    into cowork-home's .env can't point a credential read at another file."""
    planted = _write(folder=tmp_path / "elsewhere", name="token", text="planted-token")
    dotenv = _write(
        folder=tmp_path / "home",
        name=".env",
        text=f"COWORK_AUTH_TOKEN_FILE={planted}\nCOWORK_AUTH_TOKEN=from-dotenv\n",
    )
    clean_env.setenv("COWORK_AUTH_TOKEN", "from-the-environment")

    assert AppSettings(_env_file=str(dotenv)).auth_token == "from-the-environment"
    assert process_secret(name="COWORK_AUTH_TOKEN") == "from-the-environment"


def test_a_missing_file_stops_settings_naming_its_variable(clean_env, tmp_path):
    clean_env.setenv("DATABASE_URI_FILE", str(tmp_path / "run" / "DATABASE_URI"))

    with pytest.raises(SecretFileError) as refused:
        AppSettings(_env_file=None)

    message = str(refused.value)
    assert message.startswith("DATABASE_URI_FILE names ")
    assert "can't be read" in message


def test_an_empty_file_stops_settings(clean_env, tmp_path):
    _point(env=clean_env, folder=tmp_path / "run", name="COWORK_AUTH_TOKEN", text="\n")

    with pytest.raises(SecretFileError, match="COWORK_AUTH_TOKEN_FILE names .* which is empty"):
        AppSettings(_env_file=None)


def test_a_file_that_isnt_text_stops_settings_without_its_bytes(clean_env, tmp_path):
    path = tmp_path / "run" / "COWORK_AUTH_TOKEN"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfesecret-bytes")
    clean_env.setenv("COWORK_AUTH_TOKEN_FILE", str(path))

    with pytest.raises(SecretFileError) as refused:
        AppSettings(_env_file=None)

    assert "isn't UTF-8 text" in str(refused.value)
    assert "secret-bytes" not in str(refused.value)


def test_a_value_and_its_file_together_stop_settings(clean_env, tmp_path):
    _point(env=clean_env, folder=tmp_path / "run", name="COWORK_AUTH_TOKEN", text="from-the-file")
    clean_env.setenv("COWORK_AUTH_TOKEN", "from-the-environment")

    with pytest.raises(SecretFileError, match="COWORK_AUTH_TOKEN and COWORK_AUTH_TOKEN_FILE are both set"):
        AppSettings(_env_file=None)


def test_an_empty_value_beside_its_file_reads_the_file(clean_env, tmp_path):
    """Compose can pass an empty value beside the file; the file decides."""
    _point(env=clean_env, folder=tmp_path / "run", name="COWORK_AUTH_TOKEN", text="from-the-file")
    clean_env.setenv("COWORK_AUTH_TOKEN", "")

    assert AppSettings(_env_file=None).auth_token == "from-the-file"


def test_self_hosted_settings_stop_naming_each_missing_credential(clean_env, tmp_path):
    clean_env.setenv("COWORK_DEPLOYMENT_KIND", "self-hosted")
    clean_env.setenv("COWORK_AUTH_TOKEN", "from-the-environment")
    _point(env=clean_env, folder=tmp_path / "run", name="MINDS_API_KEY", text="from-the-file")

    with pytest.raises(SecretFileError) as refused:
        AppSettings(_env_file=None)

    message = str(refused.value)
    assert "DATABASE_URI or DATABASE_URI_FILE" in message
    assert "COWORK_PUBLISH_TOKEN or COWORK_PUBLISH_TOKEN_FILE" in message
    assert "COWORK_AUTH_TOKEN or" not in message
    assert "MINDS_API_KEY or" not in message
    assert "from-the-environment" not in message


def test_self_hosted_settings_never_take_a_credential_from_cowork_homes_env(clean_env, tmp_path):
    clean_env.setenv("COWORK_DEPLOYMENT_KIND", "self-hosted")
    _all_four_from_files(env=clean_env, folder=tmp_path / "run")
    clean_env.delenv("DATABASE_URI_FILE")
    dotenv = _write(folder=tmp_path / "home", name=".env", text=f"DATABASE_URI={URI}\n")

    with pytest.raises(SecretFileError, match="DATABASE_URI or DATABASE_URI_FILE"):
        AppSettings(_env_file=str(dotenv))


def test_a_self_hosted_setup_without_a_database_creates_no_sqlite_file(clean_env, tmp_path):
    from cowork.dev_setup import run_dev_setup

    clean_env.setenv("COWORK_DEPLOYMENT_KIND", "self-hosted")
    _all_four_from_files(env=clean_env, folder=tmp_path / "run")
    clean_env.delenv("DATABASE_URI_FILE")

    with pytest.raises(SecretFileError, match="DATABASE_URI or DATABASE_URI_FILE"):
        run_dev_setup()

    assert not (tmp_path / "home" / "cowork.db").exists()


def test_a_self_hosted_setup_uses_the_database_its_file_names(clean_env, tmp_path):
    from cowork.dev_setup import run_dev_setup

    clean_env.setenv("COWORK_DEPLOYMENT_KIND", "self-hosted")
    _all_four_from_files(env=clean_env, folder=tmp_path / "run")
    named = tmp_path / "named" / "store.db"
    _point(env=clean_env, folder=tmp_path / "run", name="DATABASE_URI", text=f"sqlite:///{named}")
    get_app_settings.cache_clear()

    run_dev_setup()

    assert named.exists()
    assert not (tmp_path / "home" / "cowork.db").exists()


def test_settings_outside_self_hosted_keep_the_sqlite_default(clean_env, tmp_path):
    assert AppSettings(_env_file=None).database.uri == f"sqlite:///{tmp_path / 'home' / 'cowork.db'}"
