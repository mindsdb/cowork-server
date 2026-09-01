"""The log level has to come from the app's settings, not from `os.getenv`.

The desktop keeps its configuration in `<COWORK_HOME>/.env`, which no
environment variable carries, so raising LOG_LEVEL there changed nothing and a
customer's support log held only uvicorn access lines.

These call the resolver directly and never `setup_logging()`: that ends in
`logging.basicConfig(force=True)`, which drops pytest's capture handler for
every test that runs after it.
"""
from __future__ import annotations

import pytest

from cowork.common.logger import _resolved_log_level_name
from cowork.common.paths import cowork_home
from cowork.common.settings.app_settings import get_app_settings


@pytest.fixture()
def env_file():
    """`<COWORK_HOME>/.env`, removed afterwards — COWORK_HOME is session-scoped
    (see conftest), so a file left behind would follow every later test.
    """
    path = cowork_home() / ".env"
    existed = path.exists()
    original = path.read_text(encoding="utf-8") if existed else None
    get_app_settings.cache_clear()
    try:
        yield path
    finally:
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(original, encoding="utf-8")
        get_app_settings.cache_clear()


def test_log_level_is_read_from_the_config_file(env_file, monkeypatch, tmp_path):
    # The chain is [<COWORK_HOME>/.env, ".env"] and pydantic-settings is
    # last-wins, so a developer's repo-local .env would beat the file under
    # test. Run from an empty directory instead of skipping, so the result does
    # not depend on the working tree.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    env_file.write_text("LOG_LEVEL=INFO\n", encoding="utf-8")
    get_app_settings.cache_clear()

    assert _resolved_log_level_name() == "INFO"


def test_environment_wins_over_the_config_file(env_file, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    env_file.write_text("LOG_LEVEL=INFO\n", encoding="utf-8")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    get_app_settings.cache_clear()

    assert _resolved_log_level_name() == "ERROR"


def test_unbuildable_settings_fall_back_to_the_environment(monkeypatch):
    """Logging is configured at import, before anything could report a settings
    error, so a broken config must not take the logger down with it."""
    import cowork.common.settings.app_settings as app_settings

    def _boom():
        raise RuntimeError("no settings for you")

    monkeypatch.setattr(app_settings, "get_app_settings", _boom)
    monkeypatch.setenv("LOG_LEVEL", "ERROR")

    assert _resolved_log_level_name() == "ERROR"


def test_a_missing_log_level_field_raises_instead_of_degrading(monkeypatch):
    """Only construction is guarded. If the field were renamed away, every
    deployment would quietly drop to WARNING — so this must raise, loudly."""
    import cowork.common.settings.app_settings as app_settings

    class _NoField:
        pass

    monkeypatch.setattr(app_settings, "get_app_settings", lambda: _NoField())

    with pytest.raises(AttributeError):
        _resolved_log_level_name()
