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


def test_log_level_is_read_from_the_config_file(env_file, monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    env_file.write_text("LOG_LEVEL=INFO\n", encoding="utf-8")
    get_app_settings.cache_clear()

    assert _resolved_log_level_name() == "INFO"


def test_environment_wins_over_the_config_file(env_file, monkeypatch):
    env_file.write_text("LOG_LEVEL=INFO\n", encoding="utf-8")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    get_app_settings.cache_clear()

    assert _resolved_log_level_name() == "ERROR"
