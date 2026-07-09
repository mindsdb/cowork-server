"""COWORK_HOME data-root isolation.

Every cowork path must derive from a single root so preview/stable desktop
builds can be fully isolated from a user's production ~/.cowork (ENG-324) by
setting one env var. These tests pin that contract.
"""
from pathlib import Path

from cowork.common.paths import cowork_home
from cowork.common.settings.app_settings import (
    AppSettings,
    OAuthSettings,
    StreamSettings,
    _env_file_chain,
    get_app_settings,
)
from cowork.harnesses.anton_harness.settings import AntonHarnessSettings
from cowork.harnesses.hermes_harness.settings import HermesHarnessSettings


def test_cowork_home_defaults_to_dot_cowork(monkeypatch):
    monkeypatch.delenv("COWORK_HOME", raising=False)
    assert cowork_home() == Path.home() / ".cowork"


def test_cowork_home_honors_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "cowork-preview"))
    assert cowork_home() == tmp_path / "cowork-preview"


def test_cowork_home_expands_user(monkeypatch):
    monkeypatch.setenv("COWORK_HOME", "~/.cowork-preview")
    assert cowork_home() == Path.home() / ".cowork-preview"


def test_isolated_build_does_not_inherit_legacy_anton_env(monkeypatch, tmp_path):
    # An isolated build (COWORK_HOME set) must NOT read ~/.anton/.env — a path
    # var there (DATABASE_URI, …) would resolve every build onto the same DB
    # and defeat the isolation. Only <COWORK_HOME>/.env and local .env apply.
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "cowork-preview"))
    legacy = str(Path.home() / ".anton" / ".env")
    assert legacy not in _env_file_chain()


def test_prod_build_still_reads_legacy_anton_env(monkeypatch):
    # The default (prod) home keeps the legacy fallback for un-migrated
    # installs, ordered BEFORE <COWORK_HOME>/.env so the migrated file wins.
    monkeypatch.delenv("COWORK_HOME", raising=False)
    chain = _env_file_chain()
    legacy = str(Path.home() / ".anton" / ".env")
    assert legacy in chain
    assert chain.index(legacy) < chain.index(str(cowork_home() / ".env"))


# Per-resource env vars that, when set, intentionally win over the
# COWORK_HOME-derived default. The test harness (conftest) injects some of
# these, so clear them all to observe the pure derivation.
_PER_RESOURCE_OVERRIDES = [
    "DATABASE_URI",
    "MASTER_KEY_PATH",
    "COWORK_PROJECTS_DIR",
    "PROJECTS_ROOT_DIR",
    "COWORK_FILES_DIR",
    "FILES_ROOT_DIR",
    "COWORK_SKILLS_DIR",
    "SKILLS_ROOT_DIR",
    "COWORK_VAULT_DIR",
    "CONNECTOR_VAULT_DIR",
    "COWORK_STREAMS_DIR",
    "HERMES_ROOT_DIR",
    "HERMES_HOME",
    "ANTON_SKILLS_ROOT_DIR",
]


def test_all_settings_paths_derive_from_cowork_home(monkeypatch, tmp_path):
    home = tmp_path / "cowork-preview"
    monkeypatch.setenv("COWORK_HOME", str(home))
    for var in _PER_RESOURCE_OVERRIDES:
        monkeypatch.delenv(var, raising=False)
    get_app_settings.cache_clear()

    s = AppSettings(_env_file=None)
    assert s.database.uri == f"sqlite:///{home / 'cowork.db'}"
    assert Path(s.project.root_dir) == home / "projects"
    assert Path(s.file.root_dir) == home / "files"
    assert Path(s.skill.root_dir) == home / "skills"
    assert Path(s.connector.vault_dir) == home / "data-vault"
    assert Path(s.memory.root_dir) == home / "memory"
    assert Path(s.master_key_path) == home / ".master_key"
    assert Path(StreamSettings(_env_file=None).dir) == home / "streams"
    assert Path(OAuthSettings(_env_file=None).state_path) == home / "oauth_state.json"
    assert Path(AntonHarnessSettings(_env_file=None).skills_root_dir) == home / "anton" / "skills"
    assert Path(HermesHarnessSettings(_env_file=None).root_dir) == home / "hermes"

    get_app_settings.cache_clear()


def test_explicit_database_uri_still_overrides_cowork_home(monkeypatch, tmp_path):
    # Per-resource env vars keep their precedence over the derived default.
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DATABASE_URI", "sqlite:////tmp/explicit.db")
    get_app_settings.cache_clear()

    assert AppSettings(_env_file=None).database.uri == "sqlite:////tmp/explicit.db"

    get_app_settings.cache_clear()
