"""COWORK_HOME data-root isolation.

Every cowork path must derive from a single root so preview/stable desktop
builds can be fully isolated from a user's production ~/.cowork (ENG-324) by
setting one env var. These tests pin that contract.
"""
from pathlib import Path

from cowork.common.paths import cowork_home, pod_local_only
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
    "STATE_PATH",
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


def test_explicit_state_path_still_overrides_cowork_home(monkeypatch, tmp_path):
    # OAuthSettings.state_path has no validation_alias, so pydantic-settings
    # falls back to the bare uppercased field name: STATE_PATH, not a
    # COWORK_-prefixed name (same pattern as MASTER_KEY_PATH). The cowork-server
    # Helm values file relies on this exact name to keep OAuth state off the
    # shared EFS tree; this pins it so a future validation_alias addition
    # can't silently change the env var cloud config depends on.
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATE_PATH", "/home/app/oauth_state.json")

    assert OAuthSettings(_env_file=None).state_path == "/home/app/oauth_state.json"


# pod_local_only, the mechanism that keeps scratch/deployment-local state
# (connector-probe credential files, publish's state.json, the anton
# harness's temp data-vault dir) off the shared COWORK_HOME tree in org mode,
# since none of the three carry an org_id segment for scoped_storage_root to
# key on.
def test_pod_local_only_is_a_noop_in_local_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("COWORK_TENANCY_MODE", raising=False)
    get_app_settings.cache_clear()

    local_path = tmp_path / "cowork" / "tmp"
    assert pod_local_only(local_path, "tmp") == local_path

    get_app_settings.cache_clear()


def test_pod_local_only_relocates_off_cowork_home_in_org_mode(monkeypatch, tmp_path):
    home = tmp_path / "cowork-shared"
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.delenv("COWORK_POD_SCRATCH_DIR", raising=False)
    get_app_settings.cache_clear()

    resolved = pod_local_only(home / "tmp", "tmp")

    assert home not in resolved.parents
    assert resolved != home / "tmp"

    get_app_settings.cache_clear()


def test_pod_local_only_org_mode_defaults_under_system_temp_dir(monkeypatch, tmp_path):
    import tempfile

    home = tmp_path / "cowork-shared"
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.delenv("COWORK_POD_SCRATCH_DIR", raising=False)
    get_app_settings.cache_clear()

    resolved = pod_local_only(home / "tmp", "tmp")

    assert resolved == Path(tempfile.gettempdir()) / "cowork" / "tmp"

    get_app_settings.cache_clear()


def test_pod_local_only_honors_explicit_scratch_dir_override(monkeypatch, tmp_path):
    home = tmp_path / "cowork-shared"
    scratch = tmp_path / "pod-scratch"
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_POD_SCRATCH_DIR", str(scratch))
    get_app_settings.cache_clear()

    assert pod_local_only(home / "tmp", "tmp") == scratch / "tmp"

    get_app_settings.cache_clear()


def test_bearer_auth_token_env_derives_from_cowork_home(monkeypatch, tmp_path):
    # With COWORK_REQUIRE_AUTH=true the effective token is mirrored to
    # <cowork_home()>/.env so the desktop app can read it. A hardcoded
    # ~/.cowork/.env would leave an isolated build (COWORK_HOME set) writing
    # token state into another install's data home (ENG-868).
    from cowork.server import create_app

    home = tmp_path / "cowork-preview"
    # Redirect the OS home too, so a regression writes into tmp_path instead
    # of the developer's real ~/.cowork/.env.
    fake_os_home = tmp_path / "os-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_os_home))
    monkeypatch.setenv("COWORK_HOME", str(home))
    monkeypatch.setenv("COWORK_REQUIRE_AUTH", "true")
    monkeypatch.setenv("COWORK_AUTH_TOKEN", "tok-eng-868")
    get_app_settings.cache_clear()
    try:
        create_app()
        env_file = home / ".env"
        assert env_file.exists(), "auth token state must live under cowork_home()"
        assert "COWORK_AUTH_TOKEN=tok-eng-868" in env_file.read_text(encoding="utf-8")
    finally:
        get_app_settings.cache_clear()
