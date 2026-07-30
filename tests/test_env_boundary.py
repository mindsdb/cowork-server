"""ENG-1127 Phase A: the server exports the DB's settings to the CLI's .env.

Pure derivation plus the SettingService integration (local-tenancy-only,
preserves unmanaged lines, migration never re-exports).
"""
from types import SimpleNamespace

import pytest

import cowork.services.settings as settings_mod
from cowork.common.settings.env_boundary import db_to_env, merge_env_lines
from cowork.common.settings.user_settings import UserSettings
from cowork.db.session import get_open_session
from cowork.services.settings import SettingService


# ── pure derivation ────────────────────────────────────────────────────

def test_db_to_env_formats_and_excludes_models():
    s = UserSettings(
        anthropic_api_key="sk-secret",
        planning_provider="minds_cloud",
        minds_url="https://mdb.example",
        planning_model="latest:sonnet",  # not aliased (ENG-739)
    )
    present = {"anthropic_api_key", "planning_provider", "minds_url", "planning_model"}
    out = db_to_env(s, present)
    assert out["ANTON_ANTHROPIC_API_KEY"] == "sk-secret"  # decrypted plaintext
    assert out["ANTON_PLANNING_PROVIDER"] == "minds-cloud"  # dash form
    assert out["ANTON_MINDS_URL"] == "https://mdb.example"
    assert not any("MODEL" in k for k in out)


def test_db_to_env_only_exports_stored_keys():
    # minds_url has a non-None default but no row → must not be exported
    s = UserSettings(anthropic_api_key="sk-x")
    out = db_to_env(s, present_keys={"anthropic_api_key"})
    assert out == {"ANTON_ANTHROPIC_API_KEY": "sk-x"}
    assert "ANTON_MINDS_URL" not in out
    assert db_to_env(UserSettings(), present_keys=set()) == {}


def test_merge_preserves_unmanaged_and_replaces_managed():
    existing = (
        "COWORK_AUTH_TOKEN=tok-123\n"
        "ANTON_PLANNING_MODEL=latest:sonnet\n"
        "ANTON_ANTHROPIC_API_KEY=stale\n"
        "ANTON_FIRST_RUN_DONE=true\n"
        "# a comment\n"
    )
    merged = merge_env_lines(existing, {"ANTON_ANTHROPIC_API_KEY": "fresh", "ANTON_MINDS_URL": "https://m"})
    lines = merged.strip().split("\n")
    # Unmanaged lines survive verbatim.
    assert "COWORK_AUTH_TOKEN=tok-123" in lines
    assert "ANTON_PLANNING_MODEL=latest:sonnet" in lines
    assert "ANTON_FIRST_RUN_DONE=true" in lines
    assert "# a comment" in lines
    # The stale managed line is replaced, not duplicated.
    assert "ANTON_ANTHROPIC_API_KEY=stale" not in lines
    assert lines.count("ANTON_ANTHROPIC_API_KEY=fresh") == 1
    assert "ANTON_MINDS_URL=https://m" in lines


def test_merge_drops_cleared_managed_key():
    existing = "COWORK_AUTH_TOKEN=tok\nANTON_MINDS_API_KEY=old\n"
    merged = merge_env_lines(existing, {})
    assert "ANTON_MINDS_API_KEY" not in merged
    assert "COWORK_AUTH_TOKEN=tok" in merged


# ── SettingService integration ─────────────────────────────────────────

@pytest.fixture
def local_export(monkeypatch, tmp_path):
    """Point the export at a tmp .env and pretend we're a local desktop install."""
    monkeypatch.setattr(settings_mod, "cowork_home", lambda: tmp_path)
    monkeypatch.setattr(settings_mod, "get_app_settings", lambda: SimpleNamespace(tenancy_mode="local"))
    return tmp_path / ".env"


def _cleanup(session, *keys):
    svc = SettingService(session)
    for k in keys:
        try:
            svc.delete_setting(k)
        except ValueError:
            pass


def test_save_all_exports_env_and_preserves_unmanaged(local_export):
    env_path = local_export
    env_path.write_text("COWORK_AUTH_TOKEN=tok-1\nANTON_PLANNING_MODEL=latest:sonnet\n", encoding="utf-8")
    session = get_open_session()
    try:
        _cleanup(session, "minds_api_key", "minds_url", "planning_provider")
        SettingService(session).save_all(
            {"minds_api_key": "sk-abc", "minds_url": "https://mdb", "planning_provider": "minds_cloud"}
        )
        text = env_path.read_text(encoding="utf-8")
        assert "ANTON_MINDS_API_KEY=sk-abc" in text
        assert "ANTON_MINDS_URL=https://mdb" in text
        assert "ANTON_PLANNING_PROVIDER=minds-cloud" in text
        # unmanaged lines the server must not touch
        assert "COWORK_AUTH_TOKEN=tok-1" in text
        assert "ANTON_PLANNING_MODEL=latest:sonnet" in text
        assert oct(env_path.stat().st_mode)[-3:] == "600"
    finally:
        _cleanup(session, "minds_api_key", "minds_url", "planning_provider")
        session.close()


def test_export_skipped_for_org_tenancy(monkeypatch, tmp_path):
    monkeypatch.setattr(settings_mod, "cowork_home", lambda: tmp_path)
    monkeypatch.setattr(settings_mod, "get_app_settings", lambda: SimpleNamespace(tenancy_mode="org"))
    session = get_open_session()
    try:
        _cleanup(session, "minds_url")
        SettingService(session).save_all({"minds_url": "https://mdb"})
        assert not (tmp_path / ".env").exists()  # cloud pod writes no .env
    finally:
        _cleanup(session, "minds_url")
        session.close()


def test_clear_credentials_wipes_creds_from_env_but_keeps_model(local_export):
    env_path = local_export
    env_path.write_text("ANTON_PLANNING_MODEL=latest:sonnet\n", encoding="utf-8")
    session = get_open_session()
    try:
        _cleanup(session, "minds_api_key", "minds_url", "planning_provider")
        svc = SettingService(session)
        svc.save_all(
            {"minds_api_key": "sk-abc", "minds_url": "https://mdb", "planning_provider": "minds_cloud"}
        )
        assert "ANTON_MINDS_API_KEY=sk-abc" in env_path.read_text(encoding="utf-8")

        svc.clear_credentials()
        text = env_path.read_text(encoding="utf-8")
        assert "ANTON_MINDS_API_KEY" not in text   # credential wiped from the CLI too
        assert "ANTON_MINDS_URL" not in text
        assert "ANTON_PLANNING_MODEL=latest:sonnet" in text  # CLI model pin survives
    finally:
        _cleanup(session, "minds_api_key", "minds_url", "planning_provider")
        session.close()


def test_migration_write_does_not_export(local_export):
    # migration seeds the DB from .env (export_env=False), so a seed write must not rewrite it
    env_path = local_export
    session = get_open_session()
    try:
        _cleanup(session, "minds_url")
        SettingService(session).upsert_setting("minds_url", "https://seed", export_env=False)
        assert not env_path.exists()
    finally:
        _cleanup(session, "minds_url")
        session.close()
