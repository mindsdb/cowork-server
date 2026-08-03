"""ENG-1127 Phase A: the server exports the DB's settings to the CLI's .env.

Pure derivation plus the SettingService integration (local-tenancy-only,
preserves unmanaged lines, migration never re-exports).
"""
import errno
import os
from types import SimpleNamespace

import pytest

import cowork.services.settings as settings_mod
from cowork.common.settings import env_boundary as eb
from cowork.common.settings.env_boundary import db_to_env, merge_env_lines
from cowork.common.settings.user_settings import UserSettings
from cowork.db.session import get_open_session
from cowork.services.settings import SettingService


# ── pure derivation ────────────────────────────────────────────────────

def test_db_to_env_pairs_provider_with_its_model():
    # A provider is exported WITH its resolved model (ENG-1127 review): never a
    # provider line without a matching model.
    s = UserSettings(minds_api_key="sk-minds", minds_url="https://mdb.example",
                     planning_provider="minds_cloud", coding_provider="minds_cloud")
    present = {"minds_api_key", "minds_url", "planning_provider", "coding_provider"}
    out = db_to_env(s, present)
    assert out["ANTON_PLANNING_PROVIDER"] == "minds-cloud"      # dash form
    assert out["ANTON_MINDS_API_KEY"] == "sk-minds"             # decrypted plaintext
    assert out["ANTON_MINDS_URL"] == "https://mdb.example"
    assert out["ANTON_PLANNING_MODEL"]                          # model rides with the provider
    assert out["ANTON_CODING_MODEL"]


def test_db_to_env_translates_gemini_to_openai_compatible():
    # The pinned CLI has no first-class gemini provider — it runs Gemini as
    # openai-compatible + Google's base URL + the key in the OpenAI slot. The
    # export must render that shape, not the literal provider=gemini the CLI rejects.
    s = UserSettings(gemini_api_key="sk-gem", planning_provider="gemini", coding_provider="gemini")
    out = db_to_env(s, present_keys={"gemini_api_key", "planning_provider", "coding_provider"})
    assert out["ANTON_PLANNING_PROVIDER"] == "openai-compatible"
    assert out["ANTON_OPENAI_API_KEY"] == "sk-gem"             # gemini key rides the OpenAI slot
    assert out["ANTON_OPENAI_BASE_URL"].startswith("https://generativelanguage.googleapis.com")
    assert "ANTON_GEMINI_API_KEY" not in out                   # a field the CLI ignores
    assert "gemini" not in out.get("ANTON_PLANNING_PROVIDER", "")


def test_db_to_env_exports_nothing_when_unconfigured():
    # No key anywhere → no provider/model/creds; flags export only when stored.
    assert db_to_env(UserSettings(), present_keys=set()) == {}
    # A resolved provider with no key exports no provider line for that role.
    out = db_to_env(UserSettings(), present_keys={"planning_provider"})
    assert "ANTON_PLANNING_PROVIDER" not in out


def test_merge_preserves_unmanaged_and_replaces_managed():
    existing = (
        "COWORK_AUTH_TOKEN=tok-123\n"
        "ANTON_PLANNING_MODEL=latest:sonnet\n"  # now MANAGED (ENG-1127): dropped unless re-supplied
        "ANTON_ANTHROPIC_API_KEY=stale\n"
        "ANTON_FIRST_RUN_DONE=true\n"
        "# a comment\n"
    )
    merged = merge_env_lines(existing, {"ANTON_ANTHROPIC_API_KEY": "fresh", "ANTON_MINDS_URL": "https://m"})
    lines = merged.strip().split("\n")
    # Genuinely unmanaged lines survive verbatim.
    assert "COWORK_AUTH_TOKEN=tok-123" in lines
    assert "ANTON_FIRST_RUN_DONE=true" in lines
    assert "# a comment" in lines
    # Model vars are managed now — a stale one not re-supplied is dropped (the
    # export always re-supplies the resolved model, so no mismatch survives).
    assert "ANTON_PLANNING_MODEL=latest:sonnet" not in lines
    # The stale managed line is replaced, not duplicated.
    assert "ANTON_ANTHROPIC_API_KEY=stale" not in lines
    assert lines.count("ANTON_ANTHROPIC_API_KEY=fresh") == 1
    assert "ANTON_MINDS_URL=https://m" in lines


def test_merge_drops_cleared_managed_key():
    existing = "COWORK_AUTH_TOKEN=tok\nANTON_MINDS_API_KEY=old\n"
    merged = merge_env_lines(existing, {})
    assert "ANTON_MINDS_API_KEY" not in merged
    assert "COWORK_AUTH_TOKEN=tok" in merged


# ── dotenv-injection guard (Finding 3) ─────────────────────────────────

def test_merge_env_lines_rejects_crlf_injection():
    # A CR/LF in a value must not become a second (unmanaged, surviving) line.
    poisoned = "https://x\nDATABASE_URI=sqlite:///tmp/evil.db"
    merged = merge_env_lines("", {"ANTON_MINDS_URL": poisoned, "ANTON_MINDS_API_KEY": "sk-ok"})
    assert "DATABASE_URI" not in merged          # injected assignment dropped
    assert "ANTON_MINDS_URL" not in merged        # the poisoned value is not smuggled
    assert "ANTON_MINDS_API_KEY=sk-ok" in merged  # clean siblings still export
    # every emitted line is a single well-formed assignment
    assert all(ln.count("=") >= 1 for ln in merged.split("\n") if ln)


def test_db_to_env_drops_newline_bearing_value(monkeypatch):
    # A minds-cloud user so minds_url is genuinely part of the export, then poison
    # it — the injection guard (not a missing key) is what must drop it.
    s = UserSettings(minds_api_key="sk-minds", planning_provider="minds_cloud")
    monkeypatch.setattr(s, "minds_url", "https://x\nDATABASE_URI=sqlite:///evil.db", raising=False)
    out = db_to_env(s, present_keys={"minds_api_key", "planning_provider"})
    assert "ANTON_MINDS_API_KEY" in out    # the clean sibling still exports
    assert "ANTON_MINDS_URL" not in out    # poisoned value refused by the guard


def test_export_never_writes_injected_line_end_to_end(local_export):
    # Full path: a poisoned DB value must never reach the CLI's .env as a 2nd line.
    env_path = local_export
    session = get_open_session()
    try:
        _cleanup(session, "minds_url", "minds_api_key")
        svc = SettingService(session)
        svc.save_all({"minds_api_key": "sk-clean", "minds_url": "https://h\nDATABASE_URI=x"})
        text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        assert "DATABASE_URI" not in text
        assert "ANTON_MINDS_API_KEY=sk-clean" in text  # the clean sibling still lands
    finally:
        _cleanup(session, "minds_url", "minds_api_key")
        session.close()


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
        # Genuinely unmanaged lines the server must not touch.
        assert "COWORK_AUTH_TOKEN=tok-1" in text
        # The model is managed now: the stale hand-set pin is replaced by the
        # resolved model that pairs with the exported provider (no mismatch).
        assert "ANTON_PLANNING_MODEL=latest:sonnet" not in text
        assert "ANTON_PLANNING_MODEL=" in text
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


def test_clear_credentials_wipes_creds_and_orphaned_model_from_env(local_export):
    env_path = local_export
    env_path.write_text("COWORK_AUTH_TOKEN=keep\n", encoding="utf-8")
    session = get_open_session()
    try:
        _cleanup(session, "minds_api_key", "minds_url", "planning_provider")
        svc = SettingService(session)
        svc.save_all(
            {"minds_api_key": "sk-abc", "minds_url": "https://mdb", "planning_provider": "minds_cloud"}
        )
        exported = env_path.read_text(encoding="utf-8")
        assert "ANTON_MINDS_API_KEY=sk-abc" in exported
        assert "ANTON_PLANNING_MODEL=" in exported  # provider+model exported as a pair

        svc.clear_credentials()
        text = env_path.read_text(encoding="utf-8")
        assert "ANTON_MINDS_API_KEY" not in text   # credential wiped from the CLI too
        assert "ANTON_MINDS_URL" not in text
        # With no key left, no provider resolves — so its now-orphaned model line
        # is dropped too, rather than left mismatched against a gone provider.
        assert "ANTON_PLANNING_MODEL" not in text
        assert "COWORK_AUTH_TOKEN=keep" in text     # unmanaged line untouched
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


def test_backfill_minds_url_exports_to_env_too(local_export):
    # ENG-1127 review: the legacy-host backfill writes minds_url directly, so it
    # must also export — otherwise the DB moves to the canonical host while the
    # CLI's .env keeps the dead mdb.ai endpoint.
    from cowork.common.settings.app_settings import default_minds_api_host
    from cowork.migrations import backfill_minds_url

    env_path = local_export
    canonical = default_minds_api_host()
    session = get_open_session()
    try:
        _cleanup(session, "minds_url", "minds_api_key", "planning_provider")
        svc = SettingService(session)
        # A real minds-cloud user (has the key) on the legacy host — only then is
        # minds the resolved provider, so ANTON_MINDS_URL is part of the export.
        svc.upsert_setting("minds_api_key", "sk-minds", export_env=False)
        svc.upsert_setting("planning_provider", "minds_cloud", export_env=False)
        svc.upsert_setting("minds_url", "https://mdb.ai", export_env=False)
        env_path.write_text("ANTON_MINDS_URL=https://mdb.ai\n", encoding="utf-8")

        assert backfill_minds_url(session) is True
        # Both stores land on the canonical host.
        assert svc.load().minds_url == canonical
        env_text = env_path.read_text(encoding="utf-8")
        assert f"ANTON_MINDS_URL={canonical}" in env_text
        assert "https://mdb.ai" not in env_text
    finally:
        _cleanup(session, "minds_url", "minds_api_key", "planning_provider")
        session.close()


def test_export_serializes_concurrent_writers(local_export, monkeypatch):
    # ENG-1127 review: the read/merge/write must be serialized so two concurrent
    # exporters can't lost-update the file. Instrument the critical section and
    # assert at most one thread is ever inside it.
    import threading
    import time

    state = {"cur": 0, "max": 0}
    guard = threading.Lock()
    real = settings_mod.db_to_env

    def instrumented(settings, present_keys):
        with guard:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        time.sleep(0.03)  # widen the window an unserialized export would overlap in
        with guard:
            state["cur"] -= 1
        return real(settings, present_keys)

    monkeypatch.setattr(settings_mod, "db_to_env", instrumented)

    def export():
        SettingService(get_open_session())._export_env_for_cli()

    threads = [threading.Thread(target=export) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["max"] == 1


# ── atomic_write_env: Windows share-mode lock hardening (ENG-1209/ENG-1127) ──

def test_atomic_write_env_retries_transient_lock(tmp_path, monkeypatch):
    # The CLI (or a version-skewed server) holding .env open EPERM'd the rename on
    # Windows (ENG-1209). Now that the server writes it, the rename must retry.
    dest = tmp_path / ".env"
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(errno.EACCES, "share-mode lock")
        return real_replace(src, dst)

    monkeypatch.setattr(eb.os, "replace", flaky_replace)
    monkeypatch.setattr(eb.time, "sleep", lambda *_: None)  # no real backoff in tests

    eb.atomic_write_env(dest, "ANTON_MINDS_URL=https://m\n")

    assert calls["n"] == 3  # two transient failures, third lands
    assert dest.read_text(encoding="utf-8") == "ANTON_MINDS_URL=https://m\n"
    assert list(tmp_path.glob(".env.*.tmp")) == []  # temp consumed by the rename


def test_atomic_write_env_rethrows_non_transient_and_cleans_temp(tmp_path, monkeypatch):
    # A non-lock error (ENOENT, unwritable target, …) must fail fast, not burn the
    # retry budget, and must never leave the plaintext-key temp behind.
    dest = tmp_path / ".env"
    calls = {"n": 0}

    def broken_replace(src, dst):
        calls["n"] += 1
        raise OSError(errno.ENOENT, "gone")

    monkeypatch.setattr(eb.os, "replace", broken_replace)
    monkeypatch.setattr(eb.time, "sleep", lambda *_: None)

    with pytest.raises(OSError):
        eb.atomic_write_env(dest, "x\n")

    assert calls["n"] == 1  # rethrown on the first attempt, no retry
    assert not dest.exists()
    assert list(tmp_path.glob(".env.*.tmp")) == []  # temp cleaned up on failure


def test_atomic_write_env_sweeps_stale_temp_keeps_fresh(tmp_path):
    # Orphaned temps hold the full plaintext key, so a stale one is reclaimed; a
    # concurrent writer's fresh in-flight temp is spared.
    stale = tmp_path / ".env.stale123.tmp"
    stale.write_text("ANTON_MINDS_API_KEY=leaked\n", encoding="utf-8")
    fresh = tmp_path / ".env.fresh456.tmp"
    fresh.write_text("in-flight\n", encoding="utf-8")
    old = os.stat(fresh).st_mtime - (eb._STALE_TMP_S + 60)
    os.utime(stale, (old, old))

    eb.atomic_write_env(tmp_path / ".env", "ANTON_MINDS_URL=https://m\n")

    assert not stale.exists()  # orphaned plaintext-key temp reclaimed
    assert fresh.exists()      # a live writer's temp is not yanked mid-rename
