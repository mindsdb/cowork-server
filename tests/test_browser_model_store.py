"""WS4-T1/T2: content-free data model, digest guard, permission check,
action store, and migration up/down idempotency.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session

from alembic import command
from alembic.script import ScriptDirectory

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.migrations import _alembic_config, run_schema_migrations
from cowork.db.session import get_engine
from cowork.models.browser import BrowserSession, BrowserTabGrant
from cowork.models.conversation import Conversation
from cowork.schemas.browser import (
    ALLOWED_DIGEST_KEYS,
    BrowserActionClass,
    BrowserActionType,
    BrowserErrorKind,
    DisallowedDigestKeyError,
    PermissionDecision,
    ResultCode,
    assert_content_free_digest,
    build_observed_digest,
    host_only,
    result_code_to_error_kind,
)
from cowork.services.browser.actions import BrowserActionStore
from cowork.services.browser.permissions import BrowserPermissionService
from cowork.services.projects import GENERAL_PROJECT_ID


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def _make_session_row(session: Session, domain: str = "example.com") -> BrowserSession:
    conv = Conversation(topic="t", project_id=GENERAL_PROJECT_ID)
    session.add(conv)
    session.commit()
    session.refresh(conv)
    bs = BrowserSession(
        conversation_id=conv.id,
        project_id=GENERAL_PROJECT_ID,
        active_domain=domain,
    )
    session.add(bs)
    session.commit()
    session.refresh(bs)
    return bs


# ── digest guard (AC8) ────────────────────────────────────────────────
def test_digest_allowlist_is_exactly_four_keys():
    assert ALLOWED_DIGEST_KEYS == {"http_status", "final_domain", "link_count", "settled"}


@pytest.mark.parametrize(
    "bad_key",
    ["text", "url", "path", "query", "title", "href", "cookie", "value", "selector"],
)
def test_assert_content_free_digest_rejects_disallowed_keys(bad_key):
    with pytest.raises(DisallowedDigestKeyError):
        assert_content_free_digest({"http_status": 200, bad_key: "leak"})


def test_assert_content_free_digest_rejects_full_url_final_domain():
    # A URL smuggled through the allowed `final_domain` key must be rejected
    # by the VALUE guard, not silently persisted.
    with pytest.raises(DisallowedDigestKeyError):
        assert_content_free_digest(
            {"final_domain": "https://example.com/path?token=abc"}
        )
    with pytest.raises(DisallowedDigestKeyError):
        assert_content_free_digest({"final_domain": "example.com/a/b"})
    with pytest.raises(DisallowedDigestKeyError):
        assert_content_free_digest({"final_domain": "user@example.com"})
    # A bare host is fine.
    assert_content_free_digest({"final_domain": "example.com"})


def test_assert_content_free_digest_rejects_bad_value_types():
    with pytest.raises(DisallowedDigestKeyError):
        assert_content_free_digest({"http_status": "200"})
    with pytest.raises(DisallowedDigestKeyError):
        assert_content_free_digest({"link_count": "5"})
    with pytest.raises(DisallowedDigestKeyError):
        assert_content_free_digest({"settled": "true"})
    # Correct types pass.
    assert_content_free_digest(
        {"http_status": 200, "link_count": 5, "settled": True}
    )


def test_build_observed_digest_drops_content_keys():
    transient = {
        "http_status": 200,
        "final_domain": "https://sub.Example.com/path?q=1",
        "links": [{"text": "a", "href": "x"}, {"text": "b", "href": "y"}],
        "text": "secret page body",
        "title": "Account list",
        "settled": True,
    }
    digest = build_observed_digest(transient)
    assert_content_free_digest(digest)  # must not raise
    assert digest == {
        "http_status": 200,
        "final_domain": "sub.example.com",
        "link_count": 2,
        "settled": True,
    }
    assert "text" not in digest and "title" not in digest


def test_store_record_observed_rejects_disallowed_digest(session):
    bs = _make_session_row(session)
    store = BrowserActionStore(session)
    store.append_pending(
        session_id=bs.id,
        command_id="cmd-bad",
        idempotency_key="k1",
        action_type=BrowserActionType.inspect,
        domain="example.com",
    )
    store.mark_in_flight("cmd-bad")
    with pytest.raises(DisallowedDigestKeyError):
        store.record_observed(
            "cmd-bad",
            result_code=ResultCode.ok,
            digest={"http_status": 200, "title": "leak"},
        )


def test_store_persists_only_content_free_digest(session):
    bs = _make_session_row(session)
    store = BrowserActionStore(session)
    store.append_pending(
        session_id=bs.id,
        command_id="cmd-ok",
        idempotency_key="k2",
        action_type=BrowserActionType.inspect,
        domain="example.com",
    )
    store.mark_in_flight("cmd-ok")
    action = store.record_observed(
        "cmd-ok",
        result_code=ResultCode.ok,
        transient={"http_status": 200, "text": "body", "links": [1, 2, 3], "settled": True},
        duration_ms=42,
    )
    assert action.status == "observed"
    assert set(action.observed_result.keys()) <= ALLOWED_DIGEST_KEYS
    assert action.observed_result["link_count"] == 3
    assert action.duration_ms == 42


def test_failed_action_never_records_observed_ok(session):
    bs = _make_session_row(session)
    store = BrowserActionStore(session)
    store.append_pending(
        session_id=bs.id,
        command_id="cmd-fail",
        idempotency_key="k3",
        action_type=BrowserActionType.navigate,
        domain="example.com",
    )
    store.mark_in_flight("cmd-fail")
    action = store.mark_failed("cmd-fail", result_code=ResultCode.target_lost)
    assert action.status == "failed"
    assert action.result_code == "target_lost"
    assert action.observed_result is None


def test_append_pending_reuses_pending_row_and_assigns_sequence(session):
    bs = _make_session_row(session)
    store = BrowserActionStore(session)
    a1 = store.append_pending(
        session_id=bs.id, command_id="c1", idempotency_key="same",
        action_type=BrowserActionType.inspect,
    )
    a2 = store.append_pending(
        session_id=bs.id, command_id="c2", idempotency_key="same",
        action_type=BrowserActionType.inspect,
    )
    assert a1.id == a2.id  # reused
    a3 = store.append_pending(
        session_id=bs.id, command_id="c3", idempotency_key="other",
        action_type=BrowserActionType.scroll,
    )
    assert a3.sequence == a1.sequence + 1


# ── result_code → external kind mapping table ─────────────────────────
@pytest.mark.parametrize(
    "code,action_type,expected",
    [
        (ResultCode.ok, None, BrowserErrorKind.ok),
        (ResultCode.timeout, None, BrowserErrorKind.bridge_disconnected),
        (ResultCode.target_lost, None, BrowserErrorKind.tab_closed),
        (ResultCode.unapproved_tab, None, BrowserErrorKind.permission_denied),
        (ResultCode.permission_denied, None, BrowserErrorKind.permission_denied),
        (ResultCode.error, BrowserActionType.navigate, BrowserErrorKind.navigation_failed),
        (ResultCode.error, BrowserActionType.inspect, BrowserErrorKind.bridge_disconnected),
    ],
)
def test_result_code_maps_to_canonical_kind(code, action_type, expected):
    assert result_code_to_error_kind(code, action_type) == expected


def test_host_only_strips_everything_but_host():
    assert host_only("https://user:pw@Sub.Example.com:8443/a/b?q=1#f") == "sub.example.com"
    assert host_only("example.com") == "example.com"
    assert host_only("") == ""


# ── permission check + unique constraint ──────────────────────────────
def test_permission_check_grants_same_host_read_and_navigate(session):
    bs = _make_session_row(session, domain="example.com")
    session.add(
        BrowserTabGrant(
            session_id=bs.id, domain="example.com",
            action_class=BrowserActionClass.navigate.value,
            decision=PermissionDecision.granted.value,
            granted_at=datetime.now(timezone.utc),
        )
    )
    session.commit()
    svc = BrowserPermissionService(session)
    # navigate grant satisfies read AND navigate on same host
    assert svc.check(bs.id, "example.com", BrowserActionClass.read).granted
    assert svc.check(bs.id, "example.com", BrowserActionClass.navigate).granted
    # cross-domain navigate denied
    assert not svc.check(bs.id, "evil.com", BrowserActionClass.navigate).granted


def test_permission_check_expired_and_revoked(session):
    bs = _make_session_row(session, domain="example.com")
    session.add(
        BrowserTabGrant(
            session_id=bs.id, domain="expired.com",
            action_class=BrowserActionClass.read.value,
            decision=PermissionDecision.granted.value,
            granted_at=datetime.now(timezone.utc) - timedelta(hours=2),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
    )
    session.add(
        BrowserTabGrant(
            session_id=bs.id, domain="revoked.com",
            action_class=BrowserActionClass.read.value,
            decision=PermissionDecision.revoked.value,
        )
    )
    session.commit()
    svc = BrowserPermissionService(session)
    assert svc.check(bs.id, "expired.com", BrowserActionClass.read).decision == PermissionDecision.expired
    assert svc.check(bs.id, "revoked.com", BrowserActionClass.read).decision == PermissionDecision.revoked
    assert svc.check(bs.id, "unknown.com", BrowserActionClass.read).decision == PermissionDecision.denied


def test_tab_grant_unique_constraint(session):
    bs = _make_session_row(session)
    session.add(
        BrowserTabGrant(session_id=bs.id, domain="dup.com", action_class="read")
    )
    session.commit()
    session.add(
        BrowserTabGrant(session_id=bs.id, domain="dup.com", action_class="read")
    )
    with pytest.raises(Exception):
        session.commit()
    session.rollback()


# ── migration up/down idempotency ─────────────────────────────────────
def _has_table(path, name) -> bool:
    with sqlite3.connect(path) as c:
        return c.execute(
            "select name from sqlite_master where type='table' and name=?", (name,)
        ).fetchone() is not None


def _expected_head() -> str:
    return ScriptDirectory.from_config(_alembic_config("sqlite://")).get_current_head()


def test_browser_migration_up_down_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    db_path = tmp_path / "b.db"
    uri = f"sqlite:///{db_path}"
    engine = create_engine(uri)

    run_schema_migrations(engine, uri)
    assert _expected_head() == "a1c2e3f4b5d6"
    for t in ("browser_sessions", "browser_tab_grants", "browser_actions"):
        assert _has_table(db_path, t)

    cfg = _alembic_config(uri)
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.downgrade(cfg, "f7d2b9e4a1c6")
    for t in ("browser_sessions", "browser_tab_grants", "browser_actions"):
        assert not _has_table(db_path, t)

    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")
    for t in ("browser_sessions", "browser_tab_grants", "browser_actions"):
        assert _has_table(db_path, t)

    # Re-run whole migration path is a clean no-op.
    run_schema_migrations(engine, uri)


def test_browser_migration_downgrade_guards_missing_tables(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    db_path = tmp_path / "b2.db"
    uri = f"sqlite:///{db_path}"
    engine = create_engine(uri)
    run_schema_migrations(engine, uri)

    with engine.begin() as conn:
        for t in ("browser_actions", "browser_tab_grants", "browser_sessions"):
            conn.exec_driver_sql(f"DROP TABLE {t}")

    cfg = _alembic_config(uri)
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.downgrade(cfg, "f7d2b9e4a1c6")  # must not raise


# ── review fixes: host-only domain, terminal immutability, cascade ────
def test_append_pending_normalizes_full_url_domain_to_host(session):
    # Defense in depth: a caller passing a full URL must never leak
    # path/query into the host-only `browser_actions.domain` column.
    bs = _make_session_row(session)
    store = BrowserActionStore(session)
    action = store.append_pending(
        session_id=bs.id,
        command_id="cmd-url",
        idempotency_key="k-url",
        action_type=BrowserActionType.navigate,
        domain="https://Example.com/path?token=secret#frag",
    )
    assert action.domain == "example.com"


def test_late_ok_result_does_not_overwrite_terminal_failure(session):
    # mark_failed(target_lost) is terminal; a delayed `ok` for the same
    # command_id must not flip the row back to `observed`.
    bs = _make_session_row(session)
    store = BrowserActionStore(session)
    store.append_pending(
        session_id=bs.id,
        command_id="cmd-late",
        idempotency_key="k-late",
        action_type=BrowserActionType.inspect,
        domain="example.com",
    )
    store.mark_in_flight("cmd-late")
    store.mark_failed("cmd-late", result_code=ResultCode.target_lost)
    late = store.record_observed(
        "cmd-late",
        result_code=ResultCode.ok,
        transient={"http_status": 200, "settled": True},
    )
    assert late.status == "failed"
    assert late.result_code == "target_lost"
    assert late.observed_result is None


def test_duplicate_result_does_not_overwrite_terminal_observed(session):
    bs = _make_session_row(session)
    store = BrowserActionStore(session)
    store.append_pending(
        session_id=bs.id,
        command_id="cmd-dup",
        idempotency_key="k-dup",
        action_type=BrowserActionType.inspect,
        domain="example.com",
    )
    store.mark_in_flight("cmd-dup")
    store.record_observed(
        "cmd-dup", result_code=ResultCode.ok, transient={"http_status": 200}
    )
    dup = store.record_observed("cmd-dup", result_code=ResultCode.error)
    assert dup.status == "observed"
    assert dup.result_code == "ok"
    assert dup.observed_result == {"http_status": 200}


def test_delete_conversation_cleans_browser_rows(session):
    from sqlmodel import select

    from cowork.models.browser import BrowserAction
    from cowork.services.conversations import ConversationService

    bs = _make_session_row(session, domain="example.com")
    conv_id = bs.conversation_id
    session.add(
        BrowserTabGrant(
            session_id=bs.id, domain="example.com",
            action_class=BrowserActionClass.navigate.value,
            decision=PermissionDecision.granted.value,
            granted_at=datetime.now(timezone.utc),
        )
    )
    session.commit()
    store = BrowserActionStore(session)
    store.append_pending(
        session_id=bs.id,
        command_id="cmd-del",
        idempotency_key="k-del",
        action_type=BrowserActionType.inspect,
        domain="example.com",
    )

    assert ConversationService(session).delete_conversation(conv_id)

    assert session.exec(
        select(BrowserSession).where(BrowserSession.conversation_id == conv_id)
    ).first() is None
    assert session.exec(
        select(BrowserTabGrant).where(BrowserTabGrant.session_id == bs.id)
    ).first() is None
    assert session.exec(
        select(BrowserAction).where(BrowserAction.session_id == bs.id)
    ).first() is None


def test_browser_fks_carry_on_delete_cascade(tmp_path, monkeypatch):
    # The migration must emit ON DELETE CASCADE so FK-enforcing engines
    # (Postgres) don't reject deleting a conversation with a browser session.
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    get_app_settings.cache_clear()
    db_path = tmp_path / "c.db"
    uri = f"sqlite:///{db_path}"
    engine = create_engine(uri)
    run_schema_migrations(engine, uri)

    with sqlite3.connect(db_path) as c:
        def fk_actions(table):
            # pragma foreign_key_list: (id, seq, table, from, to, on_update, on_delete, match)
            return {
                (row[2], row[3]): row[6]
                for row in c.execute(f"PRAGMA foreign_key_list({table})")
            }

        assert fk_actions("browser_sessions")[("conversations", "conversation_id")] == "CASCADE"
        assert fk_actions("browser_tab_grants")[("browser_sessions", "session_id")] == "CASCADE"
        assert fk_actions("browser_actions")[("browser_sessions", "session_id")] == "CASCADE"
