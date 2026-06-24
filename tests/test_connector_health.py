"""Provider-agnostic connection health (slice 2).

Pure unit tests over ``cowork.services.connectors.health.compute_health`` plus
the defensive runtime loader. No app, no network, no LLM — health is derived
entirely from the record's fields + stamped last-test result, with ``now``
injected for determinism.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from cowork.services.connectors import health as H


NOW = datetime(2026, 6, 24, 12, 0, 0, tzinfo=timezone.utc)


def _oauth_fields(*, expires_in=None, refresh="rt-123", expires_at=None):
    fields = {"auth_type": "oauth", "access_token": "at-abc"}
    if refresh:
        fields["refresh_token"] = refresh
    if expires_at is not None:
        fields["expires_at"] = expires_at
    elif expires_in is not None:
        fields["expires_at"] = (NOW + timedelta(seconds=expires_in)).isoformat()
    return fields


# ── is_oauth ──────────────────────────────────────────────────────────────

def test_is_oauth_detects_marker_and_tokens():
    assert H.is_oauth({"auth_type": "oauth"}) is True
    assert H.is_oauth({"access_token": "x"}) is True
    assert H.is_oauth({"refresh_token": "x"}) is True
    assert H.is_oauth({"host": "db", "password": "p"}) is False
    assert H.is_oauth(None) is False


# ── OAuth token expiry math ─────────────────────────────────────────────────

def test_oauth_far_future_expiry_is_healthy():
    s = H.compute_health(_oauth_fields(expires_in=7 * 24 * 3600), now=NOW)
    assert s.status == H.HEALTHY
    assert s.reconnectable is True  # OAuth is always reconnectable
    assert s.expires_at  # echoed back


def test_oauth_expiring_within_window_is_expiring_soon():
    s = H.compute_health(_oauth_fields(expires_in=3600), now=NOW)  # 1h < 24h window
    assert s.status == H.EXPIRING_SOON
    assert s.reconnectable is True


def test_oauth_expired_with_refresh_token_is_expiring_soon_not_broken():
    # Recoverable: a refresh token can mint a new access token.
    s = H.compute_health(_oauth_fields(expires_in=-3600, refresh="rt-123"), now=NOW)
    assert s.status == H.EXPIRING_SOON
    assert s.reconnectable is True


def test_oauth_expired_without_refresh_token_is_broken():
    s = H.compute_health(_oauth_fields(expires_in=-3600, refresh=None), now=NOW)
    assert s.status == H.BROKEN
    assert s.reconnectable is True


def test_oauth_without_expiry_is_unknown_until_tested():
    s = H.compute_health(_oauth_fields(expires_in=None, expires_at=""), now=NOW)
    assert s.status == H.UNKNOWN
    assert s.reconnectable is True


def test_oauth_without_expiry_but_passed_test_is_healthy():
    s = H.compute_health(
        _oauth_fields(expires_in=None, expires_at=""),
        last_test_result=H.TEST_PASS, now=NOW,
    )
    assert s.status == H.HEALTHY


def test_oauth_garbage_expiry_is_treated_as_no_expiry():
    s = H.compute_health(_oauth_fields(expires_at="not-a-date"), now=NOW)
    assert s.status == H.UNKNOWN


def test_naive_expiry_timestamp_is_assumed_utc():
    # No tzinfo on the stored expiry — must not crash, treated as UTC.
    naive = (NOW + timedelta(hours=1)).replace(tzinfo=None).isoformat()
    s = H.compute_health(_oauth_fields(expires_at=naive), now=NOW)
    assert s.status == H.EXPIRING_SOON


# ── last-test result precedence ─────────────────────────────────────────────

def test_failed_test_forces_broken_even_for_healthy_oauth_token():
    # A valid-looking token but the last live probe failed → BROKEN wins.
    s = H.compute_health(
        _oauth_fields(expires_in=7 * 24 * 3600),
        last_test_result=H.TEST_FAIL, now=NOW,
    )
    assert s.status == H.BROKEN
    assert s.reconnectable is True


def test_nonoauth_passed_test_is_healthy_not_reconnectable():
    s = H.compute_health(
        {"host": "db", "password": "p"},
        last_test_result=H.TEST_PASS, now=NOW,
    )
    assert s.status == H.HEALTHY
    # A healthy non-OAuth connection has nothing to "reconnect".
    assert s.reconnectable is False


def test_nonoauth_failed_test_is_broken_and_reconnectable():
    s = H.compute_health(
        {"host": "db", "password": "p"},
        last_test_result=H.TEST_FAIL, now=NOW,
    )
    assert s.status == H.BROKEN
    assert s.reconnectable is True


def test_nonoauth_untested_is_unknown():
    s = H.compute_health({"host": "db"}, now=NOW)
    assert s.status == H.UNKNOWN
    assert s.reconnectable is False


# ── safe_runtime_load ───────────────────────────────────────────────────────

class _GoodVault:
    def load(self, engine, name):
        return {"host": "db", "password": "p"}


class _MissingVault:
    def load(self, engine, name):
        return None


class _RaisingVault:
    def __init__(self):
        self.stamped = None

    def load(self, engine, name):
        raise RuntimeError("decrypt boom")

    def record_test_result(self, engine, name, *, result, error=None):
        self.stamped = (engine, name, result, error)
        return True


def test_safe_runtime_load_returns_fields_on_success():
    assert H.safe_runtime_load(_GoodVault(), "postgres", "prod") == {"host": "db", "password": "p"}


def test_safe_runtime_load_returns_empty_for_missing():
    assert H.safe_runtime_load(_MissingVault(), "postgres", "prod") == {}


def test_safe_runtime_load_stamps_broken_and_returns_empty_on_error():
    vault = _RaisingVault()
    out = H.safe_runtime_load(vault, "postgres", "prod")
    assert out == {}  # degrades gracefully, no raise
    # ...and the connection was stamped broken so the UI can prompt reconnect.
    assert vault.stamped is not None
    engine, name, result, error = vault.stamped
    assert (engine, name) == ("postgres", "prod")
    assert result == H.TEST_FAIL
    assert error
