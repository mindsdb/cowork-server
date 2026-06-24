"""Connection-lifecycle endpoints (slice 2): test / health / reconnect.

Mounts just the connections router on a bare FastAPI app (the pattern used by
test_artifact_shares) and points the vault at a per-test temp dir via
COWORK_VAULT_DIR, so the real ~/.cowork/data-vault is never touched. The live
probe (which would call an LLM) is monkeypatched — these tests cover the
endpoint wiring, persistence of the verdict, and health/reconnect logic, not
the probe itself.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints.connectors import connections as conn_endpoints
# Aliased so pytest doesn't try to collect the response model as a test class.
from cowork.schemas.connectors import TestConnectionResponse as TestConnResp
from cowork.services.connectors import health as H
from cowork.services.connectors.encrypted_vault import build_vault


@pytest.fixture()
def vault_dir(tmp_path: Path, monkeypatch) -> Path:
    # ConnectorSettings reads COWORK_VAULT_DIR fresh on each _vault() call, so a
    # monkeypatched env var redirects the whole service onto a temp vault.
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture()
def client(vault_dir: Path) -> TestClient:
    app = FastAPI()
    app.include_router(conn_endpoints.router, prefix="/api/v1/connectors/connections")
    return TestClient(app)


def _seed(vault_dir: Path, engine: str, name: str, fields: dict):
    build_vault(vault_dir).save(engine, name, fields)


# ── test endpoint ───────────────────────────────────────────────────────────

def test_test_endpoint_404_for_missing_connection(client: TestClient):
    r = client.post("/api/v1/connectors/connections/postgres/nope/test")
    assert r.status_code == 404


def test_test_endpoint_pass_stamps_metadata(client: TestClient, vault_dir: Path, monkeypatch):
    _seed(vault_dir, "postgres", "prod", {"host": "db", "password": "pw", "_method": "uri"})

    async def fake_run_test(engine, name, credentials):
        # The endpoint must pass the saved (decrypted) credentials through.
        assert credentials.get("password") == "pw"
        return TestConnResp(
            ok=True, result=H.TEST_PASS, summary="SELECT 1 worked.",
            health=H.HEALTHY, health_detail="Last test passed.",
        )

    monkeypatch.setattr("cowork.services.connectors.test_runner.run_test", fake_run_test)

    r = client.post("/api/v1/connectors/connections/postgres/prod/test")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["result"] == "pass"
    assert body["tested_at"]  # stamped + echoed back

    # The verdict was persisted onto the record and now drives health.
    record = build_vault(vault_dir).read_record("postgres", "prod")
    assert record["last_test_result"] == "pass"
    detail = client.get("/api/v1/connectors/connections/postgres/prod").json()
    assert detail["health"] == "healthy"
    assert detail["last_test_result"] == "pass"


def test_test_endpoint_failure_persists_error(client: TestClient, vault_dir: Path, monkeypatch):
    _seed(vault_dir, "postgres", "prod", {"host": "db", "password": "bad"})

    async def fake_run_test(engine, name, credentials):
        return TestConnResp(
            ok=False, result=H.TEST_FAIL, error="password authentication failed",
            follow_up="Check the password.", health=H.BROKEN,
        )

    monkeypatch.setattr("cowork.services.connectors.test_runner.run_test", fake_run_test)

    r = client.post("/api/v1/connectors/connections/postgres/prod/test")
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert "authentication failed" in r.json()["error"]

    record = build_vault(vault_dir).read_record("postgres", "prod")
    assert record["last_test_result"] == "fail"
    assert "authentication failed" in record["last_test_error"]
    # Health now reflects the failed probe.
    assert client.get("/api/v1/connectors/connections/postgres/prod/health").json()["health"] == "broken"


def test_test_endpoint_untestable_connector_is_not_marked_broken(client: TestClient, vault_dir: Path):
    """A connector with no registry spec can't be live-probed. Testing it must
    NOT persist a failure (which compute_health would read as BROKEN forever);
    it stays UNKNOWN and the record carries no last_test_result."""
    # An engine id that has no JSON spec in the registry.
    _seed(vault_dir, "totally-made-up-engine", "x", {"token": "t"})

    r = client.post("/api/v1/connectors/connections/totally-made-up-engine/x/test")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["result"] == ""          # untestable, not "fail"
    assert body["health"] == "unknown"

    # Nothing was stamped, so health stays unknown — not broken.
    record = build_vault(vault_dir).read_record("totally-made-up-engine", "x")
    assert record.get("last_test_result") is None
    assert client.get(
        "/api/v1/connectors/connections/totally-made-up-engine/x/health"
    ).json()["health"] == "unknown"


# ── health endpoint ──────────────────────────────────────────────────────────

def test_health_endpoint_404_for_missing(client: TestClient):
    assert client.get("/api/v1/connectors/connections/x/y/health").status_code == 404


def test_health_endpoint_oauth_expiring(client: TestClient, vault_dir: Path):
    soon = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    _seed(vault_dir, "google_drive", "me@x.com", {
        "auth_type": "oauth", "access_token": "at", "refresh_token": "rt", "expires_at": soon,
    })
    body = client.get("/api/v1/connectors/connections/google_drive/me@x.com/health").json()
    assert body["health"] == "expiring_soon"
    assert body["is_oauth"] is True
    assert body["reconnectable"] is True
    assert body["expires_at"] == soon


def test_health_endpoint_nonoauth_untested_unknown(client: TestClient, vault_dir: Path):
    _seed(vault_dir, "postgres", "prod", {"host": "db", "password": "pw"})
    body = client.get("/api/v1/connectors/connections/postgres/prod/health").json()
    assert body["health"] == "unknown"
    assert body["is_oauth"] is False
    assert body["reconnectable"] is False


# ── list/detail surface lifecycle fields ─────────────────────────────────────

def test_list_surfaces_health_and_encrypted(client: TestClient, vault_dir: Path):
    _seed(vault_dir, "postgres", "prod", {"host": "db", "password": "pw"})
    rows = client.get("/api/v1/connectors/connections/").json()
    assert len(rows) == 1
    row = rows[0]
    assert row["engine"] == "postgres"
    assert row["health"] == "unknown"
    assert row["encrypted"] is True
    assert "last_tested_at" in row


# ── reconnect endpoint ───────────────────────────────────────────────────────

def test_reconnect_404_for_missing(client: TestClient):
    assert client.post("/api/v1/connectors/connections/x/y/reconnect").status_code == 404


def test_reconnect_nonoauth_reports_credentials_method(client: TestClient, vault_dir: Path):
    _seed(vault_dir, "postgres", "prod", {"host": "db", "password": "pw"})
    body = client.post("/api/v1/connectors/connections/postgres/prod/reconnect").json()
    assert body["method"] == "credentials"
    assert body["refreshed"] is False
    assert body["service"] is None


def test_reconnect_oauth_attempts_silent_refresh(client: TestClient, vault_dir: Path, monkeypatch):
    expired = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _seed(vault_dir, "google_drive", "me@x.com", {
        "auth_type": "oauth", "access_token": "at", "refresh_token": "rt", "expires_at": expired,
    })

    def fake_refresh_one(engine, name, oauth_settings):
        # Simulate a successful silent refresh by re-stamping a future expiry
        # well beyond the EXPIRING_WINDOW so health reads as fully healthy.
        future = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        v = build_vault(vault_dir)
        fields = v.load(engine, name)
        fields["expires_at"] = future
        v.save(engine, name, fields)
        return True

    monkeypatch.setattr(conn_endpoints.google_service, "refresh_one", fake_refresh_one)

    body = client.post("/api/v1/connectors/connections/google_drive/me@x.com/reconnect").json()
    assert body["method"] == "oauth"
    assert body["service"] == "google-drive"
    assert body["refreshed"] is True
    assert body["health"] == "healthy"  # recomputed off the refreshed expiry


def test_reconnect_refresh_preserves_secure_keys(vault_dir: Path, monkeypatch):
    """refresh_one re-saves the record; it must carry the prior secure_keys
    forward so a refreshed token isn't downgraded to plaintext in the detail
    response."""
    from cowork.common.settings.app_settings import OAuthSettings
    from cowork.services.connectors.oauth.google import google_service

    v = build_vault(vault_dir)
    v.save("google_drive", "me@x.com", {
        "auth_type": "oauth", "access_token": "old", "refresh_token": "rt",
        "expires_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
    }, secure_keys=["access_token", "refresh_token"])

    # Stub the network token exchange.
    def fake_json_request(url, **kwargs):
        return {"access_token": "new-at", "expires_in": 3600}

    monkeypatch.setattr("cowork.services.connectors.oauth.google._json_request", fake_json_request)
    monkeypatch.setattr(
        google_service, "_resolve_credentials", lambda service, settings: ("cid", "csecret")
    )

    assert google_service.refresh_one("google_drive", "me@x.com", OAuthSettings()) is True
    record = v.read_record("google_drive", "me@x.com")
    assert record["fields"]["access_token"] == "new-at"   # refreshed
    assert record["secure_keys"] == ["access_token", "refresh_token"]  # preserved


def test_reconnect_oauth_falls_back_when_refresh_fails(client: TestClient, vault_dir: Path, monkeypatch):
    expired = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _seed(vault_dir, "google_drive", "me@x.com", {
        "auth_type": "oauth", "access_token": "at", "refresh_token": "rt", "expires_at": expired,
    })
    monkeypatch.setattr(conn_endpoints.google_service, "refresh_one", lambda *a, **k: False)

    body = client.post("/api/v1/connectors/connections/google_drive/me@x.com/reconnect").json()
    assert body["method"] == "oauth"
    assert body["refreshed"] is False
    assert "Sign in again" in body["message"]
