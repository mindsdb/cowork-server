"""Behaviour of the optional bearer-token auth layer.

Exercises BearerTokenMiddleware directly on a tiny app rather than create_app()
so it doesn't depend on the settings singleton. The CORS-ordering test guards
the reason the middleware is registered inner of CORS: a browser must receive
the 401, not an opaque CORS failure.
"""
from __future__ import annotations

import os
import stat

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from cowork.auth_middleware import (
    BearerTokenMiddleware,
    _read_token,
    ensure_auth_token,
    sync_auth_token,
)

TOKEN = "test-token-abc123"
ORIGIN = "http://localhost:1234"
WEBHOOK = "/api/v1/channels/slack/events"


def _client() -> TestClient:
    app = FastAPI()

    @app.get("/api/v1/health")
    def health():
        return {"ok": True}

    @app.get("/api/v1/projects/")
    def projects():
        return {"projects": []}

    @app.post(WEBHOOK)
    async def slack_webhook():
        return {"ack": True}

    # Mirror create_app's ordering: bearer added first (inner), CORS last
    # (outer) so a 401 still flows back out through CORS.
    app.add_middleware(BearerTokenMiddleware, token=TOKEN, exempt_paths={WEBHOOK})
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[ORIGIN],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return TestClient(app)


def test_missing_token_is_401():
    assert _client().get("/api/v1/projects/").status_code == 401


def test_wrong_token_is_401():
    res = _client().get("/api/v1/projects/", headers={"Authorization": "Bearer nope"})
    assert res.status_code == 401


def test_valid_token_passes():
    res = _client().get("/api/v1/projects/", headers={"Authorization": f"Bearer {TOKEN}"})
    assert res.status_code == 200


def test_health_is_exempt_without_token():
    assert _client().get("/api/v1/health").status_code == 200


def test_channel_webhook_is_exempt_without_token():
    assert _client().post(WEBHOOK).status_code == 200


def test_401_carries_cors_headers():
    # CORS is the outer layer, so even the auth 401 must include
    # Access-Control-Allow-Origin — otherwise the browser reports a CORS error
    # and the caller never sees the 401.
    res = _client().get("/api/v1/projects/", headers={"Origin": ORIGIN})
    assert res.status_code == 401
    assert res.headers.get("access-control-allow-origin") == ORIGIN


def test_options_preflight_is_allowed_without_token():
    res = _client().options(
        "/api/v1/projects/",
        headers={"Origin": ORIGIN, "Access-Control-Request-Method": "GET"},
    )
    assert res.status_code in (200, 204)
    assert res.headers.get("access-control-allow-origin") == ORIGIN


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")
def test_generated_token_file_is_private(tmp_path):
    env = tmp_path / ".env"
    token = ensure_auth_token(env)
    assert token and _read_token(env) == token
    assert stat.S_IMODE(env.stat().st_mode) == 0o600


def test_sync_auth_token_mirrors_and_is_idempotent(tmp_path):
    env = tmp_path / ".env"
    sync_auth_token(env, "fixed-123")
    assert _read_token(env) == "fixed-123"
    sync_auth_token(env, "fixed-123")  # no duplicate line / no error
    assert _read_token(env) == "fixed-123"
