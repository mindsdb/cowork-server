"""ENG-462: secret-bearing responses must not be persisted to a client's
on-disk HTTP cache, and providers_json must not echo the raw key.

Drives the real app over TestClient for the header behaviour and unit-tests
the masking helper directly.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from cowork.api.v1.endpoints.responses import _SSE_HEADERS
from cowork.server import app
from cowork.services.settings import _mask_provider_keys

client = TestClient(app)


def test_sse_headers_are_no_store():
    # The chat stream can echo secrets (e.g. a key in generated code).
    assert _SSE_HEADERS["Cache-Control"] == "no-store"


def test_settings_list_sends_no_store():
    r = client.get("/api/v1/settings/")
    assert r.status_code == 200
    assert r.headers.get("cache-control") == "no-store"


def test_reveal_key_sends_no_store():
    r = client.get("/api/v1/settings/reveal-key/openai")
    assert r.status_code == 200
    assert r.headers.get("cache-control") == "no-store"


def test_non_settings_route_is_not_forced_no_store():
    # The middleware is scoped to /settings and /connectors/oauth; health
    # must not be swept in.
    r = client.get("/api/v1/health/")
    assert r.headers.get("cache-control") != "no-store"


def test_oauth_credentials_sends_no_store():
    # GET .../oauth/{engine}/credentials returns a raw client_secret. Even on
    # a 404 (unknown engine) the middleware must still stamp no-store, since
    # it applies by path prefix regardless of status code.
    r = client.get("/api/v1/connectors/oauth/unknown-engine/credentials")
    assert r.status_code == 404
    assert r.headers.get("cache-control") == "no-store"


def test_mask_provider_keys_helper():
    raw = json.dumps([
        {"type": "openai", "apiKey": "sk-proj-SECRET", "baseUrl": "https://x"},
        {"type": "anthropic", "apiKey": ""},
    ])
    out = json.loads(_mask_provider_keys(raw))
    assert out[0]["apiKey"] == "***"          # raw key masked
    assert out[0]["type"] == "openai"          # non-key fields preserved
    assert out[0]["baseUrl"] == "https://x"
    assert out[1]["apiKey"] == ""              # empty stays empty (not "***")
    assert _mask_provider_keys("not json") == "[]"   # fails closed
    assert _mask_provider_keys("") == "[]"


def test_providers_json_masked_in_list_response():
    secret = "sk-proj-LISTLEAK"
    client.put(
        "/api/v1/settings/providers_json",
        json={"value": json.dumps([{"type": "openai", "apiKey": secret}])},
    )
    r = client.get("/api/v1/settings/")
    assert r.status_code == 200
    entry = next(s for s in r.json() if s["key"] == "providers_json")
    assert secret not in (entry["value"] or "")
    assert "***" in entry["value"]
