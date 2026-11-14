"""reveal-key and /raw must refuse non-loopback callers (ENG-457).

These endpoints return unmasked provider secrets (a single key, or the whole
dotenv). `guards.require_local` is defense-in-depth for a network-exposed
deployment — e.g. a self-host compose that binds 0.0.0.0 — so even with no
app-layer auth they only answer a loopback client. The desktop sidecar + UI
talk over 127.0.0.1, so the legitimate flow is unaffected.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def _request(host, *, headers=None, server=("127.0.0.1", 26866)):
    """Minimal stand-in for the request boundary used by the guard.

    ``server`` is ``scope["server"]``, the local address of the socket the
    request landed on; the desktop sidecar's is loopback."""
    client = SimpleNamespace(host=host) if host is not None else None
    return SimpleNamespace(client=client, headers=headers or {}, scope={"server": server})


def test_require_local_allows_loopback():
    from cowork.api.v1.endpoints.guards import require_local

    require_local(_request("127.0.0.1"))
    require_local(_request("::1"))
    require_local(_request("127.0.0.1", headers={"host": "localhost:26866"}))
    require_local(_request("::1", headers={"host": "[::1]:26866"}))


def test_require_local_rejects_dns_rebinding_and_cross_site_origins(monkeypatch):
    from cowork.api.v1.endpoints.guards import require_local
    from cowork.common.settings.app_settings import get_app_settings

    with pytest.raises(HTTPException, match="local host"):
        require_local(_request("127.0.0.1", headers={"host": "attacker.example"}))

    monkeypatch.setenv("COWORK_ALLOWED_ORIGINS", '["http://localhost:5173"]')
    get_app_settings.cache_clear()
    try:
        with pytest.raises(HTTPException, match="trusted local origin"):
            require_local(_request(
                "127.0.0.1",
                headers={"host": "127.0.0.1:26866", "origin": "https://attacker.example"},
            ))
    finally:
        get_app_settings.cache_clear()


def test_require_local_trusts_the_test_client_host_only_through_the_fixture(monkeypatch):
    from cowork.api.v1.endpoints import guards

    monkeypatch.setattr(guards, "_TRUSTED_LOOPBACK_HOSTS", guards._TRUSTED_LOOPBACK_HOSTS - {"testserver"})
    with pytest.raises(HTTPException, match="local host"):
        guards.require_local(_request("127.0.0.1", headers={"host": "testserver"}))


def test_testserver_does_not_appear_in_production_code():
    import cowork

    root = Path(cowork.__file__).parent
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "testserver" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


@pytest.mark.parametrize("host", ["10.0.0.5", "0.0.0.0", "192.168.1.10", "", None])
def test_require_local_rejects_non_loopback(host):
    from cowork.api.v1.endpoints.guards import require_local

    with pytest.raises(HTTPException) as exc:
        require_local(_request(host))
    assert exc.value.status_code == 403


@pytest.mark.parametrize(
    "peer,server",
    [
        ("127.0.0.1", ("127.0.0.1", 26866)),
        ("::1", ("::1", 26866)),
        ("::ffff:127.0.0.1", ("::ffff:127.0.0.1", 26866)),
        ("127.0.0.1", ["localhost", 80]),
    ],
)
def test_require_local_allows_a_loopback_socket(peer, server):
    """The desktop sidecar binds 127.0.0.1. A dual-stack bind reports both ends
    of the same connection as ``::ffff:127.0.0.1``, and Starlette's TestClient
    hands the server over as a list."""
    from cowork.api.v1.endpoints.guards import require_local

    require_local(_request(peer, headers={"host": "localhost:26866"}, server=server))


@pytest.mark.parametrize(
    "server",
    [("10.105.98.84", 9010), ("172.17.0.2", 9010), ("::ffff:10.0.0.5", 9010), ("/run/cowork.sock", None), None],
)
def test_require_local_refuses_a_loopback_peer_on_a_socket_that_is_not_loopback(server):
    """Behind uvicorn's proxy middleware an X-Forwarded-For sets the peer to
    127.0.0.1, but the request still landed on the pod's own address. A unix
    socket reports its path instead of an address, and a proxy in front of
    one would make every request look local, so it gets the same refusal, as
    does a transport that reports nothing."""
    from cowork.api.v1.endpoints.guards import require_local

    with pytest.raises(HTTPException) as exc:
        require_local(_request("127.0.0.1", headers={"host": "localhost"}, server=server))
    assert (exc.value.status_code, exc.value.detail) == (403, "local requests only")


def test_reveal_key_blocks_non_local():
    # require_local is a declared dependency now (not called in the handler
    # body), so this goes through the real router — a bare handler call would
    # no longer prove anything is enforced.
    resp = _raw_client(local=False).get("/api/v1/settings/reveal-key/openai")
    assert resp.status_code == 403


def _raw_client(*, local: bool) -> "TestClient":
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cowork.api.v1.router import api_router

    app = FastAPI()
    app.include_router(api_router)
    if local:
        return TestClient(app, client=("127.0.0.1", 50000))
    return TestClient(app)


def test_read_raw_blocks_non_local():
    # require_local/require_local_tenancy are declared dependencies now (not
    # called in the handler body), so this goes through the real router —
    # a bare handler call would no longer prove anything is enforced.
    resp = _raw_client(local=False).get("/api/v1/settings/raw")
    assert resp.status_code == 403


def test_raw_endpoints_are_disabled_in_org_mode(monkeypatch):
    # /raw reads+writes deployment-global state (the dotenv + global settings
    # rows every org falls back to) and carries no tenant scope: loopback alone
    # isn't a boundary once one deployment serves many orgs.
    from cowork.common.settings.app_settings import get_app_settings

    # local=True: the caller is loopback so require_local passes — the
    # detail is what proves the tenancy guard did the refusing.
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    try:
        client = _raw_client(local=True)

        get_resp = client.get("/api/v1/settings/raw")
        assert get_resp.status_code == 403
        assert get_resp.json()["detail"] == "not available in org deployments"

        post_resp = client.post("/api/v1/settings/raw", json={"content": ""})
        assert post_resp.status_code == 403
        assert post_resp.json()["detail"] == "not available in org deployments"
    finally:
        get_app_settings.cache_clear()
