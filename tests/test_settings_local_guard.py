"""reveal-key and /raw must refuse non-loopback callers (ENG-457).

These endpoints return unmasked provider secrets (a single key, or the whole
dotenv). `guards.require_local` is defense-in-depth for a network-exposed
deployment — e.g. a self-host compose that binds 0.0.0.0 — so even with no
app-layer auth they only answer a loopback client. The desktop sidecar + UI
talk over 127.0.0.1, so the legitimate flow is unaffected.
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def _request(host):
    """Minimal stand-in for a Starlette Request — only `.client.host` matters."""
    client = SimpleNamespace(host=host) if host is not None else None
    return SimpleNamespace(client=client)


def test_require_local_allows_loopback():
    from cowork.api.v1.endpoints.guards import require_local

    require_local(_request("127.0.0.1"))
    require_local(_request("::1"))


@pytest.mark.parametrize("host", ["10.0.0.5", "0.0.0.0", "192.168.1.10", "", None])
def test_require_local_rejects_non_loopback(host):
    from cowork.api.v1.endpoints.guards import require_local

    with pytest.raises(HTTPException) as exc:
        require_local(_request(host))
    assert exc.value.status_code == 403


def test_reveal_key_blocks_non_local_before_db():
    from cowork.api.v1.endpoints.settings import reveal_key
    from cowork.db.scoped import LOCAL_SCOPE

    # The guard is the first statement, so a non-local caller is rejected before
    # the session/DB is ever touched — session=None is safe here.
    with pytest.raises(HTTPException) as exc:
        reveal_key("openai", session=None, scope=LOCAL_SCOPE, request=_request("203.0.113.7"))
    assert exc.value.status_code == 403


def test_read_raw_blocks_non_local():
    from cowork.api.v1.endpoints.settings import read_raw_settings

    with pytest.raises(HTTPException) as exc:
        read_raw_settings(request=_request("203.0.113.7"))
    assert exc.value.status_code == 403


def test_raw_endpoints_are_disabled_in_org_mode(monkeypatch):
    # /raw reads+writes deployment-global state (the dotenv + global settings
    # rows every org falls back to) and carries no tenant scope: loopback alone
    # isn't a boundary once one deployment serves many orgs.
    from cowork.api.v1.endpoints.settings import read_raw_settings, write_raw_settings
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    try:
        with pytest.raises(HTTPException) as exc:
            read_raw_settings(request=_request("127.0.0.1"))
        assert exc.value.status_code == 501
        with pytest.raises(HTTPException) as exc:
            write_raw_settings(body=None, session=None, request=_request("127.0.0.1"))
        assert exc.value.status_code == 501
    finally:
        get_app_settings.cache_clear()
