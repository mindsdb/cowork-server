"""ENG-2094: what the declarations actually do to a request, through create_app().

The class-level tests in test_permissions.py run on a scratch app. These go
through the real router stack, because the bugs these cover were all in the
gap between "the class is right" and "the route got the class it meant to".
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cowork.server import create_app

#: Valid gateway-injected identity. TrustedHeaderMiddleware wants Keycloak
#: UUIDs and 401s anything else before the route's own declaration runs, so a
#: test about the declaration has to get past it first.
MEMBER_HEADERS = {
    "X-User-Id": "11111111-1111-4111-8111-111111111111",
    "X-Organization-Id": "22222222-2222-4222-8222-222222222222",
}


@pytest.fixture(autouse=True)
def _reset_app_settings():
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture()
def org_client(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    return TestClient(create_app())


def _local_client() -> TestClient:
    return TestClient(create_app(), client=("127.0.0.1", 50000))


def test_options_on_responses_is_open_in_org_mode(monkeypatch):
    """A route declared OpenByDesign has to actually be open.

    The responses router used to declare AuthenticatedInOrgMode at router
    level, and FastAPI ADDS a route-level dependency to its router's rather
    than substituting for it — so the OPTIONS route carried both and answered
    401, under a comment saying it "carries its own OpenByDesign instead".

    A bare OPTIONS (no CORS preflight headers at all) under
    identity_enforce=audit, not a real preflight: TrustedHeaderMiddleware's
    own OPTIONS bypass requires Origin + Access-Control-Request-Method
    together now, and asserting against that pair here would make this test
    describe the middleware's bypass, not the route's own dependency, and
    break the moment that pair is tightened further. Audit mode lets a
    request with no identity through regardless of method, so what actually
    answers 200 here is the route's own OpenByDesign — the exact thing this
    test exists to prove.
    """
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    client = TestClient(create_app())

    resp = client.options("/api/v1/responses/")

    assert resp.status_code == 200
    assert resp.json() == {"message": "OK"}


def test_health_is_open_in_org_mode(org_client):
    # The pre-auth readiness probe. If this ever needs a credential, the
    # kubelet cannot supply one.
    assert org_client.get("/api/v1/health/").status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/settings/",
        "/api/v1/settings/configured",
        "/api/v1/settings/recommended-models",
    ],
)
def test_settings_reads_require_identity_in_org_mode(org_client, path):
    """These return the deployment-global config, not the caller's.

    Masking every secret to null bounds the damage; it is not a reason the
    route needs no caller. `settings` is in _TENANCY_DEFERRED_TABLES
    (cowork/db/scoped.py), so what is left after masking still describes the
    whole deployment.
    """
    assert org_client.get(path).status_code == 401


def test_validate_provider_requires_identity_in_org_mode(org_client):
    # It reads nothing stored, but it makes the server fetch a caller-supplied
    # base_url from inside the pod and returns the upstream status, which is a
    # reachability probe for anything the caller cannot reach directly.
    resp = org_client.post(
        "/api/v1/settings/validate-provider",
        json={"provider": "openai-compatible", "apiKey": "x", "baseUrl": "http://10.0.0.5:8080"},
    )

    assert resp.status_code == 401


def test_test_providers_requires_identity_in_org_mode(org_client):
    resp = org_client.post("/api/v1/settings/test-providers", json={"providers": []})

    assert resp.status_code == 401


def test_logout_is_loopback_only_on_desktop():
    """Local mode is where logout destroys, so local mode is where it needs
    the guard. A simple POST gets no CORS preflight, so without this any page
    the user has open can sign the desktop install out."""
    remote = TestClient(create_app())

    assert remote.post("/api/v1/settings/logout").status_code == 403


def test_logout_still_answers_loopback_on_desktop():
    resp = _local_client().post("/api/v1/settings/logout")

    assert resp.status_code == 200


def test_logout_stays_reachable_in_org_mode(org_client):
    """The org-mode no-op is deliberate: one member signing out must not wipe
    the keys the whole org runs on, so it answers 200 and deletes nothing
    rather than 403ing. The loopback guard has to stay out of that path — an
    org member reaches this over the network, never over 127.0.0.1."""
    resp = org_client.post("/api/v1/settings/logout", headers=MEMBER_HEADERS)

    assert resp.status_code == 200
    assert resp.json()["deleted"] == []


def test_audit_mode_warns_that_the_front_door_still_enforces(monkeypatch, caplog):
    """The rollback lever stops half way, and the operator hears it at boot.

    identity_enforce is TrustedHeaderMiddleware's flag. The permission classes
    deliberately do not read it (AuthenticatedInOrgMode's docstring says why),
    so reaching for audit to unblock traffic leaves every identity-requiring
    route still answering 401. Without this the operator learns that from the
    401s.
    """
    import logging

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()

    with caplog.at_level(logging.WARNING):
        create_app()

    warnings = [r.message for r in caplog.records if "identity_enforce" in r.message]
    assert warnings, "org + audit booted without saying the front door still enforces"
    assert "ENG-2094" in warnings[0]


def test_enforce_mode_boots_quietly(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.delenv("COWORK_IDENTITY_ENFORCE", raising=False)
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()

    with caplog.at_level(logging.WARNING):
        create_app()

    assert [r.message for r in caplog.records if "identity_enforce" in r.message] == []


# ── the routes only the desktop app calls that hand out or write secrets ──

#: What a forged request carries. Inside the pod uvicorn reads only
#: X-Forwarded-For. CF-Connecting-IP and X-Forwarded-Host are what an
#: ingress turns into the client address and the upstream Host; they are sent
#: here as an attacker sends them.
FORGED_LOOPBACK_HEADERS = {
    "X-Forwarded-For": "127.0.0.1",
    "CF-Connecting-IP": "127.0.0.1",
    "X-Forwarded-Host": "localhost",
}

#: The detail each guard on the way in answers with. A desktop answer that
#: carries none of them came from the route's own handler.
GUARD_DETAILS = {
    "local requests only",
    "local host required",
    "trusted local origin required",
    "not available in org deployments",
}

_SAVE_BODY = {"connector_id": "no-such-connector", "method": "x", "name": "", "values": {}}

#: (method, path, body, the handler's own status for a desktop caller).
SECRET_ROUTES = [
    pytest.param("GET", "/api/v1/settings/reveal-key/anthropic", None, 200, id="reveal-key"),
    pytest.param("POST", "/api/v1/connectors/connections/save", _SAVE_BODY, 404, id="save"),
    pytest.param(
        "POST", "/api/v1/connectors/connections/validate-and-save", _SAVE_BODY, 422, id="validate-and-save"
    ),
    pytest.param(
        "PATCH",
        "/api/v1/connectors/connections/no-such-engine/no-such-name/token",
        {"status": "needs_reconnect"},
        404,
        id="token",
    ),
    pytest.param("GET", "/api/v1/connectors/oauth/no-such-engine/credentials", None, 404, id="credentials"),
    pytest.param(
        "POST", "/api/v1/connectors/oauth/no-such-engine/mcp/identity", {"access_token": "t"}, 404, id="mcp-identity"
    ),
]


def _forwarding_client(*, socket_address: str) -> TestClient:
    """The image's own path: uvicorn trusts X-Forwarded-For from every peer
    (``--forwarded-allow-ips '*'``) and rewrites the client address from it.
    ``socket_address`` sets ``scope["server"]``, the local address the
    request landed on."""
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    return TestClient(
        ProxyHeadersMiddleware(create_app(), trusted_hosts="*"),
        base_url=f"http://{socket_address}",
        client=("10.105.65.111", 40000),
    )


@pytest.mark.parametrize("method,path,body,desktop_status", SECRET_ROUTES)
def test_org_mode_refuses_the_secret_routes_to_a_forwarded_loopback_caller(
    org_deployment, method, path, body, desktop_status
):
    """A signed-in member makes the request look like it came from 127.0.0.1.
    The socket is loopback here on purpose, so ``require_local`` passes and
    the detail proves the org-mode refusal itself turned the member away."""
    resp = _forwarding_client(socket_address="localhost").request(
        method, path, json=body, headers={**MEMBER_HEADERS, **FORGED_LOOPBACK_HEADERS}
    )

    assert (resp.status_code, resp.json()["detail"]) == (403, "not available in org deployments")


@pytest.mark.parametrize("tenancy", ["local", "org"])
@pytest.mark.parametrize("method,path,body,desktop_status", SECRET_ROUTES)
def test_a_forwarded_loopback_address_on_a_pod_socket_is_not_local(
    monkeypatch, tenancy, method, path, body, desktop_status
):
    """What an ingress hands the pod: the request lands on the pod's own
    address whatever X-Forwarded-For claims, with the Host the ingress copied
    from X-Forwarded-Host. ``require_local`` reads that socket, so the forged
    address is refused before tenancy is asked, in either mode."""
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_TENANCY_MODE", tenancy)
    get_app_settings.cache_clear()

    resp = _forwarding_client(socket_address="10.105.98.84:9010").request(
        method,
        path,
        json=body,
        headers={**MEMBER_HEADERS, **FORGED_LOOPBACK_HEADERS, "Host": "localhost"},
    )

    assert (resp.status_code, resp.json()["detail"]) == (403, "local requests only")


@pytest.mark.parametrize("method,path,body,desktop_status", SECRET_ROUTES)
def test_the_desktop_still_reaches_the_secret_routes(method, path, body, desktop_status):
    """The other direction: the desktop sidecar runs local tenancy on
    127.0.0.1, and every route still answers it from its own handler."""
    client = TestClient(create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321))

    resp = client.request(method, path, json=body)

    assert resp.status_code == desktop_status, resp.text
    assert resp.json().get("detail") not in GUARD_DETAILS


def test_the_desktop_renderer_still_reveals_a_key():
    """The renderer calls reveal-key with its own Origin, which the default
    allowed origins list."""
    client = TestClient(create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321))

    resp = client.get(
        "/api/v1/settings/reveal-key/anthropic", headers={"Origin": "http://localhost:5173"}
    )

    assert resp.status_code == 200, resp.text
    assert "value" in resp.json()
