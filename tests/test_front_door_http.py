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


def test_options_on_responses_is_open_in_org_mode(org_client):
    """A route declared OpenByDesign has to actually be open.

    The responses router used to declare AuthenticatedInOrgMode at router
    level, and FastAPI ADDS a route-level dependency to its router's rather
    than substituting for it — so the OPTIONS route carried both and answered
    401, under a comment saying it "carries its own OpenByDesign instead".
    TrustedHeaderMiddleware returns before building a Principal on any OPTIONS
    request, so there was never one to satisfy the stricter check.
    """
    resp = org_client.options("/api/v1/responses/")

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
