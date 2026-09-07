"""Behaviour of the org-mode trusted-header principal layer.

Exercises TrustedHeaderMiddleware directly on a tiny app rather than
create_app() so it doesn't depend on the settings singleton (same approach as
test_auth_middleware.py). The CORS-ordering test guards the registration
order: a browser must receive the 401, not an opaque CORS failure.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from cowork.principal import (
    Principal,
    TrustedHeaderMiddleware,
    get_principal,
    identity_trace_metadata,
)

ORIGIN = "http://localhost:1234"
WEBHOOK = "/api/v1/channels/slack/events"

USER_ID = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"
ORG_ID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
OTHER_ORG_ID = "4e27728a-a002-48b5-8961-0e1ca339d13f"
IDENTITY = {"X-User-Id": USER_ID, "X-Organization-Id": ORG_ID}
EXPECTED_ORG_HEADER = "X-Cowork-Expected-Organization-Id"
RELOAD_HEADER = "X-Cowork-Organization-Reload"
JWT = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyIn0.signature"


def _app(
    org_mode: bool = True,
    enforce: bool = True,
    organization_boundary_mode: str | None = "enforce",
) -> FastAPI:
    app = FastAPI()

    @app.get("/api/v1/health")
    def health():
        return {"ok": True}

    @app.get("/api/v1/whoami")
    def whoami(principal: Principal | None = Depends(get_principal)):
        if principal is None:
            return {"principal": None}
        return {
            "user_id": principal.user_id,
            "org_id": principal.org_id,
            "email": principal.email,
            "roles": sorted(principal.roles),
        }

    @app.post(WEBHOOK)
    async def slack_webhook():
        return {"ack": True}

    # Mirror create_app's ordering: principal added first (inner), CORS last
    # (outer) so a 401 still flows back out through CORS.
    if org_mode:
        options = {}
        if organization_boundary_mode is not None:
            options["organization_boundary_mode"] = organization_boundary_mode
        app.add_middleware(
            TrustedHeaderMiddleware,
            exempt_paths={WEBHOOK},
            enforce=enforce,
            **options,
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[ORIGIN],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def _client(
    org_mode: bool = True,
    enforce: bool = True,
    organization_boundary_mode: str | None = "enforce",
) -> TestClient:
    return TestClient(_app(org_mode, enforce, organization_boundary_mode))


def _browser_headers(expected_org_id: str | None = ORG_ID) -> dict[str, str]:
    headers = {**IDENTITY, "Authorization": f"Bearer {JWT}"}
    if expected_org_id is not None:
        headers[EXPECTED_ORG_HEADER] = expected_org_id
    return headers


def test_missing_identity_is_401():
    assert _client().get("/api/v1/whoami").status_code == 401


def test_user_id_alone_is_401():
    res = _client().get("/api/v1/whoami", headers={"X-User-Id": USER_ID})
    assert res.status_code == 401


def test_org_id_alone_is_401():
    res = _client().get("/api/v1/whoami", headers={"X-Organization-Id": ORG_ID})
    assert res.status_code == 401


def test_blank_header_values_are_401():
    res = _client().get(
        "/api/v1/whoami",
        headers={"X-User-Id": "  ", "X-Organization-Id": ORG_ID},
    )
    assert res.status_code == 401


def test_non_uuid_identity_is_401():
    res = _client().get(
        "/api/v1/whoami",
        headers={"X-User-Id": "user-123", "X-Organization-Id": ORG_ID},
    )
    assert res.status_code == 401


def test_identity_pair_builds_principal():
    res = _client().get("/api/v1/whoami", headers=IDENTITY)
    assert res.status_code == 200
    assert res.json() == {
        "user_id": USER_ID,
        "org_id": ORG_ID,
        "email": "",
        "roles": [],
    }


def test_uuid_case_is_normalized():
    res = _client().get(
        "/api/v1/whoami",
        headers={"X-User-Id": USER_ID.upper(), "X-Organization-Id": ORG_ID.upper()},
    )
    assert res.status_code == 200
    assert res.json()["user_id"] == USER_ID
    assert res.json()["org_id"] == ORG_ID


def test_email_and_roles_are_parsed():
    res = _client().get(
        "/api/v1/whoami",
        headers={
            **IDENTITY,
            "X-User-Email": "zoran@mindsdb.com",
            "X-User-Roles": "admin, member,,  owner ",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["email"] == "zoran@mindsdb.com"
    assert body["roles"] == ["admin", "member", "owner"]


def test_health_is_exempt_without_identity():
    assert _client().get("/api/v1/health").status_code == 200


def test_channel_webhook_is_exempt_without_identity():
    assert _client().post(WEBHOOK).status_code == 200


def test_options_preflight_is_allowed_without_identity():
    res = _client().options(
        "/api/v1/whoami",
        headers={"Origin": ORIGIN, "Access-Control-Request-Method": "GET"},
    )
    assert res.status_code in (200, 204)
    assert res.headers.get("access-control-allow-origin") == ORIGIN


def test_401_carries_cors_headers():
    # CORS is the outer layer, so even the identity 401 must include
    # Access-Control-Allow-Origin — otherwise the browser reports a CORS error
    # and the caller never sees the 401.
    res = _client().get("/api/v1/whoami", headers={"Origin": ORIGIN})
    assert res.status_code == 401
    assert res.headers.get("access-control-allow-origin") == ORIGIN


def test_local_mode_has_no_principal_and_no_gate():
    # Local mode: no middleware, get_principal resolves to None.
    res = _client(org_mode=False).get("/api/v1/whoami")
    assert res.status_code == 200
    assert res.json() == {"principal": None}


def test_audit_mode_lets_missing_identity_through(caplog):
    with caplog.at_level("WARNING", logger="cowork.principal"):
        res = _client(enforce=False).get("/api/v1/whoami")
    assert res.status_code == 200
    assert res.json() == {"principal": None}
    assert "no principal on GET /api/v1/whoami" in caplog.text


def test_audit_mode_lets_malformed_identity_through(caplog):
    with caplog.at_level("WARNING", logger="cowork.principal"):
        res = _client(enforce=False).get(
            "/api/v1/whoami",
            headers={"X-User-Id": "user-123", "X-Organization-Id": ORG_ID},
        )
    assert res.status_code == 200
    assert res.json() == {"principal": None}
    assert "no principal on" in caplog.text


def test_org_mode_enforces_identity_when_the_env_var_is_absent(monkeypatch):
    """The whole point of the default. Every deployment sets
    COWORK_IDENTITY_ENFORCE today, so the value only decides anything when
    somebody drops it, and dropping it must not reopen the no-principal path.
    """
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.server import create_app

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.delenv("COWORK_IDENTITY_ENFORCE", raising=False)
    get_app_settings.cache_clear()
    try:
        app = create_app()

        # A route that answers 200 on its own, so reaching it is the signal.
        # Every real route touches org-scoped data and answers 401 under audit
        # too, which would make this test pass whatever the default is.
        @app.get("/api/v1/_reached")
        def _reached():
            return {"reached": True}

        client = TestClient(app)

        res = client.get("/api/v1/_reached")
        assert res.status_code == 401
        assert res.json() == {"detail": "Unauthorized"}

        res = client.get("/api/v1/health/")
        assert res.status_code == 200, "health stays reachable for the kubelet"
    finally:
        get_app_settings.cache_clear()


def test_missing_tenant_scope_maps_to_401(monkeypatch):
    # Org-scoped data touched with no org in scope (audit mode, no identity)
    # must answer 401 via the create_app exception handler, not a 500.
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.db.scoped import MissingTenantScopeError
    from cowork.server import create_app

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    get_app_settings.cache_clear()
    try:
        app = create_app()

        @app.get("/api/v1/_boom")
        def boom():
            raise MissingTenantScopeError("no org in scope")

        res = TestClient(app).get("/api/v1/_boom")
        assert res.status_code == 401
        assert res.json() == {"detail": "Unauthorized"}
    finally:
        get_app_settings.cache_clear()


def test_audit_mode_still_builds_principal_when_identity_present():
    res = _client(enforce=False).get("/api/v1/whoami", headers=IDENTITY)
    assert res.status_code == 200
    assert res.json()["user_id"] == USER_ID
    assert res.json()["org_id"] == ORG_ID


def test_browser_jwt_expected_organization_exact_match_is_allowed():
    res = _client(organization_boundary_mode="enforce").get(
        "/api/v1/whoami", headers=_browser_headers()
    )

    assert res.status_code == 200
    assert res.json()["org_id"] == ORG_ID


def test_browser_jwt_expected_organization_is_uuid_normalized():
    res = _client(organization_boundary_mode="enforce").get(
        "/api/v1/whoami", headers=_browser_headers(ORG_ID.upper())
    )

    assert res.status_code == 200
    assert res.json()["org_id"] == ORG_ID


def test_organization_boundary_audits_missing_header_and_allows(caplog):
    with caplog.at_level("WARNING", logger="cowork.principal"):
        res = _client(organization_boundary_mode="audit").get(
            "/api/v1/whoami", headers=_browser_headers(None)
        )

    assert res.status_code == 200
    assert any(
        "organization" in record.getMessage().lower()
        and "missing" in record.getMessage().lower()
        and "/api/v1/whoami" in record.getMessage()
        for record in caplog.records
    )


def test_organization_boundary_audits_malformed_header_and_allows(caplog):
    with caplog.at_level("WARNING", logger="cowork.principal"):
        res = _client(organization_boundary_mode="audit").get(
            "/api/v1/whoami", headers=_browser_headers("not-an-organization")
        )

    assert res.status_code == 200
    assert any(
        "organization" in record.getMessage().lower()
        and "malformed" in record.getMessage().lower()
        and "/api/v1/whoami" in record.getMessage()
        for record in caplog.records
    )


def test_organization_boundary_audits_mismatch_and_allows(caplog):
    with caplog.at_level("WARNING", logger="cowork.principal"):
        res = _client(organization_boundary_mode="audit").get(
            "/api/v1/whoami", headers=_browser_headers(OTHER_ORG_ID)
        )

    assert res.status_code == 200
    assert any(
        "organization" in record.getMessage().lower()
        and "mismatch" in record.getMessage().lower()
        and "/api/v1/whoami" in record.getMessage()
        for record in caplog.records
    )


def _assert_organization_boundary_rejection(response, status_code: int) -> None:
    assert response.status_code == status_code
    assert response.json() == {
        "code": "organization_reload_required",
        "detail": "Reload Cowork to continue in the active organization.",
    }
    assert response.headers.get(RELOAD_HEADER) == "required"
    assert response.headers.get("cache-control") == "no-store"
    assert response.headers.get("access-control-allow-origin") == ORIGIN


def test_organization_boundary_enforce_requires_browser_header():
    res = _client(organization_boundary_mode="enforce").get(
        "/api/v1/whoami",
        headers={**_browser_headers(None), "Origin": ORIGIN},
    )

    _assert_organization_boundary_rejection(res, 426)


def test_organization_boundary_defaults_to_enforce():
    res = _client(organization_boundary_mode=None).get(
        "/api/v1/whoami",
        headers={**_browser_headers(None), "Origin": ORIGIN},
    )

    _assert_organization_boundary_rejection(res, 426)


def test_organization_boundary_enforce_rejects_malformed_browser_header():
    res = _client(organization_boundary_mode="enforce").get(
        "/api/v1/whoami",
        headers={**_browser_headers("not-an-organization"), "Origin": ORIGIN},
    )

    _assert_organization_boundary_rejection(res, 409)


def test_organization_boundary_enforce_rejects_browser_org_mismatch():
    res = _client(organization_boundary_mode="enforce").get(
        "/api/v1/whoami",
        headers={**_browser_headers(OTHER_ORG_ID), "Origin": ORIGIN},
    )

    _assert_organization_boundary_rejection(res, 409)


def test_api_key_is_exempt_from_organization_boundary():
    res = _client(organization_boundary_mode="enforce").get(
        "/api/v1/whoami",
        headers={
            **IDENTITY,
            "Authorization": "Bearer mdb_organization_api_key",
            EXPECTED_ORG_HEADER: OTHER_ORG_ID,
        },
    )

    assert res.status_code == 200


def test_opaque_bearer_is_exempt_from_organization_boundary():
    res = _client(organization_boundary_mode="enforce").get(
        "/api/v1/whoami",
        headers={
            **IDENTITY,
            "Authorization": "Bearer opaque-token",
            EXPECTED_ORG_HEADER: OTHER_ORG_ID,
        },
    )

    assert res.status_code == 200


def test_short_three_part_bearer_is_exempt_from_organization_boundary():
    res = _client(organization_boundary_mode="enforce").get(
        "/api/v1/whoami",
        headers={
            **IDENTITY,
            "Authorization": "Bearer opaque.service.token",
            EXPECTED_ORG_HEADER: OTHER_ORG_ID,
        },
    )

    assert res.status_code == 200


def test_compact_jose_header_still_takes_part_in_the_organization_boundary():
    """`{"alg":"RS256"}` encodes to exactly 20 characters — a real signed token
    that a length threshold would wave past the boundary."""
    compact = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature"

    res = _client(organization_boundary_mode="enforce").get(
        "/api/v1/whoami",
        headers={
            **IDENTITY,
            "Authorization": f"Bearer {compact}",
            EXPECTED_ORG_HEADER: OTHER_ORG_ID,
            "Origin": ORIGIN,
        },
    )

    _assert_organization_boundary_rejection(res, 409)


def test_three_part_bearer_without_a_jose_header_is_exempt():
    """The first segment decodes, but to a JSON string rather than a header."""
    not_a_header = "IlJTMjU2Ig.eyJzdWIiOiJ1c2VyIn0.signature"

    res = _client(organization_boundary_mode="enforce").get(
        "/api/v1/whoami",
        headers={
            **IDENTITY,
            "Authorization": f"Bearer {not_a_header}",
            EXPECTED_ORG_HEADER: OTHER_ORG_ID,
        },
    )

    assert res.status_code == 200


def test_missing_bearer_is_exempt_from_organization_boundary():
    res = _client(organization_boundary_mode="enforce").get(
        "/api/v1/whoami",
        headers={**IDENTITY, EXPECTED_ORG_HEADER: OTHER_ORG_ID},
    )

    assert res.status_code == 200


def test_health_is_exempt_from_organization_boundary():
    res = _client(organization_boundary_mode="enforce").get(
        "/api/v1/health", headers=_browser_headers(OTHER_ORG_ID)
    )

    assert res.status_code == 200


def test_channel_webhook_is_exempt_from_organization_boundary():
    res = _client(organization_boundary_mode="enforce").post(
        WEBHOOK, headers=_browser_headers(OTHER_ORG_ID)
    )

    assert res.status_code == 200


def test_options_is_exempt_from_organization_boundary():
    res = _client(organization_boundary_mode="enforce").options(
        "/api/v1/whoami",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": EXPECTED_ORG_HEADER,
        },
    )

    assert res.status_code in (200, 204)
    assert res.headers.get("access-control-allow-origin") == ORIGIN


def test_identity_trace_metadata_without_principal_is_passthrough():
    assert identity_trace_metadata(None, None) is None
    assert identity_trace_metadata(None, {"turn": "3"}) == {"turn": "3"}


def test_identity_trace_metadata_adds_identity():
    principal = Principal(user_id=USER_ID, org_id=ORG_ID)
    merged = identity_trace_metadata(principal, {"harness": "anton"})
    assert merged == {
        "harness": "anton",
        "user_id": USER_ID,
        "organization_id": ORG_ID,
    }


def test_identity_trace_metadata_overrides_client_spoofing():
    principal = Principal(user_id=USER_ID, org_id=ORG_ID)
    merged = identity_trace_metadata(
        principal, {"user_id": "attacker", "organization_id": "other-org"}
    )
    assert merged["user_id"] == USER_ID
    assert merged["organization_id"] == ORG_ID


def test_identity_trace_metadata_does_not_mutate_base():
    principal = Principal(user_id=USER_ID, org_id=ORG_ID)
    base = {"harness": "anton"}
    identity_trace_metadata(principal, base)
    assert base == {"harness": "anton"}
