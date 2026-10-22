"""Behaviour of the trusted-header principal layer, in org mode and with
personal partitions (local tenancy with COWORK_TRUST_IDENTITY_HEADERS on).

Exercises TrustedHeaderMiddleware directly on a tiny app rather than
create_app() so it doesn't depend on the settings singleton (same approach as
test_auth_middleware.py). The CORS-ordering test guards the registration
order: a browser must receive the 401, not an opaque CORS failure.
"""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from cowork.common.settings.app_settings import get_app_settings
from cowork.principal import (
    Principal,
    TrustedHeaderMiddleware,
    get_principal,
    identity_trace_metadata,
    personal_partition_id,
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


def _app(org_mode: bool = True, enforce: bool = True, personal_partitions: bool = False) -> FastAPI:
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

    # The web UI's page shell, which a browser loads with no identity.
    @app.get("/")
    def page_shell(principal: Principal | None = Depends(get_principal)):
        return {"principal": None if principal is None else principal.user_id}

    @app.get("/assets/{name}")
    def page_asset(name: str):
        return {"asset": name}

    # A route with its own explicit OPTIONS handler (like responses.py's
    # OpenByDesign preflight body) — CORSMiddleware only intercepts a
    # genuine preflight (Origin + Access-Control-Request-Method) before this
    # ever runs, so a bare OPTIONS with neither reaches TrustedHeaderMiddleware
    # exactly like any other method.
    @app.options("/api/v1/explicit-options")
    def explicit_options():
        return {"ok": True}

    # Mirror create_app's ordering: principal added first (inner), CORS last
    # (outer) so a 401 still flows back out through CORS.
    if org_mode:
        app.add_middleware(
            TrustedHeaderMiddleware,
            exempt_paths={WEBHOOK},
            enforce=enforce,
            personal_partitions=personal_partitions,
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[ORIGIN],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def _client(org_mode: bool = True, enforce: bool = True) -> TestClient:
    return TestClient(_app(org_mode, enforce))


def _personal_client(enforce: bool = True) -> TestClient:
    return TestClient(_app(enforce=enforce, personal_partitions=True))


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


def test_bare_options_without_a_preflight_header_still_needs_identity():
    # Not a real preflight (no Access-Control-Request-Method), so
    # CORSMiddleware never intercepts it — it must reach identity
    # enforcement like any other request, not bypass it just for being OPTIONS.
    res = _client().options("/api/v1/explicit-options")
    assert res.status_code == 401


def test_options_with_request_method_but_no_origin_still_needs_identity():
    # Access-Control-Request-Method is caller-controlled — a browser never
    # sends it without Origin, but nothing stops a client from doing so to
    # try to reopen the bypass this closes. Both are required together.
    res = _client().options(
        "/api/v1/explicit-options",
        headers={"Access-Control-Request-Method": "GET"},
    )
    assert res.status_code == 401


def test_bare_options_with_identity_reaches_the_route():
    res = _client().options("/api/v1/explicit-options", headers=IDENTITY)
    assert res.status_code == 200


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
    res = _client().get(
        "/api/v1/whoami", headers=_browser_headers()
    )

    assert res.status_code == 200
    assert res.json()["org_id"] == ORG_ID


def test_browser_jwt_expected_organization_is_uuid_normalized():
    res = _client().get(
        "/api/v1/whoami", headers=_browser_headers(ORG_ID.upper())
    )

    assert res.status_code == 200
    assert res.json()["org_id"] == ORG_ID


def _assert_organization_boundary_rejection(response, status_code: int) -> None:
    assert response.status_code == status_code
    assert response.json() == {
        "code": "organization_reload_required",
        "detail": "Reload Cowork to continue in the active organization.",
    }
    assert response.headers.get(RELOAD_HEADER) == "required"
    assert response.headers.get("cache-control") == "no-store"
    assert response.headers.get("access-control-allow-origin") == ORIGIN


def test_organization_boundary_requires_browser_header():
    res = _client().get(
        "/api/v1/whoami",
        headers={**_browser_headers(None), "Origin": ORIGIN},
    )

    _assert_organization_boundary_rejection(res, 426)


def test_no_environment_variable_reopens_the_organization_boundary(monkeypatch):
    """The old mode key cannot bring the pass-through branch back.

    TrustedHeaderMiddleware reads no setting for the fence, so an environment
    still carrying the retired key gets the same refusal. The cache clears are
    aimed at one re-wiring in particular, a middleware that reads
    ``get_app_settings()`` itself: without them the settings load before the
    monkeypatch and this passes against that broken build. They do nothing for
    the shipped code, which never reads settings on this path. Restoring the
    original shape instead, a constructor keyword defaulting to ``"enforce"``,
    is caught by
    ``test_app_settings.py::test_stale_organization_boundary_mode_env_var_is_inert``,
    because that shape needs the AppSettings field back.
    """
    monkeypatch.setenv("COWORK_ORGANIZATION_BOUNDARY_MODE", "audit")
    get_app_settings.cache_clear()

    res = _client().get(
        "/api/v1/whoami",
        headers={**_browser_headers(None), "Origin": ORIGIN},
    )

    _assert_organization_boundary_rejection(res, 426)
    get_app_settings.cache_clear()


def test_organization_boundary_rejects_malformed_browser_header():
    res = _client().get(
        "/api/v1/whoami",
        headers={**_browser_headers("not-an-organization"), "Origin": ORIGIN},
    )

    _assert_organization_boundary_rejection(res, 409)


def test_organization_boundary_rejects_browser_org_mismatch():
    res = _client().get(
        "/api/v1/whoami",
        headers={**_browser_headers(OTHER_ORG_ID), "Origin": ORIGIN},
    )

    _assert_organization_boundary_rejection(res, 409)


def test_api_key_is_exempt_from_organization_boundary():
    res = _client().get(
        "/api/v1/whoami",
        headers={
            **IDENTITY,
            "Authorization": "Bearer mdb_organization_api_key",
            EXPECTED_ORG_HEADER: OTHER_ORG_ID,
        },
    )

    assert res.status_code == 200


def test_opaque_bearer_is_exempt_from_organization_boundary():
    res = _client().get(
        "/api/v1/whoami",
        headers={
            **IDENTITY,
            "Authorization": "Bearer opaque-token",
            EXPECTED_ORG_HEADER: OTHER_ORG_ID,
        },
    )

    assert res.status_code == 200


def test_short_three_part_bearer_is_exempt_from_organization_boundary():
    res = _client().get(
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

    res = _client().get(
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

    res = _client().get(
        "/api/v1/whoami",
        headers={
            **IDENTITY,
            "Authorization": f"Bearer {not_a_header}",
            EXPECTED_ORG_HEADER: OTHER_ORG_ID,
        },
    )

    assert res.status_code == 200


def test_missing_bearer_is_exempt_from_organization_boundary():
    res = _client().get(
        "/api/v1/whoami",
        headers={**IDENTITY, EXPECTED_ORG_HEADER: OTHER_ORG_ID},
    )

    assert res.status_code == 200


def test_health_is_exempt_from_organization_boundary():
    res = _client().get(
        "/api/v1/health", headers=_browser_headers(OTHER_ORG_ID)
    )

    assert res.status_code == 200


def test_channel_webhook_is_exempt_from_organization_boundary():
    res = _client().post(
        WEBHOOK, headers=_browser_headers(OTHER_ORG_ID)
    )

    assert res.status_code == 200


def test_options_is_exempt_from_organization_boundary():
    res = _client().options(
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


# ── personal partitions (local tenancy, trusted identity headers) ────────────

# The shared vector: cowork-self-hosted's tests pin the same three values.
GOLDEN_ORGANIZATION = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
GOLDEN_USER = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
GOLDEN_PARTITION = "44e92829-e701-599b-88f9-69f5bf1165eb"
OTHER_USER_ID = "22222222-2222-4222-8222-222222222222"


def test_personal_partition_id_golden_vector():
    assert personal_partition_id(GOLDEN_ORGANIZATION, GOLDEN_USER) == GOLDEN_PARTITION


def test_personal_partition_id_ignores_letter_case():
    assert personal_partition_id(GOLDEN_ORGANIZATION.upper(), GOLDEN_USER.upper()) == GOLDEN_PARTITION
    assert personal_partition_id(GOLDEN_ORGANIZATION, GOLDEN_USER.upper()) == GOLDEN_PARTITION


def test_personal_partition_id_refuses_a_non_uuid():
    with pytest.raises(ValueError):
        personal_partition_id(GOLDEN_ORGANIZATION, "not-a-uuid")


def test_personal_pair_builds_the_partition_principal():
    res = _personal_client().get(
        "/api/v1/whoami",
        headers={"X-User-Id": GOLDEN_USER, "X-Organization-Id": GOLDEN_ORGANIZATION},
    )
    assert res.status_code == 200
    assert res.json() == {"user_id": GOLDEN_USER, "org_id": GOLDEN_PARTITION, "email": "", "roles": []}


def test_personal_partition_survives_header_letter_case():
    res = _personal_client().get(
        "/api/v1/whoami",
        headers={"X-User-Id": GOLDEN_USER.upper(), "X-Organization-Id": GOLDEN_ORGANIZATION.upper()},
    )
    assert res.status_code == 200
    assert res.json()["user_id"] == GOLDEN_USER
    assert res.json()["org_id"] == GOLDEN_PARTITION


def test_two_people_and_two_deployments_get_four_partitions():
    client = _personal_client()
    partitions = {
        client.get("/api/v1/whoami", headers={"X-User-Id": user, "X-Organization-Id": org}).json()["org_id"]
        for user in (USER_ID, OTHER_USER_ID)
        for org in (ORG_ID, OTHER_ORG_ID)
    }
    assert len(partitions) == 4
    assert not partitions & {ORG_ID, OTHER_ORG_ID}  # never the header's organization


def test_personal_partitions_skip_the_organization_boundary():
    """The browser sends its token's organization claim as the expected
    organization, which never equals a partition id; the boundary would 409
    every request and the web UI would reload in a loop."""
    client = _personal_client()
    for expected in (OTHER_ORG_ID, ORG_ID, "not-a-uuid"):
        res = client.get("/api/v1/whoami", headers={**_browser_headers(expected), "Origin": ORIGIN})
        assert res.status_code == 200, expected
        assert res.json()["org_id"] == personal_partition_id(ORG_ID, USER_ID)
    missing = client.get("/api/v1/whoami", headers={**_browser_headers(None), "Origin": ORIGIN})
    assert missing.status_code == 200
    assert RELOAD_HEADER.lower() not in {k.lower() for k in missing.headers}


def test_org_mode_keeps_the_organization_boundary_and_the_raw_org():
    # Hosted regression for the flag's two effects.
    assert _client().get("/api/v1/whoami", headers=IDENTITY).json()["org_id"] == ORG_ID
    res = _client().get("/api/v1/whoami", headers={**_browser_headers(OTHER_ORG_ID), "Origin": ORIGIN})
    _assert_organization_boundary_rejection(res, 409)


def test_personal_partitions_enforce_answers_401_without_a_pair():
    client = _personal_client()
    assert client.get("/api/v1/whoami").status_code == 401
    assert client.get("/api/v1/whoami", headers={"X-User-Id": USER_ID}).status_code == 401
    assert client.get(
        "/api/v1/whoami", headers={"X-User-Id": "sub-not-a-uuid", "X-Organization-Id": ORG_ID}
    ).status_code == 401


def test_personal_partitions_audit_serves_no_person():
    res = _personal_client(enforce=False).get("/api/v1/whoami")
    assert res.status_code == 200
    assert res.json() == {"principal": None}


@pytest.mark.parametrize(
    "headers",
    [
        {"X-User-Id": USER_ID},
        {"X-Organization-Id": ORG_ID},
        {"X-User-Id": "f:ldap:alice", "X-Organization-Id": ORG_ID},
        {"X-User-Id": USER_ID, "X-Organization-Id": "not-a-uuid"},
        {"X-User-Id": "  ", "X-Organization-Id": ORG_ID},
    ],
)
def test_personal_partitions_audit_refuses_a_broken_pair(headers, caplog):
    """Audit serves a request that names nobody as the operator view. A pair
    the proxy sent but broke is no such request: it answers 401."""
    with caplog.at_level("WARNING", logger="cowork.principal"):
        res = _personal_client(enforce=False).get("/api/v1/whoami", headers=headers)
    assert res.status_code == 401
    assert "no principal" not in caplog.text


def test_hosted_audit_still_lets_a_broken_pair_through():
    res = _client(enforce=False).get("/api/v1/whoami", headers={"X-User-Id": USER_ID})
    assert res.status_code == 200
    assert res.json() == {"principal": None}


@pytest.mark.parametrize("enforce", [True, False])
def test_personal_partitions_leave_the_page_shell_alone(enforce, caplog):
    """The page shell names no person and logs no missing one, under either
    mode; the API beside it behaves as before."""
    client = _personal_client(enforce=enforce)
    with caplog.at_level("WARNING", logger="cowork.principal"):
        for path in ("/", "/assets/app.js"):
            assert client.get(path).status_code == 200, path
        shell = client.get("/", headers={"X-User-Id": USER_ID, "X-Organization-Id": ORG_ID})
    assert shell.json() == {"principal": None}
    assert "no principal" not in caplog.text

    with caplog.at_level("WARNING", logger="cowork.principal"):
        api = client.get("/api/v1/whoami")
    if enforce:
        assert api.status_code == 401
    else:
        assert api.json() == {"principal": None}
        assert "no principal on GET /api/v1/whoami" in caplog.text


def test_hosted_still_reads_identity_on_every_path():
    assert _client().get("/").status_code == 401


def test_underscore_spelled_identity_headers_name_no_person():
    """A client that slips the pair past a proxy in underscore spelling names no
    one: the middleware reads only the hyphenated names."""
    underscored = {"X_User_Id": USER_ID, "X_Organization_Id": ORG_ID}
    assert _personal_client().get("/api/v1/whoami", headers=underscored).status_code == 401
    res = _personal_client(enforce=False).get("/api/v1/whoami", headers=underscored)
    assert res.json() == {"principal": None}
