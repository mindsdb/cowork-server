"""Behaviour of the ENG-2094 permission front door's four classes.

Exercises ``require(...)`` on a tiny scratch app rather than ``create_app()``,
so this doesn't depend on the settings singleton or
``TrustedHeaderMiddleware`` — overriding ``get_principal`` stands in for the
identity the middleware would have built.

Every "allows" test overrides ``get_principal``, never
``require(<the class under test>)``. Overriding the latter replaces the check
with the answer and proves nothing: the whole suite still passed with
``Authenticated.check`` mutated to raise unconditionally.
"""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.permissions import (
    Authenticated,
    AuthenticatedInOrgMode,
    AuthenticatedOrgAdmin,
    OpenByDesign,
    require,
)
from cowork.principal import ORG_MANAGE_ROLE, Principal, get_principal

MEMBER = Principal(user_id="u1", org_id="o1")
ADMIN = Principal(user_id="u2", org_id="o1", roles=frozenset({ORG_MANAGE_ROLE}))


@pytest.fixture(autouse=True)
def _reset_app_settings():
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/open", dependencies=[Depends(require(OpenByDesign))])
    def open_route():
        return {"ok": True}

    @app.get("/authenticated")
    def authenticated_route(principal: Principal = Depends(require(Authenticated))):
        return {"user_id": principal.user_id}

    @app.get("/in-org-mode")
    def in_org_mode_route(
        principal: Principal | None = Depends(require(AuthenticatedInOrgMode)),
    ):
        return {"user_id": principal.user_id if principal else None}

    @app.get("/org-admin", dependencies=[Depends(require(AuthenticatedOrgAdmin))])
    def org_admin_route():
        return {"ok": True}

    return app


def _client(principal: Principal | None = None) -> TestClient:
    app = _app()
    if principal is not None:
        app.dependency_overrides[get_principal] = lambda: principal
    return TestClient(app)


# ── OpenByDesign ─────────────────────────────────────────────────────


def test_open_route_allows_no_principal():
    assert _client().get("/open").status_code == 200


def test_open_route_stays_open_in_org_mode(monkeypatch):
    # The one property OpenByDesign exists to assert. Without it, a route
    # declared open could be closed by something else and nothing would say so.
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    assert _client().get("/open").status_code == 200


# ── Authenticated ────────────────────────────────────────────────────


def test_authenticated_route_401s_with_no_principal():
    assert _client().get("/authenticated").status_code == 401


def test_authenticated_route_allows_a_verified_principal():
    resp = _client(MEMBER).get("/authenticated")
    assert resp.status_code == 200
    assert resp.json() == {"user_id": "u1"}


# ── AuthenticatedInOrgMode ───────────────────────────────────────────


def test_in_org_mode_401s_in_org_mode_with_no_principal(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    assert _client().get("/in-org-mode").status_code == 401


def test_in_org_mode_allows_a_verified_principal_in_org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    resp = _client(MEMBER).get("/in-org-mode")
    assert resp.status_code == 200
    assert resp.json() == {"user_id": "u1"}


def test_in_org_mode_is_a_no_op_in_local_mode():
    # tenancy_mode defaults to "local" — desktop never builds a Principal, so
    # enforcing here would 401 every local caller.
    resp = _client().get("/in-org-mode")
    assert resp.status_code == 200
    assert resp.json() == {"user_id": None}


# ── AuthenticatedOrgAdmin ────────────────────────────────────────────


def test_org_admin_403s_a_member_in_org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    resp = _client(MEMBER).get("/org-admin")
    assert resp.status_code == 403
    assert resp.json()["detail"] == "requires an org admin"


def test_org_admin_allows_an_admin_in_org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    assert _client(ADMIN).get("/org-admin").status_code == 200


def test_org_admin_401s_before_403ing_when_there_is_no_principal(monkeypatch):
    # Who you are is resolved before what you may do, so an anonymous caller
    # gets 401 rather than a 403 that implies a known identity.
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    assert _client().get("/org-admin").status_code == 401


def test_org_admin_is_a_no_op_in_local_mode():
    assert _client().get("/org-admin").status_code == 200


# ── require() itself ─────────────────────────────────────────────────


def test_require_stamps_permission_cls_for_the_ci_walker():
    assert require(OpenByDesign).permission_cls is OpenByDesign
    assert require(Authenticated).permission_cls is Authenticated
    assert require(AuthenticatedInOrgMode).permission_cls is AuthenticatedInOrgMode
    assert require(AuthenticatedOrgAdmin).permission_cls is AuthenticatedOrgAdmin


def test_require_is_cached_so_every_call_site_shares_one_dependency():
    assert require(OpenByDesign) is require(OpenByDesign)
    assert require(Authenticated) is require(Authenticated)


def test_require_names_the_dependency_after_its_permission():
    # scripts/dump_routes.py reads __qualname__ to build auth's
    # endpoint-authorization map; the raw closure name says nothing.
    assert require(AuthenticatedInOrgMode).__qualname__ == "require(AuthenticatedInOrgMode)"


def test_require_resolves_annotations_against_the_permissions_own_module():
    """A Permission defined elsewhere, under deferred annotations, still wires.

    ``require`` copies ``check``'s signature onto the dependency FastAPI
    inspects. Left as strings, those annotations get resolved against
    permissions.py's namespace instead of the subclass's own, and a name
    permissions.py does not import survives as a ``ForwardRef``: the route
    registers, a valid request answers 422, and ``/openapi.json`` raises.
    ``eval_str=True`` is what stops that, and this is what catches its removal.
    """
    from _foreign_permission import ForeignBodyPermission

    app = FastAPI()

    @app.post("/foreign")
    def foreign_route(who: str = Depends(require(ForeignBodyPermission))):
        return {"who": who}

    resp = TestClient(app).post("/foreign", json={"who": "x"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"who": "x"}
    # An unresolved ForwardRef only surfaces when the schema gets built.
    assert app.openapi()["paths"]["/foreign"]["post"]


@pytest.mark.parametrize(
    "permission_cls,guard_names",
    [
        ("LoopbackOnly", ["require_local"]),
        ("DesktopOnly", ["require_local_tenancy"]),
        ("LoopbackDesktopOnly", ["require_local", "require_local_tenancy"]),
    ],
)
def test_the_absorbed_guards_stay_reachable_through_dependency_overrides(
    permission_cls, guard_names
):
    """Absorbing a guard into a Permission must not hide it from FastAPI.

    ``LoopbackOnly`` and friends declare ``Depends(require_local)`` rather than
    calling it, for the same reason ``Authenticated`` declares
    ``get_principal``. Calling it directly works at runtime and silently breaks
    every test that stubs the guard out — which is how
    tests/test_coding_service.py started getting a 403 where it had overridden
    ``require_local`` away.
    """
    from cowork.api.v1 import permissions
    from cowork.api.v1.endpoints import guards

    cls = getattr(permissions, permission_cls)

    app = FastAPI()

    @app.get("/guarded", dependencies=[Depends(require(cls))])
    def guarded_route():
        return {"ok": True}

    client = TestClient(app)
    for name in guard_names:
        client.app.dependency_overrides[getattr(guards, name)] = lambda: None

    assert client.get("/guarded").status_code == 200


def test_a_loopback_permission_still_refuses_a_remote_caller():
    """The other half: overriding is a test affordance, not a hole."""
    from cowork.api.v1.permissions import LoopbackOnly

    app = FastAPI()

    @app.get("/guarded", dependencies=[Depends(require(LoopbackOnly))])
    def guarded_route():
        return {"ok": True}

    assert TestClient(app).get("/guarded").status_code == 403
