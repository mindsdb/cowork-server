"""POST/DELETE /api/v1/push/devices: the phones the mobile app registers.

Org mode only. A token names one phone, so registering it again (by anyone,
in any organization) moves it to the caller; deleting only ever removes the
caller's own row.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, delete, select

from cowork.api.v1.endpoints import push as push_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.models.push_device import PushDevice
from cowork.principal import Principal, get_principal
from cowork.services.push import devices

ORG_A = "11111111-1111-1111-1111-111111111111"
ORG_B = "22222222-2222-2222-2222-222222222222"
ALICE = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
BOB = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def _engine():
    return get_engine(get_app_settings().database.uri)


@pytest.fixture(autouse=True)
def _clean_devices():
    yield
    with Session(_engine()) as session:
        session.exec(delete(PushDevice))
        session.commit()


@pytest.fixture
def org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


class _Caller:
    principal: Principal | None = Principal(user_id=ALICE, org_id=ORG_A)


@pytest.fixture
def client():
    caller = _Caller()
    app = FastAPI()
    app.include_router(push_ep.router, prefix="/api/v1/push")
    app.dependency_overrides[get_principal] = lambda: caller.principal
    with TestClient(app) as test_client:
        test_client.caller = caller  # type: ignore[attr-defined]
        yield test_client


def _rows() -> list[PushDevice]:
    with Session(_engine()) as session:
        return list(session.exec(select(PushDevice)).all())


def _as(client, user_id: str, org_id: str) -> None:
    client.caller.principal = Principal(user_id=user_id, org_id=org_id)


def test_register_records_the_callers_device(org_mode, client):
    response = client.post("/api/v1/push/devices", json={
        "token": "tok-1", "platform": "ios", "app_version": "1.0.0", "environment": "sandbox",
    })
    assert response.status_code == 204
    (row,) = _rows()
    assert (row.org_id, row.user_id, row.platform, row.token) == (ORG_A, ALICE, "ios", "tok-1")
    assert (row.app_version, row.environment) == ("1.0.0", "sandbox")
    assert row.last_seen_at is not None


def test_register_accepts_camel_case_and_null_fields(org_mode, client):
    response = client.post("/api/v1/push/devices", json={
        "token": "tok-1", "platform": "android", "appVersion": None, "environment": None,
    })
    assert response.status_code == 204
    (row,) = _rows()
    assert row.platform == "android" and row.environment is None


def test_registering_again_upserts_one_row(org_mode, client):
    client.post("/api/v1/push/devices", json={"token": "tok-1", "platform": "ios", "app_version": "1.0"})
    client.post("/api/v1/push/devices", json={"token": "tok-1", "platform": "ios", "app_version": "1.1"})
    (row,) = _rows()
    assert row.app_version == "1.1"


def test_a_token_moves_to_whoever_registers_it_last(org_mode, client):
    client.post("/api/v1/push/devices", json={"token": "tok-1", "platform": "ios"})
    _as(client, BOB, ORG_B)
    assert client.post("/api/v1/push/devices", json={"token": "tok-1", "platform": "ios"}).status_code == 204
    (row,) = _rows()
    assert (row.org_id, row.user_id) == (ORG_B, BOB)


def test_delete_removes_only_the_callers_own_token(org_mode, client):
    client.post("/api/v1/push/devices", json={"token": "tok-1", "platform": "ios"})
    _as(client, BOB, ORG_A)
    assert client.request("DELETE", "/api/v1/push/devices", json={"token": "tok-1"}).status_code == 204
    assert len(_rows()) == 1, "someone else's token stays"
    _as(client, ALICE, ORG_B)
    client.request("DELETE", "/api/v1/push/devices", json={"token": "tok-1"})
    assert len(_rows()) == 1, "the same person in another org can't delete it either"
    _as(client, ALICE, ORG_A)
    assert client.request("DELETE", "/api/v1/push/devices", json={"token": "tok-1"}).status_code == 204
    assert _rows() == []
    # Idempotent.
    assert client.request("DELETE", "/api/v1/push/devices", json={"token": "tok-1"}).status_code == 204


def test_bad_bodies_are_refused(org_mode, client):
    assert client.post("/api/v1/push/devices", json={"token": "t", "platform": "windows"}).status_code == 422
    assert client.post("/api/v1/push/devices", json={"token": "", "platform": "ios"}).status_code == 422
    assert client.post(
        "/api/v1/push/devices", json={"token": "t", "platform": "ios", "environment": "dev"},
    ).status_code == 422


def test_no_principal_in_org_mode_is_unauthorized(org_mode, client):
    client.caller.principal = None
    assert client.post("/api/v1/push/devices", json={"token": "t", "platform": "ios"}).status_code == 401
    assert _rows() == []


def test_outside_org_mode_both_routes_are_not_found(monkeypatch, client):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    try:
        assert client.post("/api/v1/push/devices", json={"token": "t", "platform": "ios"}).status_code == 404
        assert client.request("DELETE", "/api/v1/push/devices", json={"token": "t"}).status_code == 404
    finally:
        get_app_settings.cache_clear()
    assert _rows() == []


def test_devices_for_is_bounded_by_org_and_person():
    with Session(_engine()) as raw:
        session = ScopedSession(raw, SYSTEM_SCOPE)
        devices.register_device(
            session, org_id=ORG_A, user_id=ALICE, token="a1", platform="ios", app_version=None, environment=None,
        )
        devices.register_device(
            session, org_id=ORG_B, user_id=ALICE, token="b1", platform="android", app_version=None, environment=None,
        )
        devices.register_device(
            session, org_id=ORG_A, user_id=BOB, token="a2", platform="ios", app_version=None, environment=None,
        )
        session.commit()
        assert [d.token for d in devices.devices_for(session, org_id=ORG_A, user_id=ALICE)] == ["a1"]
        assert [d.token for d in devices.devices_for(session, org_id=ORG_B, user_id=ALICE)] == ["b1"]
        devices.set_environment(session, token="a1", environment="sandbox")
        devices.forget_token(session, token="a2")
        session.commit()
        assert devices.devices_for(session, org_id=ORG_A, user_id=ALICE)[0].environment == "sandbox"
        assert devices.devices_for(session, org_id=ORG_A, user_id=BOB) == []


def test_the_push_routes_declare_a_permission():
    from cowork.api.v1.route_walker import undeclared_routes

    app = FastAPI()
    app.include_router(push_ep.router, prefix="/api/v1/push")
    assert undeclared_routes(app) == []
