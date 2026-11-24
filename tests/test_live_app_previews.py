"""Preview capabilities stay bound to the viewer, owner and current launch."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta
from threading import Barrier, Event
from uuid import UUID

import pytest
from sqlalchemy import delete
from sqlalchemy.orm.exc import StaleDataError
from sqlmodel import Session, select

from cowork.common.datetime_utils import ensure_utc
from cowork.common.encryption import decrypt
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession
from cowork.db.session import get_engine
from cowork.models.live_app import LiveAppLaunchRecord
from cowork.models.live_app_preview import LiveAppPreview, LiveAppPreviewOwner
from cowork.services import live_app_preview as preview
from tests._live_apps import A, ALICE, B, BOB, VIEWER_A, VIEWER_B, viewer_cookie
from tests.test_live_apps_access import BOOT_1, BOOT_2, _client, _launch, _records
from tests import test_live_apps_access as access_tests

services = access_tests.services

BASE = "/api/v1/live-apps"
COWORK = "http://cowork.localhost:8080"


@pytest.fixture(autouse=True)
def preview_origins(monkeypatch):
    """The web UI's origin, as compose sets COWORK_ALLOWED_ORIGINS for the cowork service."""
    monkeypatch.setenv("COWORK_ALLOWED_ORIGINS", f'["{COWORK}"]')
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture(autouse=True)
def preview_rows(db_schema):
    """Preview rows have a separate lifetime from the launch fixture's rows."""
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as session:
        session.exec(delete(LiveAppPreview))
        session.exec(delete(LiveAppPreviewOwner))
        session.commit()
    yield
    with Session(engine) as session:
        session.exec(delete(LiveAppPreview))
        session.exec(delete(LiveAppPreviewOwner))
        session.commit()


@pytest.fixture
def launched(services):
    _launch(services, 1, ALICE, slug="dashboard", boot_id=BOOT_1)
    return _records(1)[-1].id


def _create(client, launch_id, *, owner=A, origin=COWORK):
    return client.post(f"{BASE}/previews", json={"launchId": str(launch_id)}, headers={**owner, "Origin": origin})


def _post(client, action, token, *, slot=1, cookie="", origin=None):
    return client.post(f"{BASE}/{slot}/preview/{action}", json={"token": token}, headers={
        "Origin": origin if origin is not None else preview.app_origin(slot), "Cookie": cookie,
    })


def _row(record_id):
    with Session(get_engine(get_app_settings().database.uri)) as session:
        return session.get(LiveAppPreview, UUID(str(record_id)))


@dataclass(frozen=True)
class BrowserLease:
    record_id: str
    intent: str
    grant: str
    cookie: str


def _active(client, launch_id, *, owner=A, viewer="a", slot=1):
    created = _create(client, launch_id, owner=owner)
    assert created.status_code == 200, created.text
    record = created.json()
    authorized = _post(client, "authorize", record["intent"], slot=slot, cookie=viewer_cookie(viewer))
    assert authorized.status_code == 200, authorized.text
    grant = authorized.json()["grant"]
    exchanged = _post(client, "exchange", grant, slot=slot)
    assert exchanged.status_code == 204, exchanged.text
    return BrowserLease(record["id"], record["intent"], grant, exchanged.headers["set-cookie"].split(";", 1)[0])


def _access(client, cookie, slot=1):
    return client.get(f"{BASE}/{slot}/access", headers={"Cookie": cookie})


def test_full_protocol_sets_only_a_partitioned_cookie_and_rechecks_access(services, launched):
    client = _client()
    lease = _active(client, launched)
    assert _post(client, "connect", lease.intent, cookie=lease.cookie).status_code == 204
    assert _access(client, lease.cookie).status_code == 200
    assert client.post(f"{BASE}/previews/{lease.record_id}/keepalive", headers=A).status_code == 204
    row = _row(lease.record_id)
    assert row.state == "active"
    assert row.intent_hash != lease.intent
    assert row.grant_hash is None
    assert row.session_hash != lease.cookie.split("=", 1)[1]
    assert row.viewer_cookie != viewer_cookie("a")
    assert decrypt(row.viewer_cookie) == viewer_cookie("a")
    viewer_requests = [request for request in services.requests if request.url.host == "viewer-login"]
    assert len(viewer_requests) >= 4
    assert all(request.headers["cookie"] == viewer_cookie("a") for request in viewer_requests)
    assert all("authorization" not in request.headers for request in viewer_requests)


def test_exchange_cookie_flags_and_no_cache_are_pinned(services, launched):
    client = _client()
    created = _create(client, launched)
    grant = _post(client, "authorize", created.json()["intent"], cookie=viewer_cookie("a")).json()["grant"]
    response = _post(client, "exchange", grant)
    assert response.status_code == 204
    attributes = response.headers["set-cookie"].split("; ")
    assert attributes[0].startswith(preview.preview_cookie(0) + "=")
    assert set(attributes[1:]) == {"Path=/", "HttpOnly", "Secure", "SameSite=None", "Partitioned"}
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_intent_and_grant_are_one_use(services, launched):
    client = _client()
    lease = _active(client, launched)
    assert _post(client, "authorize", lease.intent, cookie=viewer_cookie("a")).status_code == 403
    replay = _post(client, "exchange", lease.grant)
    assert replay.status_code == 403
    assert "set-cookie" not in replay.headers
    assert _access(client, lease.cookie).status_code == 200


@pytest.mark.parametrize("stage", ["pending", "granted", "active"])
def test_expired_capability_cannot_progress_or_renew(services, launched, monkeypatch, stage):
    client = _client()
    clock = preview.now()
    monkeypatch.setattr(preview, "now", lambda: clock)
    created = _create(client, launched).json()
    token = created["intent"]
    if stage != "pending":
        token = _post(client, "authorize", token, cookie=viewer_cookie("a")).json()["grant"]
    if stage == "active":
        cookie = _post(client, "exchange", token).headers["set-cookie"].split(";", 1)[0]
    clock += timedelta(seconds={"pending": preview.INTENT_SECONDS, "granted": preview.GRANT_SECONDS, "active": preview.LEASE_SECONDS}[stage])
    if stage == "pending":
        response = _post(client, "authorize", token, cookie=viewer_cookie("a"))
    elif stage == "granted":
        response = _post(client, "exchange", token)
    else:
        response = _access(client, cookie)
        assert client.post(f"{BASE}/previews/{created['id']}/keepalive", headers=A).status_code == 403
    assert response.status_code == 403
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize("origin", ["", "null", "http://localhost:8000", "http://cowork.localhost:8081"])
def test_create_rejects_other_or_missing_origins(services, launched, origin):
    assert _create(_client(), launched, origin=origin).status_code == 403


def test_owner_and_viewer_must_both_be_the_launcher(services, launched):
    client = _client()
    assert _create(client, launched, owner=B).status_code == 403
    created = _create(client, launched).json()
    assert _post(client, "authorize", created["intent"], cookie=viewer_cookie("b")).status_code == 403
    assert _post(client, "authorize", created["intent"]).status_code == 401
    assert _post(client, "authorize", created["intent"], cookie=viewer_cookie("a")).status_code == 200


@pytest.mark.parametrize("action", ["authorize", "exchange", "connect"])
def test_each_browser_post_requires_the_exact_app_origin(services, launched, action):
    client = _client()
    created = _create(client, launched).json()
    token = created["intent"]
    cookie = viewer_cookie("a")
    if action == "exchange":
        token = _post(client, "authorize", token, cookie=cookie).json()["grant"]
        cookie = ""
    elif action == "connect":
        lease = _active(client, launched)
        token, cookie = lease.intent, lease.cookie
    response = _post(client, action, token, cookie=cookie, origin="http://localhost:8000")
    assert response.status_code == 403
    assert "set-cookie" not in response.headers
    assert _post(client, action, token, cookie=cookie).status_code == (200 if action == "authorize" else 204)


def test_connect_requires_matching_intent_and_cookie(services, launched):
    client = _client()
    first = _active(client, launched)
    second = _active(client, launched)
    assert _post(client, "connect", first.intent, cookie=second.cookie).status_code == 401
    assert _post(client, "connect", first.intent).status_code == 401
    assert _post(client, "connect", first.intent, cookie=first.cookie).status_code == 204
    assert _access(client, first.cookie, slot=2).status_code == 403


@pytest.mark.parametrize("change", ["reboot", "owner", "stopped", "relaunch"])
def test_access_refuses_changed_slot_or_launch(services, launched, change):
    client = _client()
    lease = _active(client, launched)
    if change == "reboot":
        services.slots[1].boot_id = BOOT_2
    elif change == "owner":
        services.slots[1].launcher_subject = BOB.user_id
        services.slots[1].launcher_email = BOB.email
    elif change == "stopped":
        services.slots[1].state = "exited"
    else:
        _launch(services, 1, ALICE, slug="replacement", boot_id=BOOT_2)
    assert _access(client, lease.cookie).status_code == 403
    assert client.post(f"{BASE}/previews/{lease.record_id}/keepalive", headers=A).status_code == 403


def test_owner_revocation_clears_proof_and_cannot_be_undone_by_keepalive(services, launched):
    client = _client()
    lease = _active(client, launched)
    assert client.delete(f"{BASE}/previews/{lease.record_id}", headers=B).status_code == 204
    assert _access(client, lease.cookie).status_code == 200
    assert client.post(f"{BASE}/previews/{lease.record_id}/keepalive", headers=B).status_code == 403
    assert client.delete(f"{BASE}/previews", headers=A).status_code == 204
    row = _row(lease.record_id)
    assert (row.state, row.viewer_cookie, row.grant_hash, row.session_hash) == ("revoked", None, None, None)
    assert _access(client, lease.cookie).status_code == 403
    assert client.post(f"{BASE}/previews/{lease.record_id}/keepalive", headers=A).status_code == 403


def test_rotated_viewer_cookie_revokes_all_states_for_identity_only(services, launched):
    client = _client()
    active = _active(client, launched)
    pending = _create(client, launched).json()
    granted = _create(client, launched).json()
    _post(client, "authorize", granted["intent"], cookie=viewer_cookie("a")).raise_for_status()
    _launch(services, 2, BOB, slug="other", boot_id=BOOT_2)
    other = _active(client, _records(2)[-1].id, owner=B, viewer="b", slot=2)
    services.sessions["rotated-a"] = VIEWER_A
    response = client.get(f"{BASE}/1/preview/revoke-viewer", headers={"Cookie": viewer_cookie("rotated-a")})
    assert response.status_code == 204
    for record_id in (active.record_id, pending["id"], granted["id"]):
        row = _row(record_id)
        assert row.state == "revoked"
        assert row.viewer_cookie is None
    assert _access(client, active.cookie).status_code == 403
    assert _access(client, other.cookie, slot=2).status_code == 200


def test_refresh_headers_never_replace_or_escape_the_captured_viewer_proof(services, launched):
    client = _client()
    services.sessions["rotated-a"] = VIEWER_A
    services.refresh_cookies = ("_oauth2_proxy=rotated-a; Path=/; HttpOnly",)
    lease = _active(client, launched)
    before = _row(lease.record_id).viewer_cookie
    services.requests.clear()
    access = _access(client, lease.cookie + "; " + viewer_cookie("b"))
    assert access.status_code == 200
    assert "set-cookie" not in access.headers
    assert client.post(f"{BASE}/previews/{lease.record_id}/keepalive", headers=A).status_code == 204
    assert _row(lease.record_id).viewer_cookie == before
    assert decrypt(before) == viewer_cookie("a")
    assert all(request.headers["cookie"] == viewer_cookie("a") for request in services.requests if request.url.host == "viewer-login")
    del services.sessions["a"]
    assert _access(client, lease.cookie).status_code == 401
    assert client.post(f"{BASE}/previews/{lease.record_id}/keepalive", headers=A).status_code == 401


def test_viewer_identity_change_and_unavailability_fail_closed(services, launched):
    client = _client()
    lease = _active(client, launched)
    services.sessions["a"] = VIEWER_B
    assert _access(client, lease.cookie).status_code == 403
    services.viewer_fails_with = 503
    assert _access(client, lease.cookie).status_code == 503
    assert client.get(f"{BASE}/1/preview/revoke-viewer", headers={"Cookie": viewer_cookie("a")}).status_code == 503


@pytest.mark.parametrize("operation", ["check", "keepalive", "authorize"])
def test_revoke_during_viewer_verification_cannot_return_access(services, launched, monkeypatch, operation):
    client = _client()
    lease = _active(client, launched)
    pending = _create(client, launched).json()
    verify = preview._verify_viewer

    async def revoke_after_verification(*args, **kwargs):
        await verify(*args, **kwargs)
        await preview.revoke(ALICE)

    monkeypatch.setattr(preview, "_verify_viewer", revoke_after_verification)
    if operation == "check":
        response = _access(client, lease.cookie)
    elif operation == "keepalive":
        response = client.post(f"{BASE}/previews/{lease.record_id}/keepalive", headers=A)
    else:
        response = _post(client, "authorize", pending["intent"], cookie=viewer_cookie("a"))
    assert response.status_code == 403
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize("winner", ["consume", "revoke"])
def test_separate_database_sessions_cannot_both_consume_or_restore_a_grant(services, launched, winner):
    client = _client()
    created = _create(client, launched).json()
    grant = _post(client, "authorize", created["intent"], cookie=viewer_cookie("a")).json()["grant"]
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as first, Session(engine) as second:
        first_row = first.get(LiveAppPreview, UUID(created["id"]))
        second_row = second.get(LiveAppPreview, UUID(created["id"]))
        assert first_row.revision == second_row.revision
        first_row.state = "active" if winner == "consume" else "revoked"
        first_row.grant_hash = None
        first_row.viewer_cookie = None if winner == "revoke" else first_row.viewer_cookie
        first.add(first_row)
        second_row.state = "active"
        second_row.grant_hash = None
        second_row.session_hash = preview.digest("e" * 64)
        second.add(second_row)
        first.commit()
        with pytest.raises(StaleDataError):
            second.commit()
        second.rollback()
    assert _row(created["id"]).state == ("active" if winner == "consume" else "revoked")
    assert _post(client, "exchange", grant).status_code == 403


def test_capacity_is_bounded_and_a_revoked_position_is_reusable(services, launched):
    client = _client()
    records = [_create(client, launched).json() for _ in range(preview.MAX_PREVIEWS)]
    assert _create(client, launched).status_code == 429
    assert client.delete(f"{BASE}/previews/{records[0]['id']}", headers=A).status_code == 204
    replacement = _create(client, launched)
    assert replacement.status_code == 200
    assert replacement.json()["id"] != records[0]["id"]
    assert _post(client, "authorize", records[0]["intent"], cookie=viewer_cookie("a")).status_code == 403
    with Session(get_engine(get_app_settings().database.uri)) as session:
        assert len(session.exec(select(LiveAppPreview)).all()) == preview.MAX_PREVIEWS


def test_preview_routes_fail_closed_when_live_apps_are_off(services, monkeypatch):
    monkeypatch.delenv("ANTON_LIVE_APP_SLOTS")
    get_app_settings.cache_clear()
    client = _client()
    for action in ("authorize", "exchange", "connect"):
        response = client.post(f"{BASE}/1/preview/{action}", headers=A, json={"token": "a" * 64})
        assert response.status_code == 404
    assert client.get(f"{BASE}/1/preview/bootstrap", headers=A).status_code == 404


def test_wrong_slot_cannot_consume_an_intent_or_grant(services, launched):
    client = _client()
    intent = _create(client, launched).json()["intent"]
    assert _post(client, "authorize", intent, cookie=viewer_cookie("a"), slot=2).status_code == 403
    grant = _post(client, "authorize", intent, cookie=viewer_cookie("a")).json()["grant"]
    assert _post(client, "exchange", grant, slot=2).status_code == 403
    assert _post(client, "exchange", grant).status_code == 204


def test_keepalive_extends_an_active_lease_and_a_pending_intent_but_not_a_grant(services, launched, monkeypatch):
    client = _client()
    clock = preview.now()
    monkeypatch.setattr(preview, "now", lambda: clock)
    lease = _active(client, launched)
    pending = _create(client, launched).json()
    granted = _create(client, launched).json()
    _post(client, "authorize", granted["intent"], cookie=viewer_cookie("a")).raise_for_status()
    granted_expiry = _row(granted["id"]).expires_at
    clock += timedelta(seconds=20)
    assert client.post(f"{BASE}/previews/{lease.record_id}/keepalive", headers=A).status_code == 204
    assert client.post(f"{BASE}/previews/{pending['id']}/keepalive", headers=A).status_code == 204
    assert client.post(f"{BASE}/previews/{granted['id']}/keepalive", headers=A).status_code == 204
    assert ensure_utc(_row(pending["id"]).expires_at) == clock + timedelta(seconds=preview.INTENT_SECONDS)
    assert _row(granted["id"]).expires_at == granted_expiry
    clock += timedelta(seconds=30)
    assert _access(client, lease.cookie).status_code == 200
    clock += timedelta(seconds=30)
    assert _access(client, lease.cookie).status_code == 403


def test_two_worker_loops_exchange_a_grant_exactly_once(services, launched, monkeypatch):
    client = _client()
    intent = _create(client, launched).json()["intent"]
    grant = _post(client, "authorize", intent, cookie=viewer_cookie("a")).json()["grant"]
    barrier = Barrier(2)
    snapshot = preview._snapshot

    def simultaneous_snapshot(*args, **kwargs):
        result = snapshot(*args, **kwargs)
        if kwargs.get("field") == "grant_hash":
            barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(preview, "_snapshot", simultaneous_snapshot)

    def exchange():
        try:
            return asyncio.run(preview.exchange(1, grant))
        except preview.PreviewRefused:
            return None

    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = list(workers.map(lambda _: exchange(), range(2)))
    leases = [lease for lease in outcomes if lease is not None]
    assert len(leases) == 1
    assert outcomes.count(None) == 1
    name, token = leases[0]
    assert _access(client, f"{name}={token}").status_code == 200


def test_app_recreation_keeps_capability_consumption_in_the_database(services, launched):
    intent = _create(_client(), launched).json()["intent"]
    grant = _post(_client(), "authorize", intent, cookie=viewer_cookie("a")).json()["grant"]
    exchanged = _post(_client(), "exchange", grant)
    cookie = exchanged.headers["set-cookie"].split(";", 1)[0]
    assert _post(_client(), "exchange", grant).status_code == 403
    assert _post(_client(), "connect", intent, cookie=cookie).status_code == 204


@pytest.mark.parametrize("owner_exists", [False, True])
def test_owner_gate_serializes_create_with_logout_across_worker_loops(services, launched, monkeypatch, owner_exists):
    if owner_exists:
        asyncio.run(preview.create(ALICE, launched, COWORK))
    operation = ContextVar("preview_owner_race_operation", default="")
    creator_read = Event()
    logout_completed = Event()
    original_get = ScopedSession.get

    def read_owner_before_logout(session, model, ident):
        row = original_get(session, model, ident)
        if model is LiveAppPreviewOwner and operation.get() == "create":
            creator_read.set()
            assert logout_completed.wait(timeout=5)
        return row

    monkeypatch.setattr(ScopedSession, "get", read_owner_before_logout)

    def create():
        operation.set("create")
        try:
            return asyncio.run(preview.create(ALICE, launched, COWORK))
        except preview.PreviewRefused:
            return None

    def revoke():
        assert creator_read.wait(timeout=5)
        try:
            asyncio.run(preview.revoke(ALICE))
        finally:
            logout_completed.set()

    with ThreadPoolExecutor(max_workers=2) as workers:
        create_future = workers.submit(create)
        revoke_future = workers.submit(revoke)
        created = create_future.result(timeout=15)
        revoke_future.result(timeout=15)
    with Session(get_engine(get_app_settings().database.uri)) as session:
        rows = session.exec(select(LiveAppPreview)).all()
        assert all(row.state == "revoked" for row in rows)
        assert len(session.exec(select(LiveAppPreviewOwner)).all()) == 1
    if created is not None:
        _, intent = created
        assert _post(_client(), "authorize", intent, cookie=viewer_cookie("a")).status_code == 403


@pytest.mark.parametrize("conflicts, status", [(2, 204), (3, 503)])
def test_revoke_retries_whole_transactions_and_fails_closed_when_exhausted(services, launched, monkeypatch, conflicts, status):
    client = _client()
    lease = _active(client, launched)
    revoke_rows = preview._revoke_rows
    attempts = 0

    def conflicting_revoke(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        revoke_rows(*args, **kwargs)
        if attempts <= conflicts:
            raise StaleDataError("synthetic concurrent lease update")

    monkeypatch.setattr(preview, "_revoke_rows", conflicting_revoke)
    response = client.delete(f"{BASE}/previews", headers=A)
    assert response.status_code == status
    assert attempts == 3
    row = _row(lease.record_id)
    if status == 204:
        assert (row.state, row.viewer_cookie, row.session_hash) == ("revoked", None, None)
        assert _access(client, lease.cookie).status_code == 403
    else:
        assert row.state == "active"
        assert row.viewer_cookie is not None
        assert response.headers["cache-control"] == "no-store"


def test_reloading_can_revoke_one_lease_while_creating_its_replacement(services, launched, monkeypatch):
    client = _client()
    previous = _active(client, launched)
    operation = ContextVar("preview_test_operation", default="")
    barrier = Barrier(2)
    creator_read = Event()
    revoker_read = Event()
    original_get, original_exec = ScopedSession.get, ScopedSession.exec

    def read_owner(session, model, ident):
        row = original_get(session, model, ident)
        if model is LiveAppPreviewOwner and operation.get() == "create" and not creator_read.is_set():
            creator_read.set()
            barrier.wait(timeout=5)
        return row

    def read_previous(session, statement):
        result = original_exec(session, statement)
        if operation.get() == "revoke-one" and not revoker_read.is_set():
            revoker_read.set()
            barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(ScopedSession, "get", read_owner)
    monkeypatch.setattr(ScopedSession, "exec", read_previous)

    def create():
        operation.set("create")
        return asyncio.run(preview.create(ALICE, launched, COWORK))

    def revoke_one():
        operation.set("revoke-one")
        asyncio.run(preview.revoke(ALICE, UUID(previous.record_id)))

    with ThreadPoolExecutor(max_workers=2) as workers:
        created = workers.submit(create)
        revoked = workers.submit(revoke_one)
        _, intent = created.result(timeout=15)
        revoked.result(timeout=15)
    assert creator_read.is_set() and revoker_read.is_set()
    assert _access(client, previous.cookie).status_code == 403
    assert _post(client, "authorize", intent, cookie=viewer_cookie("a")).status_code == 200


def test_reload_with_revoked_cookie_requests_sign_in_for_valid_replacement_only(services, launched, monkeypatch):
    client = _client()
    clock = preview.now()
    monkeypatch.setattr(preview, "now", lambda: clock)
    previous = _active(client, launched)
    assert client.delete(f"{BASE}/previews/{previous.record_id}", headers=A).status_code == 204
    replacement = _create(client, launched).json()

    reconnect = _post(client, "connect", replacement["intent"], cookie=previous.cookie)
    assert reconnect.status_code == 401
    assert "set-cookie" not in reconnect.headers
    assert _post(client, "connect", "0" * 64).status_code == 403

    authorized = _post(client, "authorize", replacement["intent"], cookie=viewer_cookie("a"))
    assert authorized.status_code == 200
    exchanged = _post(client, "exchange", authorized.json()["grant"])
    assert exchanged.status_code == 204
    cookie = exchanged.headers["set-cookie"].split(";", 1)[0]
    assert _post(client, "connect", replacement["intent"], cookie=cookie).status_code == 204

    expired = _create(client, launched).json()["intent"]
    clock += timedelta(seconds=preview.INTENT_SECONDS)
    assert _post(client, "connect", expired).status_code == 403


def test_a_malformed_neighbour_cookie_hides_neither_the_viewer_cookie_nor_the_lease(services, launched):
    """An app's own script can write a cookie SimpleCookie rejects, which made it drop the whole header."""
    client = _client()
    odd = "theme=dark mode"
    created = _create(client, launched).json()
    authorized = _post(client, "authorize", created["intent"], cookie=f"{odd}; {viewer_cookie('a')}")
    assert authorized.status_code == 200, authorized.text
    exchanged = _post(client, "exchange", authorized.json()["grant"])
    cookie = exchanged.headers["set-cookie"].split(";", 1)[0]
    assert _post(client, "connect", created["intent"], cookie=f"{cookie}; {odd}").status_code == 204
    assert _access(client, f"{odd}; {cookie}").status_code == 200
    signed_out = client.get(f"{BASE}/1/preview/revoke-viewer", headers={"Cookie": f"{viewer_cookie('a')}; {odd}"})
    assert signed_out.status_code == 204
    assert _row(created["id"]).state == "revoked"
    assert _access(client, f"{odd}; {cookie}").status_code == 403


def test_two_panes_in_one_browser_keep_separate_leases(services, launched):
    client = _client()
    first = _active(client, launched)
    second = _active(client, launched)
    assert first.cookie.split("=", 1)[0] != second.cookie.split("=", 1)[0]
    browser = f"{first.cookie}; {second.cookie}"
    assert _post(client, "connect", first.intent, cookie=browser).status_code == 204
    assert _post(client, "connect", second.intent, cookie=browser).status_code == 204
    assert client.delete(f"{BASE}/previews/{second.record_id}", headers=A).status_code == 204
    assert _access(client, browser).status_code == 200
    assert client.delete(f"{BASE}/previews/{first.record_id}", headers=A).status_code == 204
    assert _access(client, browser).status_code == 403


def test_an_open_pane_keeps_its_intent_through_a_slow_sign_in(services, launched, monkeypatch):
    client = _client()
    clock = preview.now()
    monkeypatch.setattr(preview, "now", lambda: clock)
    created = _create(client, launched).json()
    for _ in range(4):
        clock += timedelta(seconds=preview.INTENT_SECONDS - 10)
        assert client.post(f"{BASE}/previews/{created['id']}/keepalive", headers=A).status_code == 204
    assert client.post(f"{BASE}/previews/{created['id']}/keepalive", headers=B).status_code == 403
    assert _post(client, "authorize", created["intent"], cookie=viewer_cookie("a")).status_code == 200


def test_a_tab_with_a_stale_or_planted_lease_cookie_still_opens_for_the_launcher(services, launched):
    planted = f"{preview.preview_cookie(0)}=x"
    client = _client()
    assert _access(client, f"{planted}; {viewer_cookie('a')}").status_code == 200
    assert _access(client, f"{planted}; {viewer_cookie('b')}").status_code == 403
    assert _access(client, planted).status_code == 403


@pytest.mark.parametrize("slot_state", ["running", "exited"])
def test_the_stopped_page_in_a_pane_speaks_to_the_lease_owner(services, launched, slot_state):
    """A frame carries its lease but not the viewer cookie: a hung or stopped app explains itself there."""
    client = _client()
    lease = _active(client, launched)
    services.slots[1].state = slot_state
    page = client.get(f"{BASE}/1/stopped-page", headers={"Cookie": lease.cookie})
    assert page.status_code == 200
    assert "have access" not in page.text
    assert client.delete(f"{BASE}/previews", headers=A).status_code == 204
    assert client.get(f"{BASE}/1/stopped-page", headers={"Cookie": lease.cookie}).status_code == 403


def test_the_next_preview_drops_expired_copies_of_the_viewer_cookie(services, launched, monkeypatch):
    client = _client()
    clock = preview.now()
    monkeypatch.setattr(preview, "now", lambda: clock)
    for _ in range(3):
        _active(client, launched)
    clock += timedelta(seconds=preview.LEASE_SECONDS)
    assert _create(client, launched).status_code == 200
    with Session(get_engine(get_app_settings().database.uri)) as session:
        assert [row.viewer_cookie for row in session.exec(select(LiveAppPreview)).all()] == [None] * 3


def test_preview_addresses_follow_the_deployment_settings(services, launched, monkeypatch):
    monkeypatch.setenv("ANTON_LIVE_APP_ADDRESS", "http://app{n}.view.localhost:8081/")
    monkeypatch.setenv("COWORK_ALLOWED_ORIGINS", '["http://cowork.localhost:8081"]')
    monkeypatch.setenv("COWORK_LIVE_APPS_VIEWER_PUBLIC_URL", "http://view.localhost:8081")
    get_app_settings.cache_clear()
    client = _client()
    assert _create(client, launched, origin="http://cowork.localhost:8081").status_code == 403
    with Session(get_engine(get_app_settings().database.uri)) as session:
        record = session.get(LiveAppLaunchRecord, launched)
        record.address = "http://app1.view.localhost:8081/"
        session.add(record)
        session.commit()
    assert _create(client, launched).status_code == 403
    created = _create(client, launched, origin="http://cowork.localhost:8081")
    assert created.status_code == 200, created.text
    authorized = _post(client, "authorize", created.json()["intent"], cookie=viewer_cookie("a"))
    assert authorized.status_code == 200
    assert _post(client, "authorize", created.json()["intent"], cookie=viewer_cookie("a"),
                 origin="http://app1.view.localhost:8080").status_code == 403
    page = client.get(f"{BASE}/1/preview/authorize").text
    assert '"cowork": ["http://cowork.localhost:8081"]' in page
    assert '"app": "http://app1.view.localhost:8081"' in page
    assert '"signIn": "http://view.localhost:8081/oauth2/start"' in page
    assert "8080" not in page


def test_previews_are_refused_without_the_viewer_sign_in_address(services, launched, monkeypatch):
    monkeypatch.delenv("COWORK_LIVE_APPS_VIEWER_PUBLIC_URL")
    get_app_settings.cache_clear()
    refused = _create(_client(), launched)
    assert refused.status_code == 503
    assert "Open the app in your browser" in refused.json()["detail"]
