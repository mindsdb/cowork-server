"""Who opens a live app: only the person who launched the app its slot runs
now, by the server's own launch record and the slot's own status, never by
anything the request says. Then how a stop is recorded and explained: the
slot's report, the card's state, and the page at the old address."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

# Imported at module scope on purpose: cowork.server builds an app at import
# time (see tests/test_trusted_identity_local.py).
import cowork.server as cowork_server
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.live_app import LiveAppLaunchRecord
from cowork.services import live_apps
from cowork.services.live_apps import CoworkLiveAppHost, LiveAppPerson, stop_reason_text, stop_time_text
from tests._logging_assertions import assert_log_fields
from tests._live_apps import (
    ALICE,
    BEARER_TOKEN,
    BOB,
    EMAIL_A,
    EMAIL_B,
    LAUNCH_TOKEN,
    MODULE,
    PARTITION_A,
    SLOT_ADDRESSES,
    SLOTS_VARIABLE,
    SUB_A,
    SUB_B,
    VIEWER_A,
    VIEWER_B,
    A,
    B,
    FakeServices,
    ViewerUser,
    answer_over_http,
    clear_live_app_rows,
    general_project,
    headers_for,
    person_scope,
    turn_on_live_apps,
    viewer_cookie,
)

BOOT_1 = "1" * 32
BOOT_2 = "2" * 32


@pytest.fixture
def services(personal_deployment, monkeypatch, tmp_path):
    """A stack with live apps on: A and B signed in to the viewer, three empty
    slots, and every viewer-login and slot call answered by FakeServices."""
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "home" / "projects"))
    monkeypatch.setenv("COWORK_AUTH_TOKEN", BEARER_TOKEN)
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    turn_on_live_apps(monkeypatch)
    fake = FakeServices(sessions={"a": VIEWER_A, "b": VIEWER_B})
    answer_over_http(monkeypatch, fake)
    yield fake
    clear_live_app_rows()
    get_app_settings.cache_clear()


def _client() -> TestClient:
    get_app_settings.cache_clear()
    return TestClient(cowork_server.create_app())


def _launch_model(slot: int, slug: str, boot_id: str):
    return sys.modules[MODULE].LiveAppLaunch(
        slot=slot,
        slug=slug,
        artifact_dir=f"/projects/p/.anton/artifacts/{slug}",
        address=f"http://app{slot}.view.localhost:8080/",
        boot_id=boot_id,
    )


PROJECT = uuid4()


def _host(who: LiveAppPerson, project_id=PROJECT) -> CoworkLiveAppHost:
    return CoworkLiveAppHost(who, project_id=project_id, launch_token=LAUNCH_TOKEN)


def _launch(services: FakeServices, slot: int, who: LiveAppPerson, *, slug: str, boot_id: str, project_id=PROJECT) -> None:
    """``who`` launches ``slug`` into ``slot`` as anton does: claim, the slot
    takes it, launched."""
    host = _host(who, project_id)

    async def launch() -> None:
        await host.claim(slot)
        await host.launched(_launch_model(slot, slug, boot_id))

    asyncio.run(launch())
    services.run(slot, boot_id=boot_id, artifact=slug, subject=who.user_id, email=who.email)


def _records(slot: int | None = None) -> list[LiveAppLaunchRecord]:
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        statement = select(LiveAppLaunchRecord).order_by(LiveAppLaunchRecord.started_at)
        if slot is not None:
            statement = statement.where(LiveAppLaunchRecord.slot == slot)
        return list(raw.exec(statement).all())


def _access(client: TestClient, slot: int, cookie: str | None, **kwargs):
    headers = {"Cookie": viewer_cookie(cookie)} if cookie else {}
    headers.update(kwargs.pop("headers", {}))
    return client.get(f"/api/v1/live-apps/{slot}/access", headers=headers, **kwargs)


# ── the access check ─────────────────────────────────────────────────────────


def test_the_launcher_opens_the_app_with_only_a_cookie(services):
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)

    response = _access(_client(), 1, "a")

    assert response.status_code == 200
    # No bearer and no identity went to the route; the cookie went to viewer-login.
    (userinfo,) = [r for r in services.requests if r.url.host == "viewer-login"]
    assert userinfo.headers["cookie"] == viewer_cookie("a")


def test_someone_else_signed_in_is_refused(services):
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)

    assert _access(_client(), 1, "b").status_code == 403


def test_no_cookie_and_an_unknown_one_are_signed_out(services):
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    client = _client()

    assert _access(client, 1, None).status_code == 401
    assert _access(client, 1, "nobody").status_code == 401
    # No cookie asks viewer-login nothing.
    assert [r.url.path for r in services.requests if r.url.host == "viewer-login"] == ["/oauth2/userinfo"]


def test_forged_identity_and_a_query_change_nothing(services):
    """The owner comes from the launch record, never the request."""
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    forged = {
        "X-Auth-Request-Email": EMAIL_A,
        "X-Forwarded-Email": EMAIL_A,
        "X-User-Email": EMAIL_A,
        "X-User-Id": SUB_A,
    }

    response = _access(_client(), 1, "b", headers=forged, params={"email": EMAIL_A})

    assert response.status_code == 403


def test_an_empty_slot_opens_for_nobody(services):
    assert _access(_client(), 2, "a").status_code == 403


def test_the_slot_must_still_run_the_recorded_launch(services):
    """A slot that restarted, or runs another launch, opens for nobody, even
    when the record still names the visitor."""
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    client = _client()

    services.slots[1].boot_id = BOOT_2
    assert _access(client, 1, "a").status_code == 403

    services.run(1, boot_id=BOOT_1, artifact="dash", subject=SUB_B, email=EMAIL_B)
    assert _access(client, 1, "a").status_code == 403

    # The slot's launcher has A's email on another account.
    services.run(1, boot_id=BOOT_1, artifact="dash", subject=SUB_B, email=EMAIL_A)
    assert _access(client, 1, "a").status_code == 403

    # An exited slot still reports its app, A's: the state alone refuses.
    services.run(1, boot_id=BOOT_1, artifact="dash", subject=SUB_A, email=EMAIL_A)
    services.slots[1].state = "exited"
    assert _access(client, 1, "a").status_code == 403


def test_the_email_compares_without_case(services):
    services.sessions["a"] = ViewerUser(SUB_A, "Alice@Example.COM")
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)

    assert _access(_client(), 1, "a").status_code == 200


def test_the_launchers_email_on_another_account_is_refused(services):
    """The record and the slot name the launcher by sub as well as email, so
    an address the operator gave to another account opens nothing."""
    services.sessions["c"] = ViewerUser("44444444-4444-4444-8444-444444444444", EMAIL_A)
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)

    assert _access(_client(), 1, "c").status_code == 403


def test_viewer_login_or_the_slot_failing_answers_503(services):
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    client = _client()

    services.slots[1].fails_with = 500
    assert _access(client, 1, "a").status_code == 503

    services.slots[1].fails_with = None
    services.viewer_fails_with = 502
    assert _access(client, 1, "a").status_code == 503


def test_an_unavailable_check_logs_the_slot_and_which_service_failed(services, owned_logger):
    from cowork.api.v1.endpoints import live_apps as live_apps_routes

    logged = owned_logger(live_apps_routes.logger.name, level=logging.WARNING)
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    services.viewer_fails_with = 502

    assert _access(_client(), 1, "a").status_code == 503

    assert_log_fields(
        logged.output(), event="live_app_access_unavailable", slot=1, reason="viewer_login_status_502"
    )


@pytest.mark.parametrize(
    ("reply", "status"),
    [
        (httpx.Response(200, json={}), 401),
        (httpx.Response(200, json={"user": SUB_A, "email": ""}), 401),
        (httpx.Response(200, content=b"<html>sign in</html>"), 503),
        (httpx.Response(302, headers={"location": "/oauth2/sign_in"}), 503),
        (httpx.ReadTimeout("timed out"), 503),
        (httpx.Response(200, content=b"x" * (1024 * 1024 + 1)), 503),
    ],
    ids=["no-session", "no-email", "not-json", "redirect", "timeout", "too-large"],
)
def test_viewer_logins_other_answers(services, reply, status):
    """oauth2-proxy answers 200 {} for a request it finds no session for, and
    a session with no email is nobody signed in; anything else it can't say
    is an outage, never an allow."""
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    services.viewer_replies["x"] = reply

    assert _access(_client(), 1, "x").status_code == status


def test_a_refreshed_viewer_session_sets_no_cookie(services):
    """viewer-login answers a refresh with its session split over several
    Set-Cookie headers, which nginx would join into one broken cookie, so no
    reply passes them on."""
    services.refresh_cookies = ("_oauth2_proxy_0=part0; Path=/", "_oauth2_proxy_1=part1; Path=/")
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    client = _client()

    access = _access(client, 1, "a")
    page = client.get("/api/v1/live-apps/1/stopped-page", headers={"Cookie": viewer_cookie("a")})

    assert access.status_code == 200
    assert "set-cookie" not in access.headers
    assert "set-cookie" not in page.headers


def test_a_cookie_the_app_set_with_any_byte_still_opens_it(services):
    """The app's own page can set a cookie holding UTF-8; viewer-login gets
    the browser's exact bytes, and the launcher still opens their app."""
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    raw_cookie = viewer_cookie("a").encode() + b"; filter=Caf\xc3\xa9"

    response = _client().get("/api/v1/live-apps/1/access", headers={"Cookie": raw_cookie})

    assert response.status_code == 200
    (userinfo,) = [r for r in services.requests if r.url.host == "viewer-login"]
    assert [value for name, value in userinfo.headers.raw if name.lower() == b"cookie"] == [raw_cookie]


@pytest.mark.parametrize("slot", [0, 4])
def test_a_slot_outside_the_list_is_404(services, slot):
    """Only the listed slots' paths skip the bearer: another slot's needs it,
    and then answers 404."""
    client = _client()

    assert _access(client, slot, "a").status_code == 401
    assert _access(client, slot, "a", headers=A).status_code == 404


# ── launch records ───────────────────────────────────────────────────────────


def test_a_launch_that_never_lands_opens_the_slot_for_nobody(services):
    """claim stops the slot's record before anton launches; a launch the slot
    refuses never calls launched, so even the slot's previous app is shut,
    and its reason holds for a relaunch of the same app that failed too."""
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    asyncio.run(_host(ALICE).claim(1))

    assert _access(_client(), 1, "a").status_code == 403
    (record,) = _records(1)
    assert (record.state, record.stop_reason) == ("stopped", "replaced")
    assert stop_reason_text(record.stop_reason, limit_hours=None) == "a launch of yours took its slot"


def test_a_relaunch_of_the_same_app_is_no_stop(services):
    project = uuid4()
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1, project_id=project)
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_2, project_id=project)

    first, second = _records(1)
    assert (first.state, first.stop_reason) == ("stopped", "relaunched")
    assert (second.state, second.boot_id) == ("running", BOOT_2)
    assert _access(_client(), 1, "a").status_code == 200


def test_replacing_ones_own_app_records_it(services):
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    _launch(services, 1, ALICE, slug="report", boot_id=BOOT_2)

    first, second = _records(1)
    assert (first.slug, first.stop_reason, first.stop_noticed) == ("dash", "replaced", True)
    assert (second.slug, second.state) == ("report", "running")


def _launch_into_a_free_slot(services: FakeServices, slot: int, who: LiveAppPerson, *, slug: str, boot_id: str, project_id=PROJECT) -> None:
    """anton's launch into an empty or exited slot: no claim, only launched."""
    asyncio.run(_host(who, project_id).launched(_launch_model(slot, slug, boot_id)))
    services.run(slot, boot_id=boot_id, artifact=slug, subject=who.user_id, email=who.email)


def test_relaunching_an_app_that_ended_unreported_is_no_stop(services):
    project = uuid4()
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1, project_id=project)
    services.slots[1].state = "exited"
    _launch_into_a_free_slot(services, 1, ALICE, slug="dash", boot_id=BOOT_2, project_id=project)

    first, second = _records(1)
    assert (first.stop_reason, first.stop_noticed) == ("relaunched", True)
    assert (second.state, second.boot_id) == ("running", BOOT_2)


def test_a_free_slot_ends_another_persons_unreported_record(services):
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    services.slots[1].state = "empty"
    _launch_into_a_free_slot(services, 1, BOB, slug="board", boot_id=BOOT_2)

    alice, bob = _records(1)
    assert (alice.stop_reason, alice.stop_noticed) == ("lost", False)
    assert bob.state == "running"


def test_another_persons_record_on_a_claimed_slot_is_lost(services):
    """anton claims only a slot whose status runs the claimer's own app, so
    another person's running record there is one the slot no longer shows."""
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    _launch(services, 1, BOB, slug="board", boot_id=BOOT_2)

    alice, bob = _records(1)
    assert (alice.state, alice.stop_reason) == ("stopped", "lost")
    assert bob.launcher_email == EMAIL_B
    client = _client()
    assert _access(client, 1, "a").status_code == 403
    assert _access(client, 1, "b").status_code == 200


# ── the slot's stop report ───────────────────────────────────────────────────


@pytest.fixture
def from_slot_1(services, monkeypatch):
    """TestClient's peer is "testclient": make that slot 1's own address."""

    async def slot_ips(slot: int) -> frozenset[str]:
        return frozenset({"testclient"}) if slot == 1 else frozenset({"172.30.16.12"})

    monkeypatch.setattr(live_apps, "_slot_ips", slot_ips)
    return services


def _report(client: TestClient, slot: int, **body):
    headers = body.pop("headers", {})
    payload = {"boot_id": BOOT_1, "artifact": "dash", "reason": "idle", **body}
    return client.post(f"/api/v1/live-apps/{slot}/stopped", json=payload, headers=headers)


def test_the_slots_report_stops_its_running_record(from_slot_1):
    _launch(from_slot_1, 1, ALICE, slug="dash", boot_id=BOOT_1)

    response = _report(_client(), 1, limit_hours=2)

    assert response.status_code == 204
    (record,) = _records(1)
    assert (record.state, record.stop_reason, record.limit_hours) == ("stopped", "idle", 2)


def test_the_report_keeps_the_slots_stop_time_within_the_launch(from_slot_1):
    _launch(from_slot_1, 1, ALICE, slug="dash", boot_id=BOOT_1)
    (record,) = _records(1)
    started = record.started_at.replace(tzinfo=timezone.utc)
    # Just after the launch: earlier than the report reaches the server.
    when = started + timedelta(microseconds=1)

    assert _report(_client(), 1, stopped_at=when.isoformat()).status_code == 204

    (record,) = _records(1)
    assert record.stopped_at.replace(tzinfo=timezone.utc) == when

    # A time before the launch, or after now, is held to the launch's life.
    _launch(from_slot_1, 1, ALICE, slug="dash", boot_id=BOOT_2)
    before = datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert _report(_client(), 1, boot_id=BOOT_2, stopped_at=before.isoformat()).status_code == 204
    latest = _records(1)[-1]
    assert latest.stopped_at.replace(tzinfo=timezone.utc) == latest.started_at.replace(tzinfo=timezone.utc)


def test_a_report_from_another_address_is_refused(from_slot_1):
    _launch(from_slot_1, 2, ALICE, slug="dash", boot_id=BOOT_1)

    assert _report(_client(), 2).status_code == 403
    (record,) = _records(2)
    assert record.state == "running"


@pytest.mark.parametrize(
    ("header", "value"),
    [("X-Forwarded-For", "172.30.16.11"), ("Forwarded", "for=172.30.16.11"), ("X-Real-IP", "172.30.16.11")],
)
def test_a_report_through_a_proxy_is_refused(from_slot_1, header, value):
    """uvicorn rewrites the client address from a forwarding header for a
    trusted peer, which agent code in the cowork container can be, so a
    report carrying any of them is refused."""
    _launch(from_slot_1, 1, ALICE, slug="dash", boot_id=BOOT_1)

    assert _report(_client(), 1, headers={header: value}).status_code == 403
    (record,) = _records(1)
    assert record.state == "running"


def test_a_refused_report_logs_the_slot_and_why(from_slot_1, owned_logger):
    from cowork.api.v1.endpoints import live_apps as live_apps_routes

    logged = owned_logger(live_apps_routes.logger.name, level=logging.WARNING)
    _launch(from_slot_1, 1, ALICE, slug="dash", boot_id=BOOT_1)

    assert _report(_client(), 1, headers={"X-Forwarded-For": "172.30.16.11"}).status_code == 403

    assert_log_fields(logged.output(), event="live_app_stop_report_refused", slot=1, reason="forwarded")


def test_the_report_is_checked_against_the_slots_resolved_address(services, monkeypatch):
    """The real check: slot n's host name, resolved here, must be the
    report's peer."""
    resolved = {"live-app-1": "testclient", "live-app-2": "172.30.16.12"}

    async def getaddrinfo(self, host, port, *, family=0, type=0, proto=0, flags=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (resolved[host], 0))]

    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", getaddrinfo)
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    _launch(services, 2, ALICE, slug="report", boot_id=BOOT_1)
    client = _client()

    assert _report(client, 1).status_code == 204
    assert _report(client, 2).status_code == 403


def test_a_report_for_another_boot_or_a_reset_changes_nothing(from_slot_1):
    _launch(from_slot_1, 1, ALICE, slug="dash", boot_id=BOOT_1)
    client = _client()

    assert _report(client, 1, boot_id=BOOT_2).status_code == 204
    assert _report(client, 1, reason="reset").status_code == 204
    (record,) = _records(1)
    assert record.state == "running"


def test_a_report_that_doesnt_parse_is_422(from_slot_1):
    assert _report(_client(), 1, reason="vanished").status_code == 422


def test_the_stop_routes_need_no_bearer(from_slot_1):
    """The slot paths pass the bearer check: a 403 from the route itself, not a 401."""
    assert _report(_client(), 2).status_code == 403


# ── the card's state ─────────────────────────────────────────────────────────


def _mine(client: TestClient, headers=A):
    return client.get("/api/v1/live-apps/mine", headers=headers)


def test_mine_lists_the_callers_apps_with_their_state(from_slot_1):
    project = uuid4()
    _launch(from_slot_1, 1, ALICE, slug="dash", boot_id=BOOT_1, project_id=project)
    _launch(from_slot_1, 2, ALICE, slug="report", boot_id=BOOT_2, project_id=project)
    client = _client()
    assert _report(client, 1, limit_hours=2).status_code == 204

    mine = _mine(client).json()

    # Newest first: report launched after dash.
    assert [entry["slug"] for entry in mine] == ["report", "dash"]
    by_slug = {entry["slug"]: entry for entry in mine}
    assert by_slug["report"]["state"] == "running"
    assert by_slug["report"]["address"] == "http://app2.view.localhost:8080/"
    assert by_slug["report"]["stopReason"] is None
    assert by_slug["dash"]["state"] == "stopped"
    assert by_slug["dash"]["stopReason"] == "idle"
    assert by_slug["dash"]["stopReasonText"] == "no requests for 2 h"
    assert by_slug["dash"]["projectId"] == str(project)
    assert _mine(client, headers=B).json() == []


def test_mine_lists_a_relaunched_app_once_as_running(services):
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_2)

    (entry,) = _mine(_client()).json()

    assert (entry["state"], entry["stopReason"]) == ("running", None)


def test_an_app_its_slot_no_longer_runs_is_lost_at_the_slots_new_boot(services):
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    (record,) = _records(1)
    booted = record.started_at.replace(tzinfo=timezone.utc) + timedelta(microseconds=1)
    services.slots[1].boot_id = BOOT_2
    services.slots[1].state = "empty"
    services.slots[1].booted_at = booted.isoformat().replace("+00:00", "Z")

    (entry,) = _mine(_client()).json()

    assert (entry["state"], entry["stopReason"]) == ("stopped", "lost")
    assert entry["stopReasonText"] == "its app slot restarted"
    (record,) = _records(1)
    assert record.stop_reason == "lost"
    assert record.stopped_at.replace(tzinfo=timezone.utc) == booted


def test_an_exited_slot_keeps_the_record_for_its_report(from_slot_1):
    """A slot sets exited, ends the app, then reports: a read in between
    leaves the record running, and the report then records its own reason."""
    _launch(from_slot_1, 1, ALICE, slug="dash", boot_id=BOOT_1)
    from_slot_1.slots[1].state = "exited"
    client = _client()

    (entry,) = _mine(client).json()
    assert entry["state"] == "running"

    assert _report(client, 1, reason="max_time", limit_hours=8).status_code == 204
    (entry,) = _mine(client).json()
    assert (entry["stopReason"], entry["stopReasonText"]) == ("max_time", "it reached its 8 h limit")


def test_a_report_that_came_before_the_record_is_read_from_the_slots_status(from_slot_1):
    """The app exits and the slot reports before launched writes the record,
    which leaves the report nothing to stop. The slot's status still says
    why, so the record stops for that reason, not as lost once the slot
    restarts."""
    client = _client()
    assert _report(client, 1, reason="app_exited").status_code == 204
    _launch(from_slot_1, 1, ALICE, slug="dash", boot_id=BOOT_1)
    from_slot_1.slots[1].state = "exited"
    from_slot_1.slots[1].stop_reason = "app_exited"

    (entry,) = _mine(client).json()

    assert (entry["state"], entry["stopReason"]) == ("stopped", "app_exited")


def test_a_report_after_the_status_said_why_fills_in_the_hours(from_slot_1):
    _launch(from_slot_1, 1, ALICE, slug="dash", boot_id=BOOT_1)
    from_slot_1.slots[1].state = "exited"
    from_slot_1.slots[1].stop_reason = "idle"
    client = _client()

    (entry,) = _mine(client).json()
    assert (entry["stopReason"], entry["stopReasonText"]) == ("idle", "no requests for a while")

    assert _report(client, 1, reason="idle", limit_hours=2).status_code == 204
    (entry,) = _mine(client).json()
    assert (entry["stopReason"], entry["stopReasonText"]) == ("idle", "no requests for 2 h")


def test_a_record_on_a_slot_the_list_dropped_is_lost(services, monkeypatch):
    """The operator lowered the slot count while slot 3 ran an app: its
    report has no route any more, so the next read marks it lost instead of
    failing the person's card and turns."""
    _launch(services, 3, ALICE, slug="dash", boot_id=BOOT_1)
    monkeypatch.setenv(SLOTS_VARIABLE, ",".join(SLOT_ADDRESSES[:2]))

    (entry,) = _mine(_client()).json()

    assert (entry["state"], entry["stopReason"]) == ("stopped", "lost")
    (app,) = asyncio.run(live_apps.person_apps(ALICE))
    assert not app.running


def test_mine_is_empty_without_a_verified_email(services):
    response = _mine(_client(), headers=headers_for(SUB_A, None))

    assert (response.status_code, response.json()) == (200, [])


# ── the stopped page ─────────────────────────────────────────────────────────


def _stopped_page(client: TestClient, slot: int, cookie: str | None):
    headers = {"Cookie": viewer_cookie(cookie)} if cookie else {}
    return client.get(f"/api/v1/live-apps/{slot}/stopped-page", headers=headers)


def test_the_launcher_reads_why_their_app_stopped(from_slot_1):
    _launch(from_slot_1, 1, ALICE, slug="dash", boot_id=BOOT_1)
    client = _client()
    assert _report(client, 1, limit_hours=2).status_code == 204

    page = _stopped_page(client, 1, "a")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "Your app <code>dash</code> stopped at" in page.text
    assert "UTC: no requests for 2 h." in page.text
    assert "Ask your agent to relaunch it." in page.text
    assert "another person's app" not in page.text


def test_the_page_names_the_app_as_its_card_does(services):
    project, path = general_project(person_scope(PARTITION_A, SUB_A))
    folder = Path(path) / ".anton" / "artifacts" / "dash-1a2b3c4d"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(
        json.dumps({"name": "Sales <dashboard>", "type": "fullstack-stateless-app"}), encoding="utf-8"
    )
    _launch(services, 1, ALICE, slug="dash-1a2b3c4d", boot_id=BOOT_1, project_id=project)
    services.slots[1].boot_id = BOOT_2

    page = _stopped_page(_client(), 1, "a")

    assert "Your app <code>Sales &lt;dashboard&gt;</code> stopped at" in page.text


def test_anyone_else_gets_a_plain_refusal(from_slot_1):
    _launch(from_slot_1, 1, ALICE, slug="dash", boot_id=BOOT_1)
    client = _client()
    assert _report(client, 1, limit_hours=2).status_code == 204
    from_slot_1.viewer_replies["empty"] = httpx.Response(200, json={})

    for cookie in ("b", None, "empty"):
        page = _stopped_page(client, 1, cookie)
        assert page.status_code == 403
        assert "You don't have access to this app." in page.text
        assert "dash" not in page.text


def test_a_former_owner_learns_the_address_serves_someone_else_now(services):
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    _launch(services, 1, BOB, slug="board", boot_id=BOOT_2)

    page = _stopped_page(_client(), 1, "a")

    assert page.status_code == 200
    assert "Your app <code>dash</code> stopped at" in page.text
    assert "its app slot restarted" in page.text
    assert "This address now serves another person's app" in page.text
    assert "board" not in page.text


def test_the_old_address_points_to_where_the_app_runs_now(services):
    """dash idled out of slot 1, B took slot 1, and A's relaunch got slot 2."""
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    _launch(services, 1, BOB, slug="board", boot_id=BOOT_2)
    _launch(services, 2, ALICE, slug="dash", boot_id=BOOT_1)

    page = _stopped_page(_client(), 1, "a")

    assert page.status_code == 200
    assert "now runs at <a href=\"http://app2.view.localhost:8080/\">" in page.text
    assert "Ask your agent to relaunch it." not in page.text


def test_a_running_app_that_isnt_answering_says_so(services):
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)

    page = _stopped_page(_client(), 1, "a")

    assert page.status_code == 200
    assert "isn't answering right now" in page.text


# ── the words for a stop ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reason", "limit_hours", "text"),
    [
        ("idle", 2, "no requests for 2 h"),
        ("idle", 2 / 60, "no requests for 2 min"),
        ("idle", None, "no requests for a while"),
        ("max_time", 8, "it reached its 8 h limit"),
        ("max_time", 1.5, "it reached its 1.5 h limit"),
        ("max_time", None, "it reached its time limit"),
        ("app_exited", None, "its backend stopped by itself"),
        ("terminated", None, "the operator or a stack restart stopped it"),
        ("replaced", None, "a launch of yours took its slot"),
        ("lost", None, "its app slot restarted"),
    ],
)
def test_each_stop_reason_has_its_sentence(reason, limit_hours, text):
    assert stop_reason_text(reason, limit_hours=limit_hours) == text


def test_a_stop_time_names_the_day_unless_its_today():
    now = datetime(2026, 10, 21, 15, 0, tzinfo=timezone.utc)

    assert stop_time_text(datetime(2026, 10, 21, 12, 4, tzinfo=timezone.utc), now=now) == "12:04 UTC"
    assert stop_time_text(datetime(2026, 10, 19, 12, 4, tzinfo=timezone.utc), now=now) == "19 Oct 12:04 UTC"
    # A database that hands back a naive value means UTC, whatever the server's zone.
    assert stop_time_text(datetime(2026, 10, 21, 12, 4), now=now) == "12:04 UTC"


# ── a launch that claimed a slot and didn't finish ───────────────────────────


def test_a_failed_relaunch_is_told_as_one(services):
    """anton claims A's running dash to relaunch it, the launch fails, and
    anton calls launch_failed: the stop is one A hasn't heard of."""
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    host = _host(ALICE)

    async def fail_a_relaunch() -> None:
        await host.claim(1)
        await host.launch_failed(1)

    asyncio.run(fail_a_relaunch())

    (record,) = _records(1)
    assert (record.state, record.stop_reason, record.stop_noticed) == ("stopped", "launch_failed", False)
    (entry,) = _mine(_client()).json()
    assert (entry["stopReason"], entry["stopReasonText"]) == (
        "launch_failed",
        "a launch of yours into its slot didn't start",
    )


def test_launch_failed_without_a_claim_or_for_another_persons_record_changes_nothing(services):
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    asyncio.run(_host(BOB).launch_failed(1))
    (record,) = _records(1)
    assert record.state == "running"

    # B's claim ends A's record as lost (the slot no longer shows it); B's
    # failed launch leaves it A's lost stop.
    host = _host(BOB)

    async def claim_then_fail() -> None:
        await host.claim(1)
        await host.launch_failed(1)

    asyncio.run(claim_then_fail())
    (record,) = _records(1)
    assert (record.stop_reason, record.stop_noticed) == ("lost", False)


def test_launch_failed_never_raises_into_antons_launch(services, monkeypatch):
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    host = _host(ALICE)
    asyncio.run(host.claim(1))

    async def broken(*_args, **_kwargs):
        raise RuntimeError("the database went away")

    monkeypatch.setattr(live_apps, "run_db", broken)

    asyncio.run(host.launch_failed(1))


def test_a_launch_whose_record_fails_is_told_as_failed(services, monkeypatch):
    """anton claims A's dash for another app of A's, the slot takes it, and
    writing its record fails: anton resets the slot and calls launch_failed,
    and dash's stop is one A hasn't heard of."""
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    host = _host(ALICE)

    def write_fails(*_args, **_kwargs):
        raise RuntimeError("the database went away")

    async def launch_whose_record_fails() -> None:
        await host.claim(1)
        with monkeypatch.context() as patched:
            patched.setattr(live_apps, "_start_record", write_fails)
            with pytest.raises(RuntimeError):
                await host.launched(_launch_model(1, "report", BOOT_2))
        await host.launch_failed(1)

    asyncio.run(launch_whose_record_fails())

    (record,) = _records(1)
    assert (record.slug, record.stop_reason, record.stop_noticed) == ("dash", "launch_failed", False)


def test_a_relaunch_stopped_once_its_record_is_written_is_told_as_failed(services, monkeypatch):
    """A Stop lands as launched's write commits: run_db finishes the write,
    then the cancel goes on, and anton resets the slot and calls
    launch_failed, which stops the relaunch's record too."""
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    host = _host(ALICE)
    real_run_db = live_apps.run_db

    async def commit_then_stop(unit, *, scope):
        await real_run_db(unit, scope=scope)
        raise asyncio.CancelledError

    async def relaunch_stopped_after_its_record() -> None:
        await host.claim(1)
        with monkeypatch.context() as patched:
            patched.setattr(live_apps, "run_db", commit_then_stop)
            with pytest.raises(asyncio.CancelledError):
                await host.launched(_launch_model(1, "dash", BOOT_2))
        await host.launch_failed(1)

    asyncio.run(relaunch_stopped_after_its_record())

    relaunch = next(record for record in _records(1) if record.boot_id == BOOT_2)
    assert (relaunch.state, relaunch.stop_reason, relaunch.stop_noticed) == ("stopped", "launch_failed", False)
    (entry,) = _mine(_client()).json()
    assert (entry["state"], entry["stopReason"]) == ("stopped", "launch_failed")


def test_a_first_launch_stopped_once_its_record_is_written_is_told_as_failed(services, monkeypatch):
    """A first launch into a free slot claims nothing, so anton calls no
    launch_failed after a Stop that lands as launched's write commits:
    launched calls it itself, and the record stops as a failed launch, not
    as a slot that restarted."""
    host = _host(ALICE)
    real_run_db = live_apps.run_db

    async def commit_then_stop(unit, *, scope):
        result = await real_run_db(unit, scope=scope)
        if unit.func is live_apps._start_record:
            raise asyncio.CancelledError
        return result

    async def first_launch_stopped_after_its_record() -> None:
        with monkeypatch.context() as patched:
            patched.setattr(live_apps, "run_db", commit_then_stop)
            with pytest.raises(asyncio.CancelledError):
                await host.launched(_launch_model(1, "dash", BOOT_1))

    asyncio.run(first_launch_stopped_after_its_record())

    (record,) = _records(1)
    assert (record.state, record.stop_reason, record.stop_noticed) == ("stopped", "launch_failed", False)


def test_the_access_checks_build_the_tls_context_once(monkeypatch):
    """The edge asks the access route on every request to an app, and loading
    the CA bundle blocks the event loop, so every check's client shares one
    context."""
    built = []
    real = httpx.create_ssl_context

    def counted(**kwargs):
        built.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(httpx, "create_ssl_context", counted)
    live_apps._ssl_context.cache_clear()
    try:
        clients = [live_apps._http_client() for _ in range(3)]
        for client in clients:
            asyncio.run(client.aclose())
    finally:
        live_apps._ssl_context.cache_clear()
    assert built == [{"trust_env": True}]


def test_a_launch_into_a_freed_slot_dates_the_earlier_apps_loss_by_the_slots_boot(services):
    """A's app ended unreported when slot 1 restarted; B's launch there later
    stamps A's lost stop at that restart, not at B's launch."""
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    (record,) = _records(1)
    booted = record.started_at.replace(tzinfo=timezone.utc) + timedelta(microseconds=1)
    services.slots[1] = live_apps_fake_slot(boot_id=BOOT_2, booted_at=booted)

    _launch_into_a_free_slot(services, 1, BOB, slug="board", boot_id=BOOT_2)

    alice, _bob = _records(1)
    assert alice.stop_reason == "lost"
    assert alice.stopped_at.replace(tzinfo=timezone.utc) == booted


def live_apps_fake_slot(*, boot_id: str, booted_at: datetime):
    from tests._live_apps import FakeSlot

    return FakeSlot(boot_id=boot_id, booted_at=booted_at.isoformat().replace("+00:00", "Z"), state="empty")


def test_a_viewer_sub_in_another_form_is_the_same_person(services):
    """The identity middleware keeps X-User-Id as a UUID's canonical text."""
    services.sessions["a"] = ViewerUser(SUB_A.replace("-", "").upper(), EMAIL_A)
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)

    assert _access(_client(), 1, "a").status_code == 200


def test_no_cookie_is_signed_out_without_asking_viewer_login(services, monkeypatch):
    def no_client():
        raise AssertionError("no cookie asks nobody")

    monkeypatch.setattr(live_apps, "_http_client", no_client)
    client = _client()

    assert _access(client, 1, None).status_code == 401
    assert _stopped_page(client, 1, None).status_code == 403


def test_mine_never_shows_a_relaunch_as_a_stop(services):
    """A relaunched row is only ever the latest when its successor is gone:
    /mine then shows no stop reason rather than failing."""
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        record = raw.exec(select(LiveAppLaunchRecord)).one()
        record.state, record.stop_reason = "stopped", "relaunched"
        raw.add(record)
        raw.commit()

    response = _mine(_client())

    assert response.status_code == 200
    (entry,) = response.json()
    assert (entry["state"], entry["stopReason"], entry["stopReasonText"]) == ("stopped", None, None)


def test_the_access_client_reads_the_deployments_ca_variables(monkeypatch):
    """No proxy variable, but SSL_CERT_FILE still counts, for a viewer-login
    on a private CA."""
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/private-ca.pem")

    with pytest.raises(FileNotFoundError):
        live_apps._http_client()
