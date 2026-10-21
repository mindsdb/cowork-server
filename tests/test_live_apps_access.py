"""Who opens a live app: only the person who launched the app its slot runs
now, by the server's own launch record and the slot's own status, never by
anything the request says. Then how a stop is recorded and explained: the
slot's report, the card's state, and the page at the old address."""

from __future__ import annotations

import asyncio
import sys
from uuid import uuid4

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
from cowork.services.live_apps import CoworkLiveAppHost, LiveAppPerson
from tests._live_apps import (
    ALICE,
    BEARER_TOKEN,
    BOB,
    EMAIL_A,
    EMAIL_B,
    LAUNCH_TOKEN,
    MODULE,
    SUB_A,
    A,
    B,
    FakeServices,
    answer_over_http,
    clear_live_app_rows,
    headers_for,
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
    fake = FakeServices(sessions={"a": EMAIL_A, "b": EMAIL_B})
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


def _host(who: LiveAppPerson, project_id=None) -> CoworkLiveAppHost:
    return CoworkLiveAppHost(who, project_id=project_id, launch_token=LAUNCH_TOKEN)


def _launch(services: FakeServices, slot: int, who: LiveAppPerson, *, slug: str, boot_id: str, project_id=None) -> None:
    """``who`` launches ``slug`` into ``slot`` as anton does: claim, the slot
    takes it, launched."""
    host = _host(who, project_id)

    async def launch() -> None:
        await host.claim(slot)
        await host.launched(_launch_model(slot, slug, boot_id))

    asyncio.run(launch())
    services.run(slot, boot_id=boot_id, artifact=slug, email=who.email)


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

    services.run(1, boot_id=BOOT_1, artifact="dash", email=EMAIL_B)
    assert _access(client, 1, "a").status_code == 403

    services.slots[1].state = "exited"
    services.slots[1].launcher_email = EMAIL_A
    assert _access(client, 1, "a").status_code == 403


def test_the_email_compares_without_case(services):
    services.sessions["a"] = "Alice@Example.COM"
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)

    assert _access(_client(), 1, "a").status_code == 200


def test_viewer_login_or_the_slot_failing_answers_503(services):
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    client = _client()

    services.slots[1].fails_with = 500
    assert _access(client, 1, "a").status_code == 503

    services.slots[1].fails_with = None
    services.viewer_fails_with = 502
    assert _access(client, 1, "a").status_code == 503


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
    refuses never calls launched, so even the slot's previous app is shut."""
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    asyncio.run(_host(ALICE).claim(1))

    assert _access(_client(), 1, "a").status_code == 403
    (record,) = _records(1)
    assert (record.state, record.stop_reason) == ("stopped", "replaced")


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


def test_a_slot_taken_from_another_person_was_already_lost(services):
    """anton takes a slot that ran someone else's app only once it is empty."""
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


def test_a_report_from_another_address_is_refused(from_slot_1):
    _launch(from_slot_1, 2, ALICE, slug="dash", boot_id=BOOT_1)

    assert _report(_client(), 2).status_code == 403
    (record,) = _records(2)
    assert record.state == "running"


def test_a_report_through_a_proxy_is_refused(from_slot_1):
    """uvicorn rewrites the client address from X-Forwarded-For for a
    loopback peer, which agent code in the cowork container is."""
    _launch(from_slot_1, 1, ALICE, slug="dash", boot_id=BOOT_1)

    assert _report(_client(), 1, headers={"X-Forwarded-For": "172.30.16.11"}).status_code == 403
    (record,) = _records(1)
    assert record.state == "running"


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

    by_slug = {entry["slug"]: entry for entry in mine}
    assert by_slug["report"]["state"] == "running"
    assert by_slug["report"]["address"] == "http://app2.view.localhost:8080/"
    assert by_slug["report"]["stopReason"] is None
    assert by_slug["dash"]["state"] == "stopped"
    assert by_slug["dash"]["stopReason"] == "idle"
    assert by_slug["dash"]["stopReasonText"] == "no requests for 2 h"
    assert by_slug["dash"]["projectId"] == str(project)
    assert _mine(client, headers=B).json() == []


def test_an_app_its_slot_no_longer_runs_is_lost(services):
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)
    services.slots[1].boot_id = BOOT_2
    services.slots[1].state = "empty"

    (entry,) = _mine(_client()).json()

    assert (entry["state"], entry["stopReason"]) == ("stopped", "lost")
    assert entry["stopReasonText"] == "its app slot restarted"
    (record,) = _records(1)
    assert record.stop_reason == "lost"


def test_mine_needs_a_verified_email(services):
    assert _mine(_client(), headers=headers_for(SUB_A, None)).status_code == 401


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


def test_anyone_else_gets_a_plain_refusal(from_slot_1):
    _launch(from_slot_1, 1, ALICE, slug="dash", boot_id=BOOT_1)
    client = _client()
    assert _report(client, 1, limit_hours=2).status_code == 204

    for cookie in ("b", None):
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


def test_a_running_app_that_isnt_answering_says_so(services):
    _launch(services, 1, ALICE, slug="dash", boot_id=BOOT_1)

    page = _stopped_page(_client(), 1, "a")

    assert page.status_code == 200
    assert "isn't answering right now" in page.text
