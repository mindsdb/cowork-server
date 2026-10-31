"""Full-slot refusals, durable advisory notices, and person-scoped delivery."""
from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

import cowork.server as cowork_server
from cowork.common.datetime_utils import ensure_utc
from cowork.common.settings.app_settings import get_app_settings
from cowork.db import session as db_session
from cowork.db.session import get_engine
from cowork.models.live_app import LiveAppAvailabilityWait
from cowork.services import live_app_admission as admission
from cowork.services import live_apps
from tests._live_apps import (
    A, B, ALICE, BOB, BEARER_TOKEN, LAUNCH_TOKEN, MODULE,
    FakeServices, answer_over_http, clear_live_app_rows, turn_on_live_apps,
)


@pytest.fixture
def services(personal_deployment, monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "home" / "projects"))
    monkeypatch.setenv("COWORK_AUTH_TOKEN", BEARER_TOKEN)
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    turn_on_live_apps(monkeypatch)
    fake = FakeServices()
    for slot in fake.slots:
        fake.run(slot, boot_id=str(slot) * 32, artifact=f"app-{slot}", subject="someone-else", email="else@example.com")
    answer_over_http(monkeypatch, fake)
    yield fake
    clear_live_app_rows()
    get_app_settings.cache_clear()


def _client():
    return TestClient(cowork_server.create_app())


def _host(person):
    return live_apps.CoworkLiveAppHost(person, project_id=uuid4(), launch_token=LAUNCH_TOKEN)


async def _refuse(person):
    host = _host(person)
    async with host.launch_lock():
        await host.slots_full()


def _rows():
    with Session(get_engine(get_app_settings().database.uri)) as session:
        return list(session.exec(select(LiveAppAvailabilityWait)).all())


def _available(client, headers=A):
    response = client.get("/api/v1/live-apps/availability", headers=headers)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    return response.json()


def _free(services):
    services.slots[1].state = "empty"
    services.slots[1].boot_id = "f" * 32


def test_never_launched_person_gets_notice_without_sending_another_message(services):
    client = _client()
    assert client.get("/api/v1/live-apps/mine", headers=A).json() == []
    initial = _available(client)
    assert initial["waiting"] is False and initial["notice"] is None
    assert services.requests == []  # No slot probes without any refused person.
    asyncio.run(_refuse(ALICE))
    waiting = _available(client)
    assert waiting["waiting"] is True and waiting["notice"] is None

    _free(services)
    notice = _available(client)
    assert notice["waiting"] is False
    assert notice["notice"]["message"] == live_apps.AvailabilityNotice(
        id=UUID(notice["notice"]["id"]), notified_at=ensure_utc(_rows()[0].modified_at)
    ).message
    assert notice["scope"] == initial["scope"]
    assert ALICE.user_id not in notice["scope"]
    # A fresh app/client reads the same persisted notice, without generating
    # another event or requiring a prior launch row.
    assert _available(_client()) == notice
    assert _available(client, B)["notice"] is None
    assert _available(client, B)["scope"] != notice["scope"]
    assert client.get("/api/v1/live-apps/mine", headers=A).json() == []


def test_refusals_dedupe_and_old_ack_cannot_clear_new_generation(services):
    asyncio.run(_refuse(ALICE))
    generation = _rows()[0].generation
    asyncio.run(_refuse(ALICE))
    assert len(_rows()) == 1 and _rows()[0].generation == generation
    _free(services)
    client = _client()
    notice_id = _available(client)["notice"]["id"]
    assert UUID(notice_id) == generation
    ack = f"/api/v1/live-apps/availability/{notice_id}/ack"
    assert client.post(ack, headers=B).status_code == 204
    assert _available(client)["notice"]["id"] == notice_id

    services.run(1, boot_id="a" * 32, artifact="taken", subject="other", email="other@example.com")
    asyncio.run(_refuse(ALICE))
    assert _rows()[0].generation != generation
    assert client.post(ack, headers=A).status_code == 204
    assert _available(client)["waiting"] is True
    _free(services)
    second = _available(client)["notice"]["id"]
    assert second != notice_id
    assert client.post(ack, headers=A).status_code == 204
    assert _available(client)["notice"]["id"] == second
    assert client.post(f"/api/v1/live-apps/availability/{second}/ack", headers=A).status_code == 204
    assert _available(client)["notice"] is None
    assert _rows() == []


def test_every_waiter_hears_even_if_another_person_consumes_capacity_before_their_poll(services):
    asyncio.run(_refuse(ALICE))
    asyncio.run(_refuse(BOB))
    _free(services)

    async def alice_launches():
        host = _host(ALICE)
        async with host.launch_lock():
            # Anton observes empty in the same critical section as admission.
            await host.slots_available()
            services.run(1, boot_id="f" * 32, artifact="alice", subject=ALICE.user_id, email=ALICE.email)
            await host.launched(sys.modules[MODULE].LiveAppLaunch(
                slot=1, slug="alice", artifact_dir="/projects/a/.anton/artifacts/alice",
                address="http://app1.view.localhost:8080/", boot_id="f" * 32,
            ))

    asyncio.run(alice_launches())
    client = _client()
    assert _available(client, A)["notice"] is None
    bob_notice = _available(client, B)["notice"]
    assert bob_notice is not None
    assert len(_rows()) == 1 and _rows()[0].user_id == BOB.user_id
    # Successful launch cleared only Alice, and Bob's event survives now-full
    # slots. Notices are an event, not a reservation or a queue.
    assert _available(_client(), B)["notice"] == bob_notice


@pytest.mark.parametrize("state", ["exited", "unreachable"])
def test_stopped_or_unreachable_is_not_a_ready_empty_slot(services, state):
    asyncio.run(_refuse(ALICE))
    if state == "exited":
        services.slots[1].state = "exited"
        services.slots[1].stop_reason = "idle"
    else:
        services.slots[1].fails_with = 503
    result = _available(_client())
    assert result["waiting"] is True and result["notice"] is None


def test_poll_broadcasts_to_other_waiters_and_ack_needs_identity(services):
    asyncio.run(_refuse(ALICE))
    asyncio.run(_refuse(BOB))
    _free(services)
    client = _client()
    assert _available(client, A)["notice"] is not None
    services.run(1, boot_id="f" * 32, artifact="taken", subject="other", email="other@example.com")
    assert _available(client, B)["notice"] is not None
    assert client.get("/api/v1/live-apps/availability").status_code == 401
    # The deployment bearer alone names no authenticated person.
    assert client.get("/api/v1/live-apps/availability", headers={
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }).status_code == 401
    assert client.post(f"/api/v1/live-apps/availability/{uuid4()}/ack").status_code == 401


def test_notice_message_names_the_time_the_slot_was_seen_free():
    seen = datetime.now(timezone.utc).replace(hour=14, minute=5)
    assert live_apps.AvailabilityNotice(id=uuid4(), notified_at=seen).message == (
        "A live app slot became available at 14:05 UTC. You can try launching again. It is not reserved."
    )
    older = live_apps.AvailabilityNotice(id=uuid4(), notified_at=seen - timedelta(days=1)).message
    assert f"available at {(seen - timedelta(days=1)).day} " in older


def test_notice_time_is_when_the_slot_was_seen_free_not_the_refusal(services):
    asyncio.run(_refuse(ALICE))
    refused_at = datetime(2026, 1, 2, 9, 30, tzinfo=timezone.utc)
    with Session(get_engine(get_app_settings().database.uri)) as session:
        row = session.exec(select(LiveAppAvailabilityWait)).one()
        row.created_at = row.modified_at = refused_at
        session.add(row)
        session.commit()
    _free(services)
    before = datetime.now(timezone.utc)
    message = _available(_client())["notice"]["message"]
    notified_at = ensure_utc(_rows()[0].modified_at)
    assert before - timedelta(minutes=1) <= notified_at <= datetime.now(timezone.utc) + timedelta(minutes=1)
    assert "2 Jan" not in message
    assert message == live_apps.AvailabilityNotice(id=_rows()[0].generation, notified_at=notified_at).message


def test_another_persons_poll_neither_reconciles_nor_takes_admission(services, monkeypatch):
    asyncio.run(_refuse(ALICE))
    _free(services)

    def no_admission():
        pytest.fail("a person who is not waiting must not take admission")

    monkeypatch.setattr(live_apps, "try_live_app_launch_lock", no_admission)
    monkeypatch.setattr(live_apps, "live_app_launch_lock", no_admission)
    client = _client()
    result = _available(client, B)
    assert result["waiting"] is False and result["notice"] is None
    assert services.requests == []
    assert _rows()[0].notified is False


def test_busy_admission_returns_the_callers_state_without_reconciling_or_waiting(services):
    asyncio.run(_refuse(ALICE))
    _free(services)
    holder = admission._LaunchLock(db_session.get_engine(db_session.settings.database.uri))
    holder.open()
    try:
        assert holder.try_acquire() is True
        client = _client()
        started = time.monotonic()
        result = _available(client)
        # A blocking acquisition would wait out LAUNCH_LOCK_TIMEOUT_SECONDS.
        assert time.monotonic() - started < admission.LAUNCH_LOCK_TIMEOUT_SECONDS / 3
        assert result["waiting"] is True and result["notice"] is None
        assert services.requests == []
        assert _rows()[0].notified is False
    finally:
        holder.close()
    # The next poll, with admission free, reconciles.
    assert _available(client)["notice"] is not None
