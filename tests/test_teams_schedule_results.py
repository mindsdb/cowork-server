"""A scheduled run's result posted to its owner's Teams chat: the contacts the recording proxy keeps,
the delivery after a completed run, and how each delivery is recorded on the run."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from cowork.api.v1.endpoints.connectors import teams as routes
from cowork.api.v1.endpoints.connectors.teams import install_teams_routes
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.schedule import Schedule, ScheduleRun
from cowork.models.teams_contact import TeamsContact
from cowork.services.connectors.teams import bot as teams
from cowork.services.connectors.teams import delivery
from cowork.services.connectors.teams.bot import SIGNATURE_HEADER, TIMESTAMP_HEADER, TeamsBot, TeamsSettings, sign
from cowork.services.projects import GENERAL_PROJECT_ID
from tests.test_push_scheduler import _fresh_scheduler, pushes, schedule_id, turns  # noqa: F401

KEY = "internal-key-value"
SERVICE_URL = "https://smba.trafficmanager.net/amer/tenant-1/"
CONTACTS = "/api/v1/connectors/teams/contacts"


def _settings(**overrides) -> TeamsSettings:
    values = {"bot_app_id": "app-1", "bot_app_secret": "secret", "tenant_id": "tenant-1", "internal_key": KEY,
              "web_ui_url": "http://cowork.localhost:8080/"}
    values.update(overrides)
    return TeamsSettings(**values)


class StandInTeams:
    """TeamsClient's outbound calls, recorded instead of sent."""

    def __init__(self) -> None:
        self.sent: list[tuple] = []
        self.error: Exception | None = None

    async def message_person(self, service_url, tenant_id, aad_object_id, texts):
        self.sent.append((service_url, tenant_id, aad_object_id, list(texts)))
        if self.error:
            raise self.error
        return [f"m-{n}" for n, _ in enumerate(texts)]


@pytest.fixture
def stand_in(monkeypatch):
    monkeypatch.setattr(routes, "teams_available", lambda: True)
    found = StandInTeams()
    bot = TeamsBot(_settings(), make_client=lambda: found)
    teams.set_running_bot(bot)
    yield found
    teams.set_running_bot(None)


@pytest.fixture
def client(monkeypatch, stand_in):
    app = FastAPI()
    exempt: set[str] = set()
    assert install_teams_routes(app, exempt, _settings())
    app.state.teams_bot = teams.running_bot()
    assert CONTACTS in exempt
    return TestClient(app)


def _signed(method: str, target: str, body: bytes = b"") -> dict:
    stamp = str(int(time.time()))
    return {TIMESTAMP_HEADER: stamp, SIGNATURE_HEADER: sign(KEY.encode(), method, target, stamp, body),
            "Content-Type": "application/json"}


def _engine():
    return get_engine(get_app_settings().database.uri)


def _contact(user_id: str) -> TeamsContact | None:
    with Session(_engine()) as session:
        return session.get(TeamsContact, user_id)


# The contacts the proxy keeps.

def test_the_proxy_saves_and_forgets_a_persons_teams_contact(client):
    user = str(uuid4())
    body = json.dumps({"user_id": user, "tenant_id": "tenant-1", "aad_object_id": "aad-1",
                       "service_url": SERVICE_URL}).encode()
    saved = client.put(CONTACTS, content=body, headers=_signed("PUT", CONTACTS, body))
    moved = json.dumps({"user_id": user, "tenant_id": "tenant-1", "aad_object_id": "aad-2",
                        "service_url": SERVICE_URL}).encode()
    client.put(CONTACTS, content=moved, headers=_signed("PUT", CONTACTS, moved))

    assert saved.status_code == 204
    assert _contact(user).aad_object_id == "aad-2"
    target = f"{CONTACTS}?user_id={user}"
    assert client.delete(target, headers=_signed("DELETE", target)).status_code == 204
    assert _contact(user) is None


def test_re_pairing_a_microsoft_id_to_someone_else_ends_the_earlier_owners_results_there(client):
    earlier, later = str(uuid4()), str(uuid4())
    for user in (earlier, later):
        body = json.dumps({"user_id": user, "tenant_id": "tenant-1", "aad_object_id": "aad-shared",
                           "service_url": SERVICE_URL}).encode()
        assert client.put(CONTACTS, content=body, headers=_signed("PUT", CONTACTS, body)).status_code == 204

    assert _contact(earlier) is None
    assert _contact(later).aad_object_id == "aad-shared"


def test_a_contact_needs_the_proxys_signature_and_an_allowed_teams_host(client):
    user = str(uuid4())
    body = json.dumps({"user_id": user, "tenant_id": "t", "aad_object_id": "a", "service_url": SERVICE_URL}).encode()
    elsewhere = json.dumps({"user_id": user, "tenant_id": "t", "aad_object_id": "a",
                            "service_url": "https://teams.example/emea/"}).encode()

    unsigned = client.put(CONTACTS, content=body, headers={"Content-Type": "application/json"})
    other_host = client.put(CONTACTS, content=elsewhere, headers=_signed("PUT", CONTACTS, elsewhere))

    assert (unsigned.status_code, other_host.status_code) == (401, 403)
    assert _contact(user) is None


# Delivery.

def _run_with_answer(answer: str | None, *, then: tuple = ()) -> tuple:
    with Session(_engine()) as session:
        schedule = Schedule(title="Weekly sales", prompt="p", cadence="daily", model="m",
                            next_run_at=datetime.now(timezone.utc), project_id=GENERAL_PROJECT_ID, deliver_to_teams=True)
        conversation = Conversation(topic="Weekly sales", project_id=GENERAL_PROJECT_ID)
        session.add(schedule)
        session.add(conversation)
        session.commit()
        run = ScheduleRun(schedule_id=schedule.id, started_at=datetime.now(timezone.utc), status="success",
                          conversation_id=conversation.id)
        session.add(run)
        session.add(Message(conversation_id=conversation.id, role="user", content="p", seq=1))
        if answer is not None:
            session.add(Message(conversation_id=conversation.id, role="assistant", content=answer, seq=2))
        for seq, (role, content) in enumerate(then, start=3):
            session.add(Message(conversation_id=conversation.id, role=role, content=content, seq=seq))
        session.commit()
        return run.id, conversation.id


def _delivery(run_id) -> str | None:
    with Session(_engine()) as session:
        return session.get(ScheduleRun, run_id).teams_delivery


def _contact_for(user: str) -> None:
    with Session(_engine()) as session:
        session.add(TeamsContact(user_id=user, tenant_id="tenant-1", aad_object_id="aad-1", service_url=SERVICE_URL,
                                 updated_at=datetime.now(timezone.utc)))
        session.commit()


async def test_a_completed_run_posts_its_answer_and_a_link_to_the_owners_chat_once(stand_in):
    user = str(uuid4())
    _contact_for(user)
    run_id, conversation_id = _run_with_answer("Sales rose 4% in Europe.")

    await delivery.deliver_run_result(run_id=run_id, user_id=user, conversation_id=conversation_id, title="Weekly sales")

    [(service_url, tenant, aad, texts)] = stand_in.sent
    assert (service_url, tenant, aad) == (SERVICE_URL, "tenant-1", "aad-1")
    assert texts == [f"**Weekly sales**\n\nSales rose 4% in Europe.\n\n"
                     f"[Open in MindsHub](http://cowork.localhost:8080/c/{conversation_id})"]
    assert _delivery(run_id) == delivery.DELIVERED


async def test_a_long_answer_goes_in_parts(stand_in, monkeypatch):
    monkeypatch.setattr(delivery, "MAX_PART_UNITS", 200)
    user = str(uuid4())
    _contact_for(user)
    answer = "\n\n".join(f"Paragraph {n}: " + " ".join(["sales rose"] * 8) for n in range(6))
    run_id, conversation_id = _run_with_answer(answer)

    await delivery.deliver_run_result(run_id=run_id, user_id=user, conversation_id=conversation_id, title="Weekly sales")

    [(_, _, _, texts)] = stand_in.sent
    assert len(texts) > 1 and all(len(text) <= 200 for text in texts)
    assert answer in "\n\n".join(texts)


@pytest.mark.parametrize("case", ["no-contact", "teams-refuses", "no-owner"])
async def test_a_run_that_cant_be_posted_is_marked_failed_and_never_resent(stand_in, case):
    user = str(uuid4())
    if case != "no-contact":
        _contact_for(user)
    if case == "teams-refuses":
        stand_in.error = RuntimeError("Forbidden: the bot isn't installed for the user")
    run_id, conversation_id = _run_with_answer("Sales rose.")

    await delivery.deliver_run_result(run_id=run_id, user_id=None if case == "no-owner" else user,
                                      conversation_id=conversation_id, title="Weekly sales")

    assert _delivery(run_id) == delivery.FAILED
    assert len(stand_in.sent) == (1 if case == "teams-refuses" else 0)


async def test_a_failed_run_posts_a_short_notice_with_a_link(stand_in):
    user = str(uuid4())
    _contact_for(user)
    run_id, conversation_id = _run_with_answer("Half an answer")

    await delivery.deliver_run_result(run_id=run_id, user_id=user, conversation_id=conversation_id,
                                      title="Weekly sales", succeeded=False)

    [(_, _, _, texts)] = stand_in.sent
    assert texts == [f"**Weekly sales**\n\nThis scheduled run didn't complete.\n\n"
                     f"[Open in MindsHub](http://cowork.localhost:8080/c/{conversation_id}) to see what happened."]
    assert _delivery(run_id) == delivery.DELIVERED


async def test_a_contact_on_a_host_no_longer_allowed_gets_nothing(stand_in):
    user = str(uuid4())
    with Session(_engine()) as session:
        session.add(TeamsContact(user_id=user, tenant_id="tenant-1", aad_object_id="aad-1",
                                 service_url="https://teams.example/emea/", updated_at=datetime.now(timezone.utc)))
        session.commit()
    run_id, conversation_id = _run_with_answer("Sales rose.")

    await delivery.deliver_run_result(run_id=run_id, user_id=user, conversation_id=conversation_id, title="Weekly sales")

    assert stand_in.sent == []
    assert _delivery(run_id) == delivery.FAILED


def _teams_error(code: int):
    import httpx

    request = httpx.Request("POST", SERVICE_URL)
    return httpx.HTTPStatusError("failed", request=request, response=httpx.Response(code, request=request))


def _partly_sent(posted: int, cause: Exception) -> Exception:
    from cowork.services.connectors.teams.client import TeamsPartlySent

    try:
        raise TeamsPartlySent(posted) from cause
    except TeamsPartlySent as error:
        return error


@pytest.mark.parametrize(("error", "state"), [
    (lambda: _partly_sent(0, _teams_error(503)), delivery.UNKNOWN),
    (lambda: _partly_sent(0, _teams_error(408)), delivery.UNKNOWN),
    (lambda: _partly_sent(0, __import__("httpx").ReadTimeout("timed out")), delivery.UNKNOWN),
    (lambda: _partly_sent(1, _teams_error(403)), delivery.UNKNOWN),
    (lambda: _partly_sent(0, _teams_error(403)), delivery.FAILED),
    (lambda: _partly_sent(0, __import__("httpx").ConnectError("refused")), delivery.FAILED),
    (lambda: _teams_error(503), delivery.FAILED),
], ids=["5xx", "408", "timeout", "later-part", "refused", "unreachable", "opening-the-chat"])
async def test_a_post_that_may_have_reached_teams_is_unknown_not_failed(stand_in, error, state):
    user = str(uuid4())
    _contact_for(user)
    stand_in.error = error()
    run_id, conversation_id = _run_with_answer("Sales rose.")

    await delivery.deliver_run_result(run_id=run_id, user_id=user, conversation_id=conversation_id, title="Weekly sales")

    assert _delivery(run_id) == state


async def test_a_failed_post_logs_the_run_and_teams_status(stand_in, caplog):
    user = str(uuid4())
    _contact_for(user)
    stand_in.error = _partly_sent(0, _teams_error(429))
    run_id, conversation_id = _run_with_answer("Sales rose.")

    with caplog.at_level("WARNING"):
        await delivery.deliver_run_result(run_id=run_id, user_id=user, conversation_id=conversation_id,
                                          title="Weekly sales")

    [record] = [r for r in caplog.records if getattr(r, "event", None) == "teams_delivery_failed"]
    assert (record.conversation_id, record.status_code) == (str(conversation_id), 429)


async def test_recording_the_delivery_tries_again_when_the_database_refuses(stand_in, monkeypatch):
    user = str(uuid4())
    _contact_for(user)
    run_id, conversation_id = _run_with_answer("Sales rose.")
    monkeypatch.setattr(delivery, "_RECORD_PAUSE_SECONDS", 0)
    real, refusals = delivery._set_delivery, iter([True])

    def flaky(run, state):
        if state == delivery.DELIVERED and next(refusals, False):
            raise RuntimeError("pool timeout")
        real(run, state)

    monkeypatch.setattr(delivery, "_set_delivery", flaky)

    await delivery.deliver_run_result(run_id=run_id, user_id=user, conversation_id=conversation_id, title="Weekly sales")

    assert _delivery(run_id) == delivery.DELIVERED


async def test_a_delivery_the_database_keeps_refusing_is_logged_not_raised(stand_in, monkeypatch, caplog):
    user = str(uuid4())
    _contact_for(user)
    run_id, conversation_id = _run_with_answer("Sales rose.")
    monkeypatch.setattr(delivery, "_RECORD_PAUSE_SECONDS", 0)
    real = delivery._set_delivery

    def refuses(run, state):
        if state == delivery.DELIVERED:
            raise RuntimeError("database gone")
        real(run, state)

    monkeypatch.setattr(delivery, "_set_delivery", refuses)

    with caplog.at_level("WARNING"):
        await delivery.deliver_run_result(run_id=run_id, user_id=user, conversation_id=conversation_id,
                                          title="Weekly sales")

    assert len(stand_in.sent) == 1
    assert any(getattr(r, "event", None) == "teams_delivery_unrecorded" for r in caplog.records)


def test_a_delivery_a_stop_cut_off_is_unknown_not_resent():
    run_id, _ = _run_with_answer("Sales rose.")
    delivery._set_delivery(run_id, delivery.SENDING)

    assert delivery.forget_interrupted_deliveries() >= 1
    assert _delivery(run_id) == delivery.UNKNOWN


# The scheduler.

@pytest.fixture
def posted(monkeypatch, stand_in):
    """The results the scheduler starts posting to Teams, by run and owner."""
    from cowork.services.push import dispatch

    started: list[dict] = []

    async def deliver(**kwargs):
        started.append(kwargs)

    monkeypatch.setattr(delivery, "deliver_run_result", deliver)
    monkeypatch.setattr(dispatch, "spawn", _run_now)
    return started


def _run_now(coro):
    try:
        coro.send(None)
    except StopIteration:
        pass


def _ask_for_teams(schedule_id, on: bool) -> None:
    with Session(_engine()) as session:
        schedule = session.get(Schedule, schedule_id)
        schedule.deliver_to_teams = on
        session.add(schedule)
        session.commit()


@pytest.mark.parametrize(("on", "reason", "count"), [(True, "completed", 1), (True, "error", 1), (False, "completed", 0)])
async def test_the_scheduler_posts_a_finished_run_only_for_a_schedule_that_asks(
    schedule_id, turns, pushes, posted, on, reason, count
):
    from test_push_scheduler import OWNER
    from tests.test_schedule_run_outcomes import _poll, _run

    _ask_for_teams(schedule_id, on)
    turns.reason = reason
    (slot,) = _poll(schedule_id)
    await _run(slot)

    assert len(posted) == count
    if count:
        assert posted[0]["user_id"] == OWNER and posted[0]["conversation_id"] is not None
        assert posted[0]["succeeded"] is (reason == "completed")


async def test_without_a_teams_bot_a_run_posts_nothing_and_records_no_delivery(schedule_id, turns, pushes, posted):
    from cowork import scheduler
    from tests.test_schedule_run_outcomes import _poll, _run

    teams.set_running_bot(None)
    _ask_for_teams(schedule_id, True)
    (slot,) = _poll(schedule_id)
    await _run(slot)

    assert posted == []
    with Session(_engine()) as session:
        runs = session.exec(select(ScheduleRun).where(ScheduleRun.schedule_id == schedule_id)).all()
    assert [run.teams_delivery for run in runs] == [None]


async def test_a_trial_run_posts_nothing(schedule_id, turns, pushes, posted):
    from cowork import scheduler

    _ask_for_teams(schedule_id, True)
    await scheduler.execute_schedule(schedule_id, is_manual=True, trial=True)
    assert posted == []


async def test_a_schedule_keeps_the_setting_and_its_runs_show_the_delivery():
    import test_turn_pool_connections as turn_pool

    from cowork.db.scoped import SYSTEM_SCOPE
    from cowork.db.units import unit_session
    from cowork.services.schedules import ScheduleService

    created_id = None
    try:
        async with turn_pool._client() as client:
            created = await client.post("/api/v1/schedules/", json={
                "title": "Weekly sales", "prompt": "p", "cadence": "daily",
                "nextRunAt": "2030-01-08T09:00:00Z", "deliverToTeams": True,
            })
            assert created.status_code in (200, 201), created.text
            created_id = created.json()["id"]
            assert (await client.get(f"/api/v1/schedules/{created_id}")).json()["deliverToTeams"] is True
            edited = await client.patch(f"/api/v1/schedules/{created_id}", json={"deliverToTeams": False})
            assert edited.json()["deliverToTeams"] is False
            plain = await client.post("/api/v1/schedules/", json={
                "title": "Plain", "prompt": "p", "cadence": "daily", "nextRunAt": "2030-01-08T09:00:00Z"})
            assert plain.json()["deliverToTeams"] is False
            with unit_session(scope=SYSTEM_SCOPE) as session:
                ScheduleService(session).delete_schedule(UUID(plain.json()["id"]))
    finally:
        if created_id:
            with unit_session(scope=SYSTEM_SCOPE) as session:
                ScheduleService(session).delete_schedule(UUID(created_id))

    run_id, _ = _run_with_answer("x")
    delivery._set_delivery(run_id, delivery.DELIVERED)
    from cowork.schemas.schedules import ScheduleRunResponse
    with Session(_engine()) as session:
        dumped = ScheduleRunResponse.model_validate(session.get(ScheduleRun, run_id)).model_dump(by_alias=True)
    assert dumped["teamsDelivery"] == "delivered"


async def test_the_answer_is_the_last_assistant_text_even_after_later_steps(stand_in):
    user = str(uuid4())
    _contact_for(user)
    steps = (("assistant", [{"type": "tool_call", "name": "x"}]), ("thought.tool_call.result", "done"), ("assistant", ""))
    run_id, conversation_id = _run_with_answer("Sales rose 4% in Europe.", then=steps)

    await delivery.deliver_run_result(run_id=run_id, user_id=user, conversation_id=conversation_id, title="Weekly sales")

    [(_, _, _, texts)] = stand_in.sent
    assert "Sales rose 4% in Europe." in texts[0]


def test_a_long_line_of_wide_characters_is_cut_in_one_pass_and_within_the_limit(monkeypatch):
    line = "\U0001F4C8" * 60_000
    counted = []
    units = delivery._units
    monkeypatch.setattr(delivery, "_units", lambda text: counted.append(1) or units(text))

    parts = delivery.split_text(line)

    # Re-measuring the line once per character dropped would take tens of thousands of calls.
    assert len(counted) < 50
    assert "".join(parts) == line
    assert all(delivery._units(part) <= delivery.MAX_PART_UNITS for part in parts)
