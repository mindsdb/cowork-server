"""The Teams bot's internal routes and credentials check (cowork/channels/teams.py).

A stand-in TeamsClient takes the place of Microsoft, so these run on every
platform; TeamsClient itself is tested against stand-in Microsoft endpoints in
test_teams_client.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.channels import teams
from cowork.channels.teams import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    TeamsBot,
    TeamsSettings,
    install_teams_routes,
    sign,
)
from cowork.channels.teams_client import TeamsAuthError

KEY = "internal-key-value"
SECRET = "bot-secret-value"
SERVICE_URL = "https://smba.trafficmanager.net/amer/tenant-1/"
DELIVER = "/api/v1/channels/teams/deliver"
MEMBER = "/api/v1/channels/teams/member"


def _settings(**overrides) -> TeamsSettings:
    values = {
        "bot_app_id": "app-1",
        "bot_app_secret": SECRET,
        "tenant_id": "tenant-1",
        "internal_key": KEY,
    }
    values.update(overrides)
    return TeamsSettings(**values)


class StandInTeams:
    """TeamsClient's outbound calls, recorded instead of sent."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple] = []
        self.error = error
        self.closed = False

    async def bot_token(self) -> str:
        if self.error:
            raise self.error
        return "token"

    async def reply(self, service_url, conversation_id, activity_id, text):
        self.calls.append(("reply", service_url, conversation_id, activity_id, text))
        if self.error:
            raise self.error
        return "reply-1"

    async def get_member(self, service_url, conversation_id, member_id):
        self.calls.append(("member", service_url, conversation_id, member_id))
        if self.error:
            raise self.error
        return SimpleNamespace(
            id=member_id,
            name="Ada Lovelace",
            email="ada@example.com",
            user_principal_name="ada@example.com",
            aad_object_id="aad-1",
            tenant_id="tenant-1",
        )

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def teams_on(monkeypatch):
    monkeypatch.setattr(teams, "teams_available", lambda: True)


@pytest.fixture
def stand_in():
    return StandInTeams()


@pytest.fixture
def client(teams_on, stand_in):
    app = FastAPI()
    assert install_teams_routes(app, set(), _settings())
    app.state.teams_bot = TeamsBot(_settings(), make_client=lambda: stand_in)
    return TestClient(app)


def _signed(method: str, target: str, body: bytes = b"", *, key: str = KEY, at: float | None = None) -> dict:
    timestamp = str(int(time.time() if at is None else at))
    return {
        TIMESTAMP_HEADER: timestamp,
        SIGNATURE_HEADER: sign(key.encode(), method, target, timestamp, body),
        "Content-Type": "application/json",
    }


def _deliver_body(**overrides) -> bytes:
    body = {
        "service_url": SERVICE_URL,
        "conversation_id": "19:abc@thread.tacv2;messageid=111",
        "reply_to_id": "111",
        "text": "The answer",
    }
    body.update(overrides)
    return json.dumps(body).encode()


def test_signed_deliver_posts_the_reply_in_the_thread(client, stand_in):
    body = _deliver_body()
    response = client.post(DELIVER, content=body, headers=_signed("POST", DELIVER, body))
    assert response.status_code == 200
    assert response.json() == {"id": "reply-1"}
    assert stand_in.calls == [
        ("reply", SERVICE_URL, "19:abc@thread.tacv2;messageid=111", "111", "The answer")
    ]


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {TIMESTAMP_HEADER: str(int(time.time())), SIGNATURE_HEADER: "0" * 64},
        "wrong key",
        "stale",
        "future",
        "other body",
        "not a number",
    ],
)
def test_deliver_refuses_a_call_the_proxy_didnt_sign(client, stand_in, headers):
    body = _deliver_body()
    if headers == "wrong key":
        headers = _signed("POST", DELIVER, body, key="another-key")
    elif headers == "stale":
        headers = _signed("POST", DELIVER, body, at=time.time() - 61)
    elif headers == "future":
        headers = _signed("POST", DELIVER, body, at=time.time() + 61)
    elif headers == "other body":
        headers = _signed("POST", DELIVER, _deliver_body(text="something else"))
    elif headers == "not a number":
        headers = {TIMESTAMP_HEADER: "soon", SIGNATURE_HEADER: sign(KEY.encode(), "POST", DELIVER, "soon", body)}
    response = client.post(DELIVER, content=body, headers=headers)
    assert response.status_code == 401
    assert stand_in.calls == []


def test_a_non_ascii_signature_is_refused_not_a_server_error(client, stand_in):
    body = _deliver_body()
    headers = {TIMESTAMP_HEADER: str(int(time.time())), SIGNATURE_HEADER: "é".encode("latin-1")}
    response = client.post(DELIVER, content=body, headers=headers)
    assert response.status_code == 401
    assert stand_in.calls == []


def test_a_signature_for_one_route_doesnt_open_the_other(client, stand_in):
    body = _deliver_body()
    response = client.post(DELIVER, content=body, headers=_signed("POST", MEMBER, body))
    assert response.status_code == 401
    assert stand_in.calls == []


@pytest.mark.parametrize(
    "service_url",
    [
        "https://attacker.example/amer/",
        "http://smba.trafficmanager.net/amer/",
        "https://smba.trafficmanager.net.attacker.example/amer/",
        "https://smba.trafficmanager.net@attacker.example/amer/",
        "https://user@smba.trafficmanager.net/amer/",
        "https://smba.trafficmanager.net:8443/amer/",
        "not a url",
    ],
)
def test_deliver_posts_only_to_an_allowed_teams_host(client, stand_in, service_url):
    body = _deliver_body(service_url=service_url)
    response = client.post(DELIVER, content=body, headers=_signed("POST", DELIVER, body))
    assert response.status_code == 403
    assert stand_in.calls == []


def test_the_allowed_hosts_come_from_the_settings(teams_on, stand_in):
    settings = _settings(service_hosts="teams.example, smba.trafficmanager.net")
    app = FastAPI()
    install_teams_routes(app, set(), settings)
    app.state.teams_bot = TeamsBot(settings, make_client=lambda: stand_in)
    body = _deliver_body(service_url="https://teams.example/emea/")
    response = TestClient(app).post(DELIVER, content=body, headers=_signed("POST", DELIVER, body))
    assert response.status_code == 200
    assert stand_in.calls[0][1] == "https://teams.example/emea/"


def test_signed_member_lookup_returns_the_person(client, stand_in):
    target = MEMBER + "?" + urlencode(
        {"service_url": SERVICE_URL, "conversation_id": "19:abc@thread.tacv2", "member_id": "29:user-1"}
    )
    response = client.get(target, headers=_signed("GET", target))
    assert response.status_code == 200
    assert response.json() == {
        "id": "29:user-1",
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "user_principal_name": "ada@example.com",
        "aad_object_id": "aad-1",
        "tenant_id": "tenant-1",
    }
    assert stand_in.calls == [("member", SERVICE_URL, "19:abc@thread.tacv2", "29:user-1")]


def test_member_signature_covers_the_query(client, stand_in):
    query = {"service_url": SERVICE_URL, "conversation_id": "19:abc@thread.tacv2", "member_id": "29:user-1"}
    signed_for = MEMBER + "?" + urlencode(query)
    asked = MEMBER + "?" + urlencode({**query, "member_id": "29:someone-else"})
    response = client.get(asked, headers=_signed("GET", signed_for))
    assert response.status_code == 401
    assert stand_in.calls == []


def test_member_lookup_posts_only_to_an_allowed_teams_host(client, stand_in):
    target = MEMBER + "?" + urlencode(
        {"service_url": "https://attacker.example/", "conversation_id": "c", "member_id": "m"}
    )
    response = client.get(target, headers=_signed("GET", target))
    assert response.status_code == 403
    assert stand_in.calls == []


def test_a_refused_bot_answers_502_without_the_details(teams_on):
    refused = StandInTeams(error=TeamsAuthError("invalid_client: AADSTS7000215 secret " + SECRET))
    app = FastAPI()
    install_teams_routes(app, set(), _settings())
    app.state.teams_bot = TeamsBot(_settings(), make_client=lambda: refused)
    body = _deliver_body()
    response = TestClient(app).post(DELIVER, content=body, headers=_signed("POST", DELIVER, body))
    assert response.status_code == 502
    assert response.json() == {"detail": "Microsoft refused the bot's credentials"}


def test_a_teams_service_error_answers_502(teams_on):
    failing = StandInTeams(error=RuntimeError("500 from the Teams service"))
    app = FastAPI()
    install_teams_routes(app, set(), _settings())
    app.state.teams_bot = TeamsBot(_settings(), make_client=lambda: failing)
    body = _deliver_body()
    response = TestClient(app).post(DELIVER, content=body, headers=_signed("POST", DELIVER, body))
    assert response.status_code == 502
    assert response.json() == {"detail": "Teams reply failed"}


def test_the_client_is_created_once_and_closed(teams_on):
    created = []

    def make():
        created.append(StandInTeams())
        return created[-1]

    app = FastAPI()
    install_teams_routes(app, set(), _settings())
    bot = TeamsBot(_settings(), make_client=make)
    app.state.teams_bot = bot
    with TestClient(app) as http:
        for _ in range(3):
            body = _deliver_body()
            assert http.post(DELIVER, content=body, headers=_signed("POST", DELIVER, body)).status_code == 200
    assert len(created) == 1
    asyncio.run(bot.aclose())
    assert created[0].closed


def test_routes_mount_and_skip_the_bearer_layer_when_configured(teams_on):
    app, exempt = FastAPI(), set()
    assert install_teams_routes(app, exempt, _settings())
    assert exempt == {DELIVER, MEMBER}
    assert {route.path for route in app.routes} >= {DELIVER, MEMBER}


def test_teams_stays_off_and_quiet_with_nothing_set(teams_on, caplog):
    app, exempt = FastAPI(), set()
    with caplog.at_level(logging.WARNING):
        assert not install_teams_routes(
            app, exempt, _settings(bot_app_id="", bot_app_secret="", tenant_id="", internal_key="")
        )
    assert exempt == set()
    assert not hasattr(app.state, "teams_bot")
    assert caplog.records == []


def test_teams_stays_off_and_says_what_is_missing(teams_on, caplog):
    app, exempt = FastAPI(), set()
    with caplog.at_level(logging.WARNING):
        assert not install_teams_routes(app, exempt, _settings(tenant_id="", internal_key=""))
    assert exempt == set()
    assert "COWORK_TEAMS_TENANT_ID, COWORK_TEAMS_INTERNAL_KEY not set" in caplog.text
    assert SECRET not in caplog.text


def test_teams_stays_off_without_the_teams_packages(monkeypatch, caplog):
    monkeypatch.setattr(teams, "teams_available", lambda: False)
    app, exempt = FastAPI(), set()
    with caplog.at_level(logging.WARNING):
        assert not install_teams_routes(app, exempt, _settings())
    assert exempt == set()
    assert "installed only on Linux" in caplog.text


def test_settings_come_from_cowork_teams_environment_names(monkeypatch):
    monkeypatch.setenv("COWORK_TEAMS_BOT_APP_ID", "app-env")
    monkeypatch.setenv("COWORK_TEAMS_BOT_APP_SECRET", "secret-env")
    monkeypatch.setenv("COWORK_TEAMS_TENANT_ID", "tenant-env")
    monkeypatch.setenv("COWORK_TEAMS_INTERNAL_KEY", "key-env")
    monkeypatch.setenv("COWORK_TEAMS_SERVICE_HOSTS", "a.example,B.example")
    settings = TeamsSettings()
    assert settings.missing() == []
    assert settings.bot_app_id == "app-env"
    assert settings.service_hosts == ("a.example", "b.example")


def test_the_server_boots_with_the_routes_declared(monkeypatch):
    """create_app() refuses to start when a route declares no permission."""
    from cowork.server import create_app

    monkeypatch.setattr(teams, "teams_available", lambda: True)
    for name, value in {
        "COWORK_TEAMS_BOT_APP_ID": "app-1",
        "COWORK_TEAMS_BOT_APP_SECRET": SECRET,
        "COWORK_TEAMS_TENANT_ID": "tenant-1",
        "COWORK_TEAMS_INTERNAL_KEY": KEY,
    }.items():
        monkeypatch.setenv(name, value)
    app = create_app()
    assert {DELIVER, MEMBER} <= {route.path for route in app.routes}
    assert {DELIVER, MEMBER} <= app.state.channel_webhook_paths


def test_check_reports_accepted_credentials(teams_on, capsys):
    assert teams.check(_settings(), make_client=StandInTeams) == 0
    out = capsys.readouterr().out
    assert "Microsoft accepted the bot's credentials (App ID app-1, tenant tenant-1)" in out
    assert SECRET not in out


def test_check_reports_refused_credentials(teams_on, capsys):
    refused = StandInTeams(error=TeamsAuthError("invalid_client: AADSTS7000215"))
    assert teams.check(_settings(), make_client=lambda: refused) == 1
    assert "Microsoft refused the bot's credentials: invalid_client: AADSTS7000215" in capsys.readouterr().out


def test_check_names_missing_settings(teams_on, capsys):
    assert teams.check(_settings(bot_app_secret="")) == 2
    assert "COWORK_TEAMS_BOT_APP_SECRET not set" in capsys.readouterr().out


def test_check_flags_a_missing_internal_key(teams_on, capsys):
    assert teams.check(_settings(internal_key=""), make_client=StandInTeams) == 1
    assert "COWORK_TEAMS_INTERNAL_KEY is not set" in capsys.readouterr().out
