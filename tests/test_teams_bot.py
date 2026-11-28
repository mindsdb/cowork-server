"""The Teams connector's internal routes and the bot's credentials check
(cowork/api/v1/endpoints/connectors/teams.py, cowork/services/connectors/teams/bot.py).

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

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints.connectors import teams as routes
from cowork.api.v1.endpoints.connectors.teams import install_teams_routes
from cowork.services.connectors.teams import bot as teams
from cowork.services.connectors.teams.bot import (
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    TeamsBot,
    TeamsSettings,
    sign,
)
from cowork.services.connectors.teams.client import TeamsAuthError

KEY = "internal-key-value"
SECRET = "bot-secret-value"
SERVICE_URL = "https://smba.trafficmanager.net/amer/tenant-1/"
DELIVER = "/api/v1/connectors/teams/deliver"
MEMBER = "/api/v1/connectors/teams/member"
STATUS = "/api/v1/connectors/teams/status"


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
        # The stream id Teams gives a first status update; None when it gives none.
        self.opens: str | None = "stream-1"
        self.calls: list[tuple] = []
        self.error = error
        # Raised by a reply that closes a stream only, as Teams refuses one past its 10 minutes.
        self.stream_error: Exception | None = None
        self.closed = False

    async def bot_token(self) -> str:
        if self.error:
            raise self.error
        return "token"

    async def reply(self, service_url, conversation_id, activity_id, text, *, mention=None, stream_id=None):
        call = ("reply", service_url, conversation_id, activity_id, text)
        if mention:
            call += (mention,)
        if stream_id:
            call += (("stream", stream_id),)
        self.calls.append(call)
        if stream_id and self.stream_error:
            raise self.stream_error
        if self.error:
            raise self.error
        return "reply-1"

    async def status(self, service_url, conversation_id, activity_id, text, *, stream_id=None, sequence=1):
        self.calls.append(("status", service_url, conversation_id, activity_id, text, stream_id, sequence))
        if self.error:
            raise self.error
        return stream_id or self.opens

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
            user_role="user",
        )

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def teams_on(monkeypatch):
    # Both modules import it: the routes check it at install, the bot module at boot.
    monkeypatch.setattr(teams, "teams_available", lambda: True)
    monkeypatch.setattr(routes, "teams_available", lambda: True)


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


def test_deliver_mentions_the_member_the_proxy_names(client, stand_in):
    body = _deliver_body(mention={"id": "29:ana", "name": "Ana"})
    response = client.post(DELIVER, content=body, headers=_signed("POST", DELIVER, body))
    assert response.status_code == 200
    assert stand_in.calls == [
        ("reply", SERVICE_URL, "19:abc@thread.tacv2;messageid=111", "111", "The answer", ("29:ana", "Ana"))
    ]


def _status_body(**overrides) -> bytes:
    body = {"service_url": SERVICE_URL, "conversation_id": "19:abc@thread.tacv2;messageid=111",
            "reply_to_id": "111", "text": "Working on it…"}
    body.update(overrides)
    return json.dumps(body).encode()


def test_status_answers_no_stream_when_teams_opened_none(client, stand_in):
    stand_in.opens = None
    body = _status_body()
    response = client.post(STATUS, content=body, headers=_signed("POST", STATUS, body))
    assert (response.status_code, response.json()) == (200, {"stream_id": None})


def test_signed_status_opens_a_stream_and_a_later_one_continues_it(client, stand_in):
    first = _status_body()
    opened = client.post(STATUS, content=first, headers=_signed("POST", STATUS, first))
    later = _status_body(text="Still working on it…", stream_id="stream-1", sequence=2)
    continued = client.post(STATUS, content=later, headers=_signed("POST", STATUS, later))

    assert (opened.status_code, opened.json()) == (200, {"stream_id": "stream-1"})
    assert (continued.status_code, continued.json()) == (200, {"stream_id": "stream-1"})
    assert stand_in.calls == [
        ("status", SERVICE_URL, "19:abc@thread.tacv2;messageid=111", "111", "Working on it…", None, 1),
        ("status", SERVICE_URL, "19:abc@thread.tacv2;messageid=111", "111", "Still working on it…", "stream-1", 2),
    ]


def test_status_refuses_a_call_the_proxy_didnt_sign_or_a_line_teams_wont_take(client, stand_in):
    body = _status_body()
    unsigned = client.post(STATUS, content=body, headers={"Content-Type": "application/json"})
    too_long = _status_body(text="x" * 1001)
    long_reply = client.post(STATUS, content=too_long, headers=_signed("POST", STATUS, too_long))
    other_host = _status_body(service_url="https://teams.example/emea/")
    elsewhere = client.post(STATUS, content=other_host, headers=_signed("POST", STATUS, other_host))

    assert (unsigned.status_code, long_reply.status_code) == (401, 422)
    assert elsewhere.status_code == 403
    assert stand_in.calls == []


def test_deliver_with_a_stream_closes_it_with_the_answer(client, stand_in):
    body = _deliver_body(stream_id="stream-1")
    response = client.post(DELIVER, content=body, headers=_signed("POST", DELIVER, body))
    assert response.status_code == 200
    assert stand_in.calls == [
        ("reply", SERVICE_URL, "19:abc@thread.tacv2;messageid=111", "111", "The answer", ("stream", "stream-1"))
    ]


def _teams_status(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", SERVICE_URL)
    return httpx.HTTPStatusError("refused", request=request, response=httpx.Response(code, request=request))


@pytest.mark.parametrize("code", [400, 403, 404])
def test_an_answer_teams_wont_close_the_stream_with_still_goes_out_as_a_plain_reply(client, stand_in, code):
    stand_in.stream_error = _teams_status(code)
    body = _deliver_body(stream_id="stream-1")
    response = client.post(DELIVER, content=body, headers=_signed("POST", DELIVER, body))
    assert response.status_code == 200
    assert stand_in.calls == [
        ("reply", SERVICE_URL, "19:abc@thread.tacv2;messageid=111", "111", "The answer", ("stream", "stream-1")),
        ("reply", SERVICE_URL, "19:abc@thread.tacv2;messageid=111", "111", "The answer"),
    ]


@pytest.mark.parametrize("error", [
    _teams_status(500), _teams_status(401), _teams_status(408), _teams_status(429),
    httpx.ReadTimeout("timed out"), TeamsAuthError("invalid_client: refused"),
], ids=["5xx", "401", "408", "429", "timeout", "credentials"])
def test_a_stream_reply_that_may_have_posted_or_cant_succeed_isnt_sent_again(client, stand_in, error):
    stand_in.stream_error = error
    body = _deliver_body(stream_id="stream-1")
    response = client.post(DELIVER, content=body, headers=_signed("POST", DELIVER, body))
    assert response.status_code == 502
    assert len(stand_in.calls) == 1


@pytest.mark.parametrize("mention", [{"id": "", "name": "Ana"}, {"id": "29:ana", "name": ""}, {"id": "29:ana"}])
def test_deliver_refuses_an_incomplete_mention(client, stand_in, mention):
    body = _deliver_body(mention=mention)
    response = client.post(DELIVER, content=body, headers=_signed("POST", DELIVER, body))
    assert response.status_code == 422
    assert stand_in.calls == []


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
        "https://smba.trafficmanager.net:abc/amer/",
        "https://smba.trafficmanager.net:99999/amer/",
        "https://[x/amer/",
    ],
)
def test_deliver_posts_only_to_an_allowed_teams_host(client, stand_in, service_url):
    body = _deliver_body(service_url=service_url)
    response = client.post(DELIVER, content=body, headers=_signed("POST", DELIVER, body))
    assert response.status_code == 403
    assert stand_in.calls == []


def test_a_body_over_the_cap_is_refused_before_it_is_read(client, stand_in):
    body = _deliver_body(text="x" * (routes.MAX_BODY_BYTES + 1))
    response = client.post(DELIVER, content=body, headers=_signed("POST", DELIVER, body))
    assert response.status_code == 413
    assert stand_in.calls == []


def test_a_body_without_a_length_is_still_capped(client, stand_in):
    def chunks():
        for _ in range(routes.MAX_BODY_BYTES // 1024 + 2):
            yield b"x" * 1024

    response = client.post(DELIVER, content=chunks(), headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    assert stand_in.calls == []


def test_a_text_over_the_part_limit_is_refused(client, stand_in):
    body = _deliver_body(text="x" * (routes.MAX_TEXT_CHARS + 1))
    response = client.post(DELIVER, content=body, headers=_signed("POST", DELIVER, body))
    assert response.status_code == 422
    assert stand_in.calls == []


def test_malformed_json_is_refused_before_the_signature_is_checked(client, stand_in):
    # FastAPI parses the body before any dependency runs; nothing reaches Teams either way.
    response = client.post(DELIVER, content=b"{not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert stand_in.calls == []


@pytest.mark.parametrize("field", ["conversation_id", "reply_to_id"])
@pytest.mark.parametrize("value", ["19:abc/../../v3/x", "19:abc?x=1", "19:abc#x", "19:abc%2Fx", "..", "19:a b"])
def test_deliver_refuses_ids_that_would_change_the_teams_path(client, stand_in, field, value):
    body = _deliver_body(**{field: value})
    response = client.post(DELIVER, content=body, headers=_signed("POST", DELIVER, body))
    assert response.status_code == 422
    assert stand_in.calls == []


def test_deliver_takes_real_teams_ids(client, stand_in):
    body = _deliver_body(conversation_id="a:1Kx-_9QzE.abc=", reply_to_id="1:1abc-_x")
    response = client.post(DELIVER, content=body, headers=_signed("POST", DELIVER, body))
    assert response.status_code == 200
    assert stand_in.calls[0][2:4] == ("a:1Kx-_9QzE.abc=", "1:1abc-_x")


def test_member_lookup_refuses_ids_that_would_change_the_teams_path(client, stand_in):
    target = MEMBER + "?" + urlencode(
        {"service_url": SERVICE_URL, "conversation_id": "19:abc@thread.tacv2", "member_id": "29:x/../../y"}
    )
    response = client.get(target, headers=_signed("GET", target))
    assert response.status_code == 422
    assert stand_in.calls == []


def test_a_short_internal_key_is_warned_about(teams_on, caplog):
    with caplog.at_level(logging.WARNING):
        assert install_teams_routes(FastAPI(), set(), _settings(internal_key="changeme"))
    assert any(getattr(r, "event", None) == "teams_weak_key" for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert install_teams_routes(FastAPI(), set(), _settings(internal_key="k" * 32))
    assert not any(getattr(r, "event", None) == "teams_weak_key" for r in caplog.records)


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
        "user_role": "user",
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
    assert exempt == {DELIVER, MEMBER, STATUS}
    assert {route.path for route in app.routes} >= {DELIVER, MEMBER, STATUS}


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
    assert [record.reason for record in caplog.records] == ["COWORK_TEAMS_TENANT_ID, COWORK_TEAMS_INTERNAL_KEY not set"]
    assert SECRET not in caplog.text


def test_teams_stays_off_without_the_teams_packages(monkeypatch, caplog):
    monkeypatch.setattr(routes, "teams_available", lambda: False)
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
    monkeypatch.setattr(routes, "teams_available", lambda: True)
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


def test_the_server_log_names_the_missing_settings(teams_on):
    """The server's log format drops values filled into a message, so the
    names must travel as a field it keeps."""
    from cowork.common.structured_logging import make_formatter

    lines: list[str] = []
    handler = logging.Handler()
    handler.setFormatter(make_formatter())
    handler.emit = lambda record: lines.append(handler.format(record))
    logging.getLogger("cowork.api.v1.endpoints.connectors.teams").addHandler(handler)
    try:
        install_teams_routes(FastAPI(), set(), _settings(internal_key=""))
    finally:
        logging.getLogger("cowork.api.v1.endpoints.connectors.teams").removeHandler(handler)
    [line] = lines
    assert json.loads(line)["reason"] == "COWORK_TEAMS_INTERNAL_KEY not set"
