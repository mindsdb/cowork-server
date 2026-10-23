"""TeamsClient against a stand-in Microsoft sign-in endpoint and Teams service.

The Teams packages are Linux-only dependencies, so the SDK tests skip on
other platforms; the no-SDK test runs everywhere.
"""

from __future__ import annotations

import json
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
    """The Teams service: records requests and answers replies and member lookups."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST" and request.url.path.endswith("/activities"):
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
    from cowork.channels import teams_client

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
    with pytest.raises(ValueError):
        sdk.TeamsClient(APP_ID, SECRET, "")


def test_imports_without_the_teams_packages():
    # A desktop install has neither package; the module must still import and
    # refuse to build a client, rather than fail at import time.
    code = (
        "import sys\n"
        "sys.modules['msal'] = None\n"
        "sys.modules['microsoft_teams'] = None\n"
        "from cowork.channels import teams_client as t\n"
        "assert not t.teams_available()\n"
        "try:\n"
        "    t.TeamsClient('a', 's', 't')\n"
        "except t.TeamsUnavailableError:\n"
        "    print('unavailable')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "unavailable"
