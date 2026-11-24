"""GET and POST /hub/deletion-request/: the caller's request to delete their
MindsHub account, passed to MindsHub's /v1/me/deletion-request/ with the
gateway's credential. MindsHub is stubbed at get_mindshub_reply and
send_mindshub."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints import mindshub_account_deletion as ep
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.services import mindshub_account_deletion
from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply
from tests._org_sharing import MEMBER_BEARER, member_request

ORG_ID = "33333333-3333-4333-8333-333333333333"
USER = "a1111111-1111-4111-8111-111111111111"
SCOPE = TenantScope(org_mode=True, org_id=ORG_ID, user_id=USER)
OPEN_REQUEST = {"requested_at": "2026-11-17T10:00:00Z", "delete_by": "2026-12-17T10:00:00Z"}
OPEN_VIEW = {"requestedAt": "2026-11-17T10:00:00+00:00", "deleteBy": "2026-12-17T10:00:00+00:00"}


class FakeAuth:
    def __init__(self) -> None:
        self.get_reply = MindsHubReply(MindsHubAnswer.OK, payload=OPEN_REQUEST, status_code=200)
        self.post_reply = MindsHubReply(MindsHubAnswer.OK, payload=OPEN_REQUEST, status_code=202)
        self.calls: list[tuple[str, str, str]] = []

    async def get(self, path, bearer):
        self.calls.append(("GET", path, bearer))
        return self.get_reply

    async def send(self, method, path, bearer, *, json_body=None):
        assert json_body is None
        self.calls.append((method, path, bearer))
        return self.post_reply


@pytest.fixture
def auth(monkeypatch):
    fake = FakeAuth()
    monkeypatch.setattr(mindshub_account_deletion, "get_mindshub_reply", fake.get)
    monkeypatch.setattr(mindshub_account_deletion, "send_mindshub", fake.send)
    return fake


def _get(scope=SCOPE) -> dict:
    view = asyncio.run(ep.get_deletion_request(member_request(), scope))
    return view.model_dump(mode="json", by_alias=True)


def _post(scope=SCOPE) -> dict:
    view = asyncio.run(ep.post_deletion_request(member_request(), scope))
    return view.model_dump(mode="json", by_alias=True)


def _refusal(call) -> HTTPException:
    with pytest.raises(HTTPException) as caught:
        call()
    return caught.value


def test_get_answers_the_open_request(auth):
    assert _get() == OPEN_VIEW
    # The gateway's credential, never the client-set X-MindsHub-Authorization.
    assert auth.calls == [("GET", "/me/deletion-request/", MEMBER_BEARER)]


def test_get_is_404_when_mindshub_has_no_open_request(auth):
    auth.get_reply = MindsHubReply(MindsHubAnswer.REFUSED, payload=None, status_code=404)
    assert _refusal(_get).status_code == 404


def test_post_answers_the_request_mindshub_recorded(auth):
    assert _post() == OPEN_VIEW
    assert auth.calls == [("POST", "/me/deletion-request/", MEMBER_BEARER)]


def test_the_post_route_answers_202():
    from cowork.api.v1.router import api_router

    (route,) = [
        r for r in api_router.routes
        if getattr(r, "path", "") == "/api/v1/hub/deletion-request/" and "POST" in r.methods
    ]
    assert route.status_code == 202


@pytest.mark.parametrize(
    "reply",
    [
        MindsHubReply(MindsHubAnswer.UNREACHABLE, status_code=503),
        MindsHubReply(MindsHubAnswer.UNREACHABLE),
        MindsHubReply(MindsHubAnswer.OK, payload={"unexpected": True}, status_code=200),
        MindsHubReply(MindsHubAnswer.REFUSED, payload=None, status_code=400),
    ],
)
def test_mindshub_failing_is_a_502(auth, reply):
    auth.get_reply = auth.post_reply = reply
    assert _refusal(_get).status_code == 502
    assert _refusal(_post).status_code == 502
    # One call each: the POST is never retried.
    assert [call[0] for call in auth.calls] == ["GET", "POST"]


@pytest.mark.parametrize("status_code", [401, 403])
def test_mindshub_refusing_the_credential_is_a_403(auth, status_code):
    """Never a 401, which the app reads as its own sign-out."""
    auth.get_reply = auth.post_reply = MindsHubReply(MindsHubAnswer.REFUSED, status_code=status_code)
    assert _refusal(_get).status_code == 403
    assert _refusal(_post).status_code == 403


def test_mindshub_throttling_is_a_429(auth):
    auth.post_reply = MindsHubReply(MindsHubAnswer.THROTTLED, status_code=429)
    assert _refusal(_post).status_code == 429


@pytest.mark.parametrize(
    "scope", [LOCAL_SCOPE, None, TenantScope(org_mode=False, org_id=None, user_id=None)]
)
def test_outside_mindshub_mode_is_404_and_asks_no_one(auth, scope):
    """No MindsHub account to delete, and no credential sent: on the desktop
    the Authorization is this server's own."""
    assert _refusal(lambda: _get(scope)).status_code == 404
    assert _refusal(lambda: _post(scope)).status_code == 404
    assert auth.calls == []
