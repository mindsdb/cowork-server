"""GET /hub/org-icons: the caller's organizations' icons, read
from auth's GET /v1/organizations/available/ with the caller's own
credential, for the org switcher. auth is stubbed at get_mindshub_reply."""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from cowork.api.v1.endpoints import mindshub_org_icons as ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.services import mindshub_org_icons, org_branding
from cowork.services.mindshub_client import MindsHubAnswer, MindsHubReply

ORG_A = "33333333-3333-4333-8333-333333333333"
ORG_B = "44444444-4444-4444-8444-444444444444"
ORG_C = "55555555-5555-4555-8555-555555555555"
USER = "a1111111-1111-4111-8111-111111111111"
PUBLIC_AUTH = "https://auth-pr-cowork-1.dev.mindshub.ai"


def _session_jwt(subject: str) -> str:
    def part(value: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'RS256', 'typ': 'JWT'})}.{part({'sub': subject})}.c2lnbmF0dXJl"


SESSION = _session_jwt("member")
OTHER_SESSION = _session_jwt("other")
HUB_SESSION = _session_jwt("hub")
AVAILABLE = {
    "results": [
        {"id": ORG_A.upper(), "name": "acme", "display_name": "Acme",
         "brand_icon_url": f"/v1/organizations/{ORG_A}/branding/icon?v=1"},
        {"id": ORG_B, "name": "globex", "display_name": "Globex", "brand_icon_url": None},
        {"id": "not-a-uuid", "name": "x", "display_name": "X", "brand_icon_url": None},
        {"id": ORG_C, "name": "evil", "display_name": "Evil", "brand_icon_url": "javascript:alert(1)"},
    ],
    "blocked_organizations": [
        {"id": "66666666-6666-4666-8666-666666666666", "name": "initech", "display_name": "Initech",
         "brand_icon_url": "https://auth.acme.com/v1/organizations/6/branding/icon?v=2",
         "reason": "organization_owner_required"},
        {"id": ORG_B, "name": "globex", "display_name": "Globex", "brand_icon_url": None},
    ],
}


class FakeAuth:
    def __init__(self) -> None:
        self.reply = MindsHubReply(MindsHubAnswer.OK, payload=AVAILABLE, status_code=200)
        self.calls: list[tuple[str, str]] = []

    async def get(self, path, bearer):
        self.calls.append((path, bearer))
        return self.reply


@pytest.fixture
def auth(monkeypatch):
    fake = FakeAuth()
    monkeypatch.setattr(mindshub_org_icons, "get_mindshub_reply", fake.get)
    monkeypatch.setattr(org_branding, "default_minds_auth_host", lambda: PUBLIC_AUTH)
    mindshub_org_icons._cache.clear()
    yield fake
    mindshub_org_icons._cache.clear()


@pytest.fixture
def local_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def _request(bearer: str | None = None, *, hub: str | None = None) -> SimpleNamespace:
    headers = {}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    if hub:
        headers["X-MindsHub-Authorization"] = f"Bearer {hub}"
    return SimpleNamespace(headers=headers, state=SimpleNamespace())


def _hosted(org_id: str = ORG_A) -> TenantScope:
    return TenantScope(org_mode=True, org_id=org_id, user_id=USER)


def _get(request, scope) -> dict:
    return asyncio.run(ep.get_org_icons(request, scope)).model_dump(by_alias=True)


EXPECTED = {
    "reachable": True,
    "orgs": [
        {"id": ORG_A, "iconUrl": f"{PUBLIC_AUTH}/v1/organizations/{ORG_A}/branding/icon?v=1"},
        {"id": ORG_B, "iconUrl": None},
        {"id": ORG_C, "iconUrl": None},
        {"id": "66666666-6666-4666-8666-666666666666",
         "iconUrl": "https://auth.acme.com/v1/organizations/6/branding/icon?v=2"},
    ],
}


def test_hosted_lists_every_org_by_keycloak_id_with_a_loadable_icon(org_deployment, auth):
    assert _get(_request(SESSION, hub=HUB_SESSION), _hosted()) == EXPECTED
    # The gateway's credential, never the client-set hub header.
    assert auth.calls == [("/organizations/available/", SESSION)]


def test_local_reads_with_the_desktops_hub_credential(local_mode, auth):
    assert _get(_request("this-servers-own-token", hub=HUB_SESSION), LOCAL_SCOPE) == EXPECTED
    assert auth.calls == [("/organizations/available/", HUB_SESSION)]


@pytest.mark.parametrize(
    "request_", [_request(), _request("this-servers-own-token"), _request(hub="mdb_" + "0" * 32)]
)
def test_local_without_a_session_is_empty_and_asks_no_one(local_mode, auth, request_):
    assert _get(request_, LOCAL_SCOPE) == {"reachable": True, "orgs": []}
    assert auth.calls == []


def test_hosted_without_an_org_or_with_an_api_key_is_empty(org_deployment, auth):
    assert _get(_request(SESSION), TenantScope(org_mode=True, org_id=None, user_id=USER)) == {
        "reachable": True, "orgs": [],
    }
    assert _get(_request("mdb_" + "0" * 32), _hosted()) == {"reachable": True, "orgs": []}
    assert auth.calls == []


def test_auth_unreachable_is_reachable_false(org_deployment, auth):
    auth.reply = MindsHubReply(MindsHubAnswer.UNREACHABLE, status_code=503)
    assert _get(_request(SESSION), _hosted()) == {"reachable": False, "orgs": []}


def test_auths_403_is_no_orgs_reachable(org_deployment, auth):
    auth.reply = MindsHubReply(MindsHubAnswer.REFUSED, status_code=403)
    assert _get(_request(SESSION), _hosted()) == {"reachable": True, "orgs": []}


@pytest.mark.parametrize("status_code", [401, 404, 429])
def test_any_other_refusal_is_unknown_and_not_cached(org_deployment, auth, status_code):
    answer = MindsHubAnswer.THROTTLED if status_code == 429 else MindsHubAnswer.REFUSED
    auth.reply = MindsHubReply(answer, status_code=status_code)
    assert _get(_request(SESSION), _hosted()) == {"reachable": False, "orgs": []}
    auth.reply = MindsHubReply(MindsHubAnswer.OK, payload=AVAILABLE, status_code=200)
    assert _get(_request(SESSION), _hosted()) == EXPECTED
    assert len(auth.calls) == 2


def test_the_answer_is_cached_per_caller(org_deployment, auth):
    _get(_request(SESSION), _hosted())
    _get(_request(SESSION), _hosted())
    assert len(auth.calls) == 1
    _get(_request(OTHER_SESSION), _hosted())
    assert [bearer for _path, bearer in auth.calls] == [SESSION, OTHER_SESSION]


def test_the_route_is_get_hub_org_icons():
    from cowork.api.v1.router import api_router

    assert [(r.path, sorted(r.methods)) for r in api_router.routes if "org-icons" in r.path] == [
        ("/api/v1/hub/org-icons", ["GET"]),
    ]
