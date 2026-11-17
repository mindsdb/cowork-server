"""The browser's decisions proxy (ENG-4138): a web turn's pod asks the
organization's own decision model through this server with its turn key, and
the organization's key never leaves this server."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from cowork.api.v1.endpoints import browse
from cowork.handlers import jev_route
from cowork.principal import Principal, get_principal
from cowork.services import decisions_proxy

REAL_POST_PINNED = jev_route._post_pinned
ORG, OTHER_ORG = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"
USER = "33333333-3333-3333-3333-333333333333"
TURN_KEY = "mdb_abcdefgh.0123456789abcdef0123456789abcdef"
OWN_URL = "https://decide.example.test/v1/systemone"
DECISION = {"model": "whatever-the-pod-says", "state": {"goal": "search"}, "questions": {"risky": {"type": "noul"}},
            "extra": "dropped"}


class FakeRedis:
    """get/set/expire/delete and a pipeline, with expiries recorded."""

    def __init__(self):
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.fail = False

    async def get(self, key):
        if self.fail:
            raise ConnectionError("redis down")
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key], self.ttls[key] = value, ex

    async def expire(self, key, seconds):
        if key in self.values:
            self.ttls[key] = seconds

    async def delete(self, *keys):
        for key in keys:
            self.values.pop(key, None)

    def pipeline(self, transaction=True):
        redis = self

        class Pipe:
            def __init__(self):
                self.ops = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def set(self, *args, **kwargs):
                self.ops.append((args, kwargs))

            async def execute(self):
                for args, kwargs in self.ops:
                    await redis.set(*args, **kwargs)

        return Pipe()


@pytest.fixture
def redis(monkeypatch):
    from cowork.turnqueue import redis_client

    fake = FakeRedis()
    monkeypatch.setattr(redis_client, "get_redis", lambda: fake)
    return fake


@pytest.fixture
def org(monkeypatch):
    """The organization's settings: its own decision model, stored with its key."""
    settings = SimpleNamespace(
        decision_model_url=OWN_URL, decision_model_api_key=SecretStr("org-own-key"), decision_model="jev-own",
    )
    asked = []

    def get_user_settings(scope=None):
        asked.append((scope.org_id, scope.user_id))
        return settings

    monkeypatch.setattr(browse, "get_user_settings", get_user_settings)
    return SimpleNamespace(settings=settings, asked=asked)


@pytest.fixture
def upstream(monkeypatch):
    """The organization's endpoint, reached only through the gate's pinned
    egress on the hosted server: what it was sent, and what it answers."""
    state = SimpleNamespace(calls=[], answer=None, raises=None)

    async def post_pinned(target, *, headers, payload, timeout_seconds):
        state.calls.append((target.url, headers, payload, timeout_seconds))
        if state.raises is not None:
            raise state.raises
        import httpx

        return state.answer or httpx.Response(200, json={"answers": {"risky": {"noul": 0.1}}})

    monkeypatch.setattr(jev_route, "_post_pinned", post_pinned)
    return state


@pytest.fixture
def client(monkeypatch, org_deployment, redis, org, upstream):
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    app = FastAPI()
    app.include_router(browse.router, prefix="/api/v1/browse")
    state = SimpleNamespace(principal=None)
    app.dependency_overrides[get_principal] = lambda: state.principal
    client = TestClient(app)
    client.as_principal = lambda principal: setattr(state, "principal", principal)
    return client


def _admit(redis, *, org_id=ORG, user_id=USER, turn_key=TURN_KEY):
    asyncio.run(decisions_proxy.admit(
        redis, turn_key=turn_key, org_id=org_id, user_id=user_id, correlation_id="corr-1", ttl_seconds=1200,
    ))


def _ask(client, *, key=TURN_KEY, body=None):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return client.post("/api/v1/browse/decisions", headers=headers, json=DECISION if body is None else body)


def test_the_turn_key_reaches_the_organizations_model_with_its_stored_key(client, redis, org, upstream):
    _admit(redis)
    answer = _ask(client)
    assert answer.status_code == 200 and answer.json() == {"answers": {"risky": {"noul": 0.1}}}
    [(url, headers, payload, timeout)] = upstream.calls
    assert url == OWN_URL and headers["Authorization"] == "Bearer org-own-key"
    # Only the decision goes on, under the organization's model name.
    assert payload == {"model": "jev-own", "state": DECISION["state"], "questions": DECISION["questions"]}
    assert timeout == decisions_proxy.UPSTREAM_TIMEOUT_S
    assert org.asked == [(ORG, USER)]
    # The key itself is never stored, only its digest.
    assert not any(TURN_KEY in key or TURN_KEY in value for key, value in redis.values.items())


def test_the_gateways_identity_must_match_the_key(client, redis, upstream):
    _admit(redis)
    client.as_principal(Principal(user_id=USER, org_id=OTHER_ORG))
    assert _ask(client).status_code == 403
    client.as_principal(Principal(user_id=USER, org_id=ORG))
    assert _ask(client).status_code == 200
    assert len(upstream.calls) == 1


@pytest.mark.parametrize("key", ["", "mdb_other123.0123456789abcdef0123456789abcdef", "eyJhbGciOiJSUzI1NiJ9.e30.sig"])
def test_no_credential_or_any_other_one_is_refused(client, redis, upstream, key):
    """No key, another turn's key, a person's sign-in: none reaches the model,
    whatever identity came with it."""
    _admit(redis)
    client.as_principal(Principal(user_id=USER, org_id=ORG))
    assert _ask(client, key=key).status_code == 401
    assert upstream.calls == []


def test_a_key_admitted_for_another_organization_reaches_only_its_model(client, redis, org, upstream):
    _admit(redis, org_id=OTHER_ORG)
    assert _ask(client).status_code == 200
    assert org.asked == [(OTHER_ORG, USER)]


def test_a_turn_that_ended_is_refused(client, redis, upstream):
    _admit(redis)
    asyncio.run(decisions_proxy.forget(redis, correlation_id="corr-1"))
    assert _ask(client).status_code == 401 and upstream.calls == []
    assert redis.values == {}


def test_extending_the_key_keeps_its_record(redis):
    _admit(redis)
    asyncio.run(decisions_proxy.extend(redis, correlation_id="corr-1", ttl_seconds=4000))
    assert set(redis.ttls.values()) == {4000}


def test_an_unreadable_record_is_unavailable(client, redis, upstream):
    _admit(redis)
    redis.fail = True
    assert _ask(client).status_code == 503 and upstream.calls == []


def test_off_the_remote_queue_there_is_no_proxy(client, redis, monkeypatch, upstream):
    _admit(redis)
    monkeypatch.setenv("COWORK_TURN_BACKEND", "inprocess")
    assert _ask(client).status_code == 404 and upstream.calls == []


def test_a_body_past_the_limit_is_refused_unread(client, redis, upstream):
    _admit(redis)
    big = {"state": {"text": "x" * (decisions_proxy.MAX_BODY_BYTES + 1)}, "questions": {}}
    assert _ask(client, body=big).status_code == 413
    # Without a Content-Length too.
    chunks = (b"x" * 65536 for _ in range(decisions_proxy.MAX_BODY_BYTES // 65536 + 2))
    streamed = client.post(
        "/api/v1/browse/decisions", headers={"Authorization": f"Bearer {TURN_KEY}"}, content=chunks,
    )
    assert streamed.status_code == 413
    assert upstream.calls == []


@pytest.mark.parametrize("body", [{"state": {}}, {"state": [], "questions": {}}, ["state"]])
def test_a_malformed_decision_is_refused(client, redis, upstream, body):
    _admit(redis)
    assert _ask(client, body=body).status_code == 400 and upstream.calls == []


def test_once_the_organization_removed_its_model_there_is_none(client, redis, org, upstream):
    _admit(redis)
    org.settings.decision_model_url = ""
    answer = _ask(client)
    assert answer.status_code == 404 and answer.json()["detail"]["code"] == "no_decision_model"
    assert upstream.calls == []


def test_the_models_failures_come_back_as_gateway_errors(client, redis, upstream):
    import httpx

    _admit(redis)
    upstream.raises = TimeoutError()
    assert _ask(client).status_code == 504
    upstream.raises = None
    upstream.answer = httpx.Response(401, json={})
    failed = _ask(client)
    assert failed.status_code == 502 and failed.json()["detail"]["error"] == "http_401"
    upstream.answer = httpx.Response(200, text="not json")
    assert _ask(client).json()["detail"]["error"] == "malformed_response"


def test_a_private_endpoint_is_never_dialed(client, redis, monkeypatch):
    """The gate's own egress: on the hosted server an organization's
    endpoint that resolves to a private address is refused before the key
    goes anywhere."""
    from cowork.services.connectors.egress import EgressHostNotPublic

    _admit(redis)
    monkeypatch.setattr(jev_route, "_post_pinned", REAL_POST_PINNED)

    def private(host, resolver):
        raise EgressHostNotPublic(host)

    monkeypatch.setattr("cowork.services.connectors.egress.vetted_public_addresses", private)
    answer = _ask(client)
    assert answer.status_code == 502 and answer.json()["detail"]["error"] == "endpoint_refused"


def test_the_proxy_url_is_the_setting_or_the_public_base(monkeypatch):
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.delenv("COWORK_BROWSER_DECISIONS_URL", raising=False)
    monkeypatch.setenv("COWORK_PUBLIC_BASE_URL", "https://cowork.example.test/")
    get_app_settings.cache_clear()
    try:
        assert decisions_proxy.proxy_url() == "https://cowork.example.test/api/v1/browse/decisions"
        monkeypatch.setenv("COWORK_BROWSER_DECISIONS_URL", "https://internal.example.test/d/")
        get_app_settings.cache_clear()
        assert decisions_proxy.proxy_url() == "https://internal.example.test/d"
        monkeypatch.delenv("COWORK_BROWSER_DECISIONS_URL")
        monkeypatch.setenv("COWORK_PUBLIC_BASE_URL", "http://localhost:8000")
        get_app_settings.cache_clear()
        assert decisions_proxy.proxy_url() == ""
    finally:
        get_app_settings.cache_clear()


def test_the_route_declares_the_turn_key_as_its_credential():
    from cowork.api.v1.permissions import TurnKey
    from cowork.api.v1.route_walker import declared_permissions

    [route] = [r for r in browse.router.routes if r.path == "/decisions"]
    assert declared_permissions(route) == [TurnKey]
