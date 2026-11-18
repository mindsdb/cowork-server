"""A turn key past auth's cap: minted again with the untimed lifetime, so the turn runs untimed.

auth refuses a key whose expiry is past its cap with a 400 that carries only a
detail. Every auth accepts the untimed key, so a time-limited turn still runs,
without the time limit its shorter key can't cover. A refusal of the person,
an outage, or any other answer sends no second mint, and every
permission_unavailable logs the auth route and its cause.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import AsyncMock

import httpx
import pytest

from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.db.scoped import TenantScope
from cowork.services import product_permissions
from cowork.services.product_permissions import ProductPermissionDenied, ProductPermissionUnavailable
from cowork.turnqueue import auth_keys
from cowork.turnqueue import producer as prod

from tests.turnqueue.test_producer import FakeRedis, _drain, _reply

SECRET = "internal-secret-never-logged"
PRODUCER_SECRET = "producer-secret-never-logged"
KEY = "mdb_pfx.plaintext-key-never-logged"
ORG = TenantScope(
    org_mode=True,
    org_id="11111111-1111-4111-8111-111111111111",
    user_id="22222222-2222-4222-8222-222222222222",
)


def _length_refusal(cap_seconds: int) -> httpx.Response:
    # auth's turn-key mint route answers this text, copied here because auth
    # can't be imported; the fallback matches its start (TTL_REFUSAL_PREFIX).
    return httpx.Response(400, json={"detail": f"expiry_date exceeds the max turn-key TTL of {cap_seconds}s"})


class _Auth:
    """auth over MockTransport. The mint refuses an expiry past ``cap_seconds``
    as auth does; ``first`` scripts the answers to the first posts (None falls
    through to the cap check). Records each post's body and how far ahead its
    expiry is."""

    def __init__(self, cap_seconds: int, *, first: list | None = None):
        self.cap_seconds = cap_seconds
        self.first = list(first or [])
        self.posts: list[dict] = []

    def ahead(self, n: int) -> float:
        expiry = datetime.fromisoformat(self.posts[n]["expiry_date"])
        return (expiry - datetime.now(timezone.utc)).total_seconds()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == auth_keys.TURN_KEYS_ROUTE, request.url
        self.posts.append(json.loads(request.content))
        answer = self.first.pop(0) if self.first else None
        if isinstance(answer, Exception):
            raise answer
        if answer is not None:
            return answer
        if self.ahead(-1) > self.cap_seconds:
            return _length_refusal(self.cap_seconds)
        return httpx.Response(201, json={"key": KEY, "prefix": "mdb_pfx"})


def _serve(monkeypatch, handler) -> None:
    """Every client this test opens talks to ``handler``, keeping the code's own client arguments."""
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


@pytest.fixture
def auth_env(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_AUTH_INTERNAL_BASE_URL", "http://auth.internal")
    monkeypatch.setenv("COWORK_TURN_AUTH_INTERNAL_SECRET", SECRET)


@pytest.fixture
def time_limit(monkeypatch, auth_env):
    monkeypatch.setenv("COWORK_TURN_TIME_LIMIT_SECONDS", "3600")


@pytest.fixture
def forgotten(monkeypatch):
    calls = []
    monkeypatch.setattr(auth_keys, "forget_stale_mindshub_workspace", lambda **kw: calls.append(kw))
    return calls


async def _mint(*, ttl_seconds: int, workspace_id: str | None = None, purpose="execution"):
    return await auth_keys.mint_turn_key_details(
        user_id=ORG.user_id, org_id=ORG.org_id, correlation_id="r", ttl_seconds=ttl_seconds,
        settings=TurnQueueSettings(), purpose=purpose, workspace_id=workspace_id,
    )


def _records(caplog, name: str, event: str) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == name and record.levelno == logging.WARNING and getattr(record, "event", None) == event
    ]


# ── The mint ────────────────────────────────────────────────────────────────


async def test_a_key_past_auths_cap_is_minted_again_with_the_untimed_lifetime(
    time_limit, monkeypatch, caplog, owned_logger
):
    logged = owned_logger("cowork.turnqueue.auth_keys", level=logging.WARNING)
    auth = _Auth(3600)
    _serve(monkeypatch, auth)

    minted = await _mint(ttl_seconds=TurnQueueSettings().turn_key_lifetime_seconds)

    assert minted == auth_keys.MintedTurnKey(key=KEY, prefix="mdb_pfx", ttl_seconds=1200)
    assert len(auth.posts) == 2
    assert 6000 - 60 < auth.ahead(0) <= 6000
    # The second post asks for the untimed 20 minutes, with nothing else changed.
    assert 1200 - 60 < auth.ahead(1) <= 1200
    assert {k: v for k, v in auth.posts[1].items() if k != "expiry_date"} == {
        k: v for k, v in auth.posts[0].items() if k != "expiry_date"
    }
    [record] = _records(caplog, "cowork.turnqueue.auth_keys", "turn_key_untimed_fallback")
    assert (record.purpose, record.requested_ttl_seconds, record.ttl_seconds) == ("execution", 6000, 1200)
    assert record.reason == "expiry_date exceeds the max turn-key TTL of 3600s"
    line = json.loads(logged.output().strip())
    assert {k: line.get(k) for k in ("log_level", "event", "purpose", "requested_ttl_seconds", "ttl_seconds")} == {
        "log_level": "WARNING", "event": "turn_key_untimed_fallback", "purpose": "execution",
        "requested_ttl_seconds": 6000, "ttl_seconds": 1200,
    }
    assert SECRET not in logged.output() and KEY not in logged.output()


async def test_a_key_within_auths_cap_is_minted_once_for_its_whole_lifetime(time_limit, monkeypatch, caplog):
    auth = _Auth(6600)
    _serve(monkeypatch, auth)

    minted = await _mint(ttl_seconds=6000)

    assert minted.ttl_seconds == 6000
    assert len(auth.posts) == 1
    assert 6000 - 60 < auth.ahead(0) <= 6000
    assert _records(caplog, "cowork.turnqueue.auth_keys", "turn_key_untimed_fallback") == []


async def test_an_untimed_key_auth_refuses_is_unavailable_after_one_post(auth_env, monkeypatch):
    """Nothing shorter to fall back to: the turn fails as it did before."""
    auth = _Auth(600)
    _serve(monkeypatch, auth)

    with pytest.raises(ProductPermissionUnavailable):
        await _mint(ttl_seconds=1200)

    assert len(auth.posts) == 1


@pytest.mark.parametrize(
    ("answer", "raised"),
    [
        (httpx.Response(403, json={"code": "permission_denied", "detail": "no"}), ProductPermissionDenied),
        (httpx.Response(500, json={"detail": "boom"}), ProductPermissionUnavailable),
        (httpx.Response(503, text="upstream down"), ProductPermissionUnavailable),
        (httpx.ReadTimeout("slow"), ProductPermissionUnavailable),
        (httpx.Response(400, json={"purpose": ["not a valid choice"]}), ProductPermissionUnavailable),
        (httpx.Response(400, json={"detail": "expiry_date must be in the future"}), ProductPermissionUnavailable),
    ],
    ids=["denied", "server-error", "unavailable", "timeout", "other-400", "other-expiry-400"],
)
async def test_any_other_answer_to_a_long_key_sends_no_second_mint(time_limit, monkeypatch, answer, raised):
    auth = _Auth(3600, first=[answer])
    _serve(monkeypatch, auth)

    with pytest.raises(raised):
        await _mint(ttl_seconds=6000)

    assert len(auth.posts) == 1


async def test_an_auth_that_checks_the_length_first_still_refuses_the_person(time_limit, monkeypatch):
    """Such an auth answers the long key 400 before it checks the person, so
    the untimed post is what it refuses with 403. No mint follows that 403."""
    auth = _Auth(3600, first=[None, httpx.Response(403, json={"code": "permission_denied"})])
    _serve(monkeypatch, auth)

    with pytest.raises(ProductPermissionDenied):
        await _mint(ttl_seconds=6000)

    assert len(auth.posts) == 2


@pytest.mark.parametrize("workspace_first", [True, False])
async def test_both_refusals_fall_back_once_each_in_either_order(
    time_limit, monkeypatch, forgotten, workspace_first
):
    gone = httpx.Response(404, json={"code": "workspace_not_found"})
    auth = _Auth(3600, first=[gone] if workspace_first else [None, gone])
    _serve(monkeypatch, auth)

    minted = await _mint(ttl_seconds=6000, workspace_id="ws-stale")

    assert minted.ttl_seconds == 1200
    assert len(auth.posts) == 3
    assert ["workspace_id" in post for post in auth.posts] == ([True, False, False] if workspace_first
                                                                else [True, True, False])
    assert auth.ahead(2) <= 1200
    assert forgotten == [{"org_id": ORG.org_id, "user_id": ORG.user_id, "workspace_id": "ws-stale"}]


async def test_a_refusal_that_repeats_after_both_fallbacks_ends_at_three_posts(time_limit, monkeypatch, forgotten):
    gone = httpx.Response(404, json={"code": "workspace_not_found"})
    auth = _Auth(3600, first=[gone, None, _length_refusal(600)])
    _serve(monkeypatch, auth)

    with pytest.raises(ProductPermissionUnavailable):
        await _mint(ttl_seconds=6000, workspace_id="ws-stale")

    assert len(auth.posts) == 3


async def test_the_connector_key_falls_back_with_the_turn_key(time_limit, monkeypatch):
    """Without it, an auth with the shorter cap drops every MCP connection from the turn."""
    auth = _Auth(3600)
    _serve(monkeypatch, auth)

    key = await prod._mint_connector_key(
        org_id=ORG.org_id, user_id=ORG.user_id, correlation_id="r", settings=TurnQueueSettings(), workspace_id=None,
    )

    assert key == KEY
    assert [post["purpose"] for post in auth.posts] == ["connector_token", "connector_token"]
    assert auth.ahead(1) <= 1200


# ── The job ─────────────────────────────────────────────────────────────────


def _producer_turn(monkeypatch) -> FakeRedis:
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply("turn_completed", {}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    monkeypatch.setattr(prod, "_mint_oauth_block", AsyncMock(return_value=None))
    monkeypatch.setattr(prod, "revoke_turn_key", AsyncMock())
    return fake


def _job(fake: FakeRedis) -> dict:
    [(_, fields)] = fake.added
    return json.loads(fields["payload"])["params"]


async def test_a_turn_whose_key_auth_shortened_completes_untimed(time_limit, monkeypatch):
    """The regression: an auth that caps keys at 3,600 s, under a 3,600 s time
    limit whose key asks for 6,000 s. Before the fallback this turn failed with
    permission_unavailable before anything was queued."""
    auth = _Auth(3600)
    _serve(monkeypatch, auth)
    fake = _producer_turn(monkeypatch)

    events = await _drain(prod.stream_remote_replies(
        conversation_id="conv-1", org_id=None, user_id=None, input_text="hi", model="m",
    ))

    assert [kind for kind, _ in events] == ["turn_completed"], events
    params = _job(fake)
    assert params["llm"]["api_key"] == KEY
    assert "max_turn_seconds" not in params["limits"]
    assert len(auth.posts) == 2 and auth.ahead(1) <= 1200


async def test_a_turn_within_auths_cap_keeps_its_time_limit(time_limit, monkeypatch):
    auth = _Auth(6600)
    _serve(monkeypatch, auth)
    fake = _producer_turn(monkeypatch)

    await _drain(prod.stream_remote_replies(
        conversation_id="conv-1", org_id=None, user_id=None, input_text="hi", model="m",
    ))

    assert _job(fake)["limits"]["max_turn_seconds"] == 3600
    assert len(auth.posts) == 1 and 6000 - 60 < auth.ahead(0) <= 6000


@pytest.mark.parametrize(("key_ttl", "time_limit_sent"), [(1200, None), (6000, 3600)])
async def test_a_gate_key_carries_its_lifetime_into_the_job(time_limit, monkeypatch, key_ttl, time_limit_sent):
    fake = _producer_turn(monkeypatch)
    monkeypatch.setattr(prod, "mint_turn_key_details", AsyncMock(side_effect=AssertionError("minted twice")))

    await _drain(prod.stream_remote_replies(
        conversation_id="conv-1", org_id=None, user_id=None, input_text="hi", model="m",
        correlation_id="r", llm={"provider": "minds-cloud", "api_key": "mdb_gate", "base_url": "http://gw/v1"},
        turn_key_ttl_seconds=key_ttl,
    ))

    assert _job(fake)["limits"].get("max_turn_seconds") == time_limit_sent


class _Everyday(NamedTuple):
    """One turn against an auth with ``cap`` seconds of key lifetime, and what it must do."""

    time_limit: int
    cap: int
    first: tuple = ()
    mcp: bool = False
    workspace: str | None = None
    #: completed, unavailable (turn_failed permission_unavailable) or denied (403).
    outcome: str = "completed"
    #: Mint posts per purpose.
    posts: dict = {"execution": 1}
    max_turn_seconds: int | None = None


_GONE = httpx.Response(404, json={"code": "workspace_not_found"})
_DENIED = httpx.Response(403, json={"code": "permission_denied"})

# The turns a web user starts every day, against the auth cap in production
# (6,600) and the old one (3,600). The fix may change only the rows that end
# "changed"; every other row answers the same at the base commit.
_EVERYDAY = {
    "untimed-new-auth": _Everyday(0, 6600),
    "untimed-old-auth": _Everyday(0, 3600),
    "timed-new-auth": _Everyday(3600, 6600, max_turn_seconds=3600),
    "timed-cap-just-covers": _Everyday(3600, 6300, max_turn_seconds=3600),
    "timed-stale-workspace-new-auth": _Everyday(3600, 6600, first=(_GONE,), workspace="ws-stale",
                                                posts={"execution": 2}, max_turn_seconds=3600),
    "timed-mcp-new-auth": _Everyday(3600, 6600, mcp=True, posts={"execution": 1, "connector_token": 1},
                                    max_turn_seconds=3600),
    "timed-denied-new-auth": _Everyday(3600, 6600, first=(_DENIED,), outcome="denied"),
    "untimed-auth-down": _Everyday(0, 6600, first=(httpx.Response(500, json={"detail": "x"}),),
                                   outcome="unavailable"),
    "timed-auth-timeout": _Everyday(3600, 6600, first=(httpx.ReadTimeout("slow"),), outcome="unavailable"),
    "timed-old-auth-changed": _Everyday(3600, 3600, posts={"execution": 2}),
    "timed-stale-workspace-old-auth-changed": _Everyday(3600, 3600, first=(_GONE,), workspace="ws-stale",
                                                        posts={"execution": 3}),
    "timed-mcp-old-auth-changed": _Everyday(3600, 3600, mcp=True,
                                            posts={"execution": 2, "connector_token": 2}),
    "timed-denied-old-auth-changed": _Everyday(3600, 3600, first=(None, _DENIED), outcome="denied",
                                               posts={"execution": 2}),
    "timed-auth-below-untimed-changed": _Everyday(3600, 600, outcome="unavailable", posts={"execution": 2}),
}


@pytest.mark.parametrize("case", list(_EVERYDAY.values()), ids=list(_EVERYDAY))
async def test_everyday_turns_against_each_auth_cap(auth_env, monkeypatch, forgotten, case):
    monkeypatch.setenv("COWORK_TURN_TIME_LIMIT_SECONDS", str(case.time_limit))
    auth = _Auth(case.cap, first=list(case.first))
    _serve(monkeypatch, auth)
    fake = FakeRedis(replies=[("scratchpad:reply:conv-1", _reply("turn_completed", {}))])
    monkeypatch.setattr(prod, "get_redis", lambda: fake)
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    monkeypatch.setattr(prod, "revoke_turn_key", AsyncMock())
    monkeypatch.setattr(prod, "list_active_connections",
                        AsyncMock(return_value=[{"engine": "notion", "name": "acme"}] if case.mcp else []))
    if case.workspace:
        from cowork.common.settings import user_settings

        real_settings = user_settings.get_user_settings
        monkeypatch.setattr(user_settings, "get_user_settings", lambda scope: real_settings(scope).model_copy(
            update={"hub_workspace_id": case.workspace}))

    try:
        events = await _drain(prod.stream_remote_replies(
            conversation_id="conv-1", org_id="o1", user_id="u1", input_text="hi", model="m",
        ))
        outcome = "completed" if events == [("turn_completed", {})] else events[-1][1].get("code")
        outcome = {"permission_unavailable": "unavailable"}.get(outcome, outcome)
    except ProductPermissionDenied:
        outcome = "denied"

    posts: dict[str, int] = {}
    for post in auth.posts:
        posts[post["purpose"]] = posts.get(post["purpose"], 0) + 1
    assert (outcome, posts) == (case.outcome, case.posts)
    if outcome == "completed":
        assert _job(fake)["limits"].get("max_turn_seconds") == case.max_turn_seconds


async def test_a_gate_key_without_its_lifetime_is_refused_before_anything_runs(monkeypatch):
    redis = AsyncMock(side_effect=AssertionError("queued"))
    monkeypatch.setattr(prod, "get_redis", redis)

    with pytest.raises(ValueError):
        await _drain(prod.stream_remote_replies(
            conversation_id="conv-1", org_id=None, user_id=None, input_text="hi", model="m",
            llm={"api_key": "mdb_gate"},
        ))

    redis.assert_not_called()


async def test_the_decisions_proxy_knows_the_key_as_long_as_auth_does(time_limit, monkeypatch):
    """Under a time limit the turn's lifetime is 6,000 s; a record that long
    would outlive the 1,200 s key auth actually minted."""
    admit = AsyncMock()
    monkeypatch.setattr(prod.decisions_proxy, "admit", admit)

    block = await prod._admit_decisions_proxy(
        object(), {"decisions_proxy": {"url": "http://proxy"}}, turn_key="mdb_gate", turn_key_ttl_seconds=1200,
        org_id=ORG.org_id, user_id=ORG.user_id, correlation_id="r",
    )

    assert block == {"decisions_proxy": {"url": "http://proxy"}}
    assert admit.await_args.kwargs["ttl_seconds"] == 1200


async def test_the_routing_gate_hands_on_the_lifetime_auth_gave_its_key(time_limit, monkeypatch):
    """The gate's own mint, against the same capped auth: it gets the untimed
    key where it used to raise permission_unavailable (503)."""
    import cowork.handlers.responses as responses
    from cowork.common.settings.user_settings import Provider
    from cowork.handlers.responses import ResponsesHandler

    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    monkeypatch.setattr(
        responses, "get_user_settings",
        lambda scope: SimpleNamespace(resolved_router_provider=Provider.MINDS_CLOUD, resolved_router_model="kimi",
                                      resolved_gate_model="mindshub_air", hub_workspace_id=""),
    )
    monkeypatch.setattr(responses, "provider_api_key", lambda settings, provider: None)
    auth = _Auth(3600)
    _serve(monkeypatch, auth)
    handler = ResponsesHandler()
    handler.scope = ORG

    _, turn_context = await handler._router_binding()

    assert turn_context["llm"]["api_key"] == KEY
    assert turn_context["turn_key_ttl_seconds"] == 1200
    assert len(auth.posts) == 2


# ── Why permission_unavailable ──────────────────────────────────────────────


class _Settings(TurnQueueSettings):
    auth_internal_base_url: str = "http://auth.internal"
    auth_internal_secret: str = SECRET
    datasource_producer_key_id: str = "producer-v1"
    datasource_producer_key: str = PRODUCER_SECRET


def _answering(answer):
    def handler(request):
        if isinstance(answer, Exception):
            raise answer
        return answer
    return handler


async def _authorize(settings):
    return await product_permissions.has_product_permission(ORG, "product.execute")


async def _turn_key(settings):
    return await auth_keys.mint_turn_key(
        user_id=ORG.user_id, org_id=ORG.org_id, correlation_id="r", ttl_seconds=1200, settings=settings,
    )


async def _turn_key_details(settings):
    return await auth_keys.mint_turn_key_details(
        user_id=ORG.user_id, org_id=ORG.org_id, correlation_id="r", ttl_seconds=1200, settings=settings,
    )


async def _datasources(settings):
    return await auth_keys.list_verified_datasource_connections(
        org_id=ORG.org_id, user_id=ORG.user_id, turn_key_id="mdb_pfx", settings=settings,
    )


async def _datasource_grants(settings):
    return await auth_keys.register_datasource_grants(
        user_id=ORG.user_id, org_id=ORG.org_id, correlation_id="r", turn_key_id="mdb_pfx", connection_ids=[7],
        settings=settings,
    )


async def _custom_connections(settings):
    return await auth_keys.list_custom_api_connections(
        org_id=ORG.org_id, user_id=ORG.user_id, turn_key_id="mdb_pfx", settings=settings,
    )


async def _custom_grants(settings):
    return await auth_keys.register_custom_api_grants(
        user_id=ORG.user_id, org_id=ORG.org_id, correlation_id="r", turn_key_id="mdb_pfx", connection_ids=[7],
        settings=settings,
    )


_UNCONFIGURED = {"auth_internal_base_url": "", "auth_internal_secret": ""}
_NO_PRODUCER_KEY = {"datasource_producer_key": ""}
_KEYS = auth_keys.TURN_KEYS_ROUTE
_DS = auth_keys.DATASOURCE_CONNECTIONS_ROUTE
_DSG = auth_keys.DATASOURCE_GRANTS_ROUTE
_CC = auth_keys.CUSTOM_CONNECTIONS_ROUTE
_CCG = auth_keys.CUSTOM_GRANTS_ROUTE
_AUTHZ = product_permissions.AUTHORIZE_ROUTE

# Every raise of ProductPermissionUnavailable: the call, what auth answers (None:
# auth is never asked), the settings it runs with, and the fields its line carries.
_CAUSES = [
    ("authorize-unconfigured", _authorize, None, _UNCONFIGURED, {"route": _AUTHZ, "reason": "not_configured"}),
    ("authorize-500", _authorize, httpx.Response(500, json={"detail": "boom"}), {},
     {"route": _AUTHZ, "status_code": 500, "reason": "boom"}),
    ("authorize-timeout", _authorize, httpx.ReadTimeout("slow"), {}, {"route": _AUTHZ, "error_type": "ReadTimeout"}),
    ("authorize-malformed", _authorize, httpx.Response(200, json={"allowed": "yes"}), {},
     {"route": _AUTHZ, "error_type": "ValidationError"}),
    ("mint-unconfigured", _turn_key, None, _UNCONFIGURED, {"route": _KEYS, "reason": "not_configured"}),
    ("mint-500", _turn_key, httpx.Response(500, json={"code": "db_down", "detail": "x"}), {},
     {"route": _KEYS, "status_code": 500, "reason": "db_down"}),
    ("mint-timeout", _turn_key, httpx.ConnectTimeout("slow"), {}, {"route": _KEYS, "error_type": "ConnectTimeout"}),
    ("mint-not-json", _turn_key, httpx.Response(201, text="<html>"), {}, {"route": _KEYS, "error_type": "JSONDecodeError"}),
    ("mint-no-key", _turn_key, httpx.Response(201, json={"key": " ", "prefix": "mdb_pfx"}), {},
     {"route": _KEYS, "reason": "answer_without_key"}),
    ("mint-length-untimed", _turn_key, _length_refusal(600), {},
     {"route": _KEYS, "status_code": 400, "reason": "expiry_date exceeds the max turn-key TTL of 600s"}),
    ("mint-no-prefix", _turn_key_details, httpx.Response(201, json={"key": KEY}), {},
     {"route": _KEYS, "reason": "answer_without_prefix"}),
    ("datasources-no-producer-key", _datasources, None, _NO_PRODUCER_KEY,
     {"route": _DS, "reason": "producer_key_not_configured"}),
    ("datasources-502", _datasources, httpx.Response(502, text="bad gateway"), {}, {"route": _DS, "status_code": 502}),
    ("datasources-no-items", _datasources, httpx.Response(200, json={"rows": []}), {},
     {"route": _DS, "reason": "answer_without_items"}),
    ("datasource-grants-no-producer-key", _datasource_grants, None, _NO_PRODUCER_KEY,
     {"route": _DSG, "reason": "producer_key_not_configured"}),
    ("datasource-grants-409", _datasource_grants, httpx.Response(409, json={"code": "stale_grant"}), {},
     {"route": _DSG, "status_code": 409, "reason": "stale_grant"}),
    ("datasource-grants-not-object", _datasource_grants, httpx.Response(201, json=[]), {},
     {"route": _DSG, "reason": "answer_not_an_object"}),
    ("custom-no-producer-key", _custom_connections, None, _NO_PRODUCER_KEY,
     {"route": _CC, "reason": "producer_key_not_configured"}),
    ("custom-read-error", _custom_connections, httpx.ReadError("reset"), {}, {"route": _CC, "error_type": "ReadError"}),
    ("custom-no-items", _custom_connections, httpx.Response(200, json={"items": "x"}), {},
     {"route": _CC, "reason": "answer_without_items"}),
    ("custom-grants-no-producer-key", _custom_grants, None, _NO_PRODUCER_KEY,
     {"route": _CCG, "reason": "producer_key_not_configured"}),
    ("custom-grants-401", _custom_grants, httpx.Response(401, json={"detail": "who are you"}), {},
     {"route": _CCG, "status_code": 401, "reason": "who are you"}),
    ("custom-grants-not-object", _custom_grants, httpx.Response(201, json="ok"), {},
     {"route": _CCG, "reason": "answer_not_an_object"}),
]


@pytest.mark.parametrize(("call", "answer", "overrides", "fields"),
                         [case[1:] for case in _CAUSES], ids=[case[0] for case in _CAUSES])
async def test_each_permission_unavailable_logs_its_route_and_cause_and_no_secret(
    monkeypatch, caplog, owned_logger, call, answer, overrides, fields
):
    """Through each real raise: the line the console prints names the route and
    the status or exception class, and holds neither internal secret nor the key."""
    settings = _Settings(**overrides)
    monkeypatch.setattr(product_permissions, "TurnQueueSettings", lambda: settings)

    def unexpected(request):
        raise AssertionError("auth was asked")

    _serve(monkeypatch, _answering(answer) if answer is not None else unexpected)
    logged = owned_logger("cowork.services.product_permissions", level=logging.WARNING)

    with pytest.raises(ProductPermissionUnavailable):
        await call(settings)

    [record] = _records(caplog, "cowork.services.product_permissions", "permission_unavailable")
    for name, value in fields.items():
        assert getattr(record, name) == value, name
    for absent in {"status_code", "error_type", "reason"} - set(fields):
        assert not hasattr(record, absent), absent
    [line] = [json.loads(text) for text in logged.output().splitlines()]
    assert line["log_level"] == "WARNING" and line["event"] == "permission_unavailable"
    assert {name: line.get(name) for name in fields} == fields
    for secret in (SECRET, PRODUCER_SECRET, KEY):
        assert secret not in logged.output()
        assert not any(secret in repr(value) for value in vars(record).values())


def test_auths_detail_is_cut_to_two_hundred_characters():
    response = httpx.Response(400, json={"detail": "x" * 500})
    assert product_permissions.auth_refusal_reason(response=response) == "x" * 200
