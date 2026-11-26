"""A member's custom MCP servers in a cloud turn's job: listed by auth for the turn
key's owner, with grants bound to that key, and each switched-on definition as the
string auth stored. Auth is faked at the transport with the answers of its
published producer contract, so the real helpers, retry and parsing run."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from anton.cloud_turn.contract import TurnRequestV1
from anton.cloud_turn.custom_mcp import build_custom_mcp_tools

import test_producer as base
from cowork.common.settings import user_settings
from cowork.common.settings.app_settings import TurnQueueSettings, get_app_settings
from cowork.turnqueue import auth_keys
from cowork.turnqueue import producer as prod

FakeRedis = base.FakeRedis
_drain = base._drain
_reply_for = base._reply_for
_stub_llm_mint = base._stub_llm_mint

pytestmark = pytest.mark.usefixtures("granted_product_permissions")

_CONTRACT_FILE = Path(__file__).resolve().parents[1] / "fixtures" / "custom_mcp_producer_v1.json"
#: Copied byte for byte from auth's docs/contracts/custom-mcp-producer-v1.json at this
#: commit. Copy again, and update both, when auth's file changes.
_CONTRACT_SOURCE = "7841f91d49b75ba5f6d26221039de35f85d37fc9"
_CONTRACT_SHA256 = "53b3cb0d103d3764eefc8c56afd5d33bf5481217b732486bb3b06811e5d3c5e6"
_CONTRACT = json.loads(_CONTRACT_FILE.read_text(encoding="utf-8"))

USER = _CONTRACT["listing_request"]["user_id"]
ORG = _CONTRACT["listing_request"]["organization_id"]
CORRELATION = _CONTRACT["grant_request"]["correlation_id"]
#: The prefix test_producer's mint stub returns; the contract writes a placeholder.
PREFIX = "turnkey"
SECOND = "cm-" + "fedcba9876543210" * 2
_LISTING_PATH = "/internal/custom-mcp/connections/"
_GRANT_PATH = "/internal/custom-mcp/turn-grants/"


def _listing(*extra_items: dict) -> dict:
    """The contract's listing answer, with more servers appended."""
    answer = copy.deepcopy(_CONTRACT["listing_response"])
    answer["items"].extend(extra_items)
    return answer


def _second_server(**overrides) -> dict:
    """Another server shaped like the contract's item, at id 2 and version 4."""
    item = {**copy.deepcopy(_CONTRACT["listing_response"]["items"][0]), "id": 2, "name": SECOND,
            "label": "Docs", "credential_version": 4}
    item.update(overrides)
    return item


def _grant_answer(request_body: dict, **overrides) -> dict:
    """The contract's grant answer for the requested connections, bound to this turn's prefix."""
    answer = copy.deepcopy(_CONTRACT["grant_response"])
    template = answer["grants"][0]
    answer["grants"] = [{**template, "connection_id": cid, "credential_version": 1 if cid == 1 else 5}
                        for cid in request_body["connection_ids"]]
    answer["turn_key_id"] = PREFIX
    answer.update(overrides)
    return answer


@pytest.fixture()
def auth(monkeypatch):
    """Org tenancy, the custom MCP setting on alone, a producer key; auth answered at the transport.

    ``listing`` and ``grant`` hold a list of answers each, one per call: an
    ``httpx.Response``, an exception to raise, or a callable taking the request.
    """
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_TURN_CUSTOM_MCP_ENABLED", "true")
    monkeypatch.delenv("COWORK_TURN_DATASOURCE_ENABLED", raising=False)
    monkeypatch.delenv("COWORK_TURN_CUSTOM_API_ENABLED", raising=False)
    monkeypatch.setenv("COWORK_TURN_DATASOURCE_PRODUCER_KEY_ID", "producer")
    monkeypatch.setenv("COWORK_TURN_DATASOURCE_PRODUCER_KEY", "producer-secret")
    monkeypatch.setenv("COWORK_TURN_AUTH_INTERNAL_BASE_URL", "http://auth.internal")
    get_app_settings.cache_clear()
    state = {
        "redis": FakeRedis(workspace_mode=None, replies=[
            ("scratchpad:reply:conv-1", _reply_for(
                CORRELATION, "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"})),
            ("scratchpad:reply:conv-1", _reply_for(CORRELATION, "turn_completed", {})),
        ]),
        "listing": [httpx.Response(200, json=_listing())],
        "grant": [lambda request: httpx.Response(200, json=_grant_answer(json.loads(request.content)))],
        "calls": [],
        "timeouts": [],
    }

    def handle(request: httpx.Request) -> httpx.Response:
        state["calls"].append(request)
        queue = state["listing"] if request.url.path == _LISTING_PATH else state["grant"]
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer(request) if callable(answer) else answer

    real_client = httpx.AsyncClient

    def client(**kwargs):
        state["timeouts"].append(kwargs.get("timeout"))
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(auth_keys.httpx, "AsyncClient", client)
    monkeypatch.setattr(prod, "get_redis", lambda: state["redis"])
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: CORRELATION)
    monkeypatch.setattr(prod, "_mint_oauth_block", AsyncMock(return_value=None))
    monkeypatch.setattr(user_settings, "get_user_settings",
                        lambda scope: SimpleNamespace(limits_enabled=False, resolved_planning_model="m",
                                                      hub_workspace_id=""))
    state["log"] = MagicMock()
    monkeypatch.setattr(prod, "logger", state["log"])
    yield state
    get_app_settings.cache_clear()


async def _run(state, **kwargs):
    """Run one turn; return its job params, or None and the yields when nothing was queued."""
    call = {"conversation_id": "conv-1", "org_id": ORG, "user_id": USER, "input_text": "hi", "model": "m"}
    call.update(kwargs)
    items = await _drain(prod.stream_remote_replies(**call))
    if not state["redis"].added:
        return None, items
    return json.loads(state["redis"].added[0][1]["payload"])["params"], items


def _events(state, *, level: str = "warning") -> list[dict]:
    """The ``extra`` of every leave-out event the producer logged at ``level``."""
    return [
        call.kwargs["extra"] for call in getattr(state["log"], level).call_args_list
        if call.kwargs.get("extra", {}).get("event") == "turn_custom_mcp_left_out"
    ]


def _paths(state) -> list[str]:
    return [request.url.path for request in state["calls"]]


# ── the contract ─────────────────────────────────────────────────────────────


def test_the_fixture_is_auths_published_producer_file():
    assert hashlib.sha256(_CONTRACT_FILE.read_bytes()).hexdigest() == _CONTRACT_SHA256, (
        f"re-copy auth's custom-mcp-producer-v1.json from {_CONTRACT_SOURCE} or newer and update the pin"
    )


async def test_the_block_built_from_the_contracts_answers_is_the_contracts_block(auth):
    params, _ = await _run(auth)

    assert params["custom_mcp"] == _CONTRACT["turn_custom_mcp"]
    listing_request = next(r for r in auth["calls"] if r.url.path == _LISTING_PATH)
    assert dict(listing_request.url.params) == {**_CONTRACT["listing_request"], "turn_key_id": PREFIX}
    grant_request = json.loads(next(r for r in auth["calls"] if r.url.path == _GRANT_PATH).content)
    assert grant_request == {**_CONTRACT["grant_request"], "turn_key_id": PREFIX}
    assert listing_request.headers["X-Datasource-Service-Key-Id"] == "producer"
    assert auth["timeouts"] == [5.0, 5.0]
    assert _events(auth) == []


async def test_the_pod_parses_and_builds_the_block_this_producer_queues(auth):
    """The producer's real block through the installed anton's own parser and tool builder."""
    auth["listing"] = [httpx.Response(200, json=_listing(_second_server()))]

    params, _ = await _run(auth)
    request = TurnRequestV1.from_json(json.dumps({
        "protocol_version": 1, "conversation_id": "conv-1", "input": "hi", "correlation_id": CORRELATION,
        "llm": params["llm"], "custom_mcp": params["custom_mcp"],
    }))

    assert request.custom_mcp is not None, "anton dropped the block this producer built"
    assert [(c.connection_id, c.name, c.label, c.credential_version, [t.name for t in c.tools])
            for c in request.custom_mcp.connections] == [
        (1, _CONTRACT["turn_custom_mcp"]["connections"][0]["name"], "Issues", 1, ["empty_schema"]),
        (2, SECOND, "Docs", 5, ["empty_schema"]),
    ]
    tools = build_custom_mcp_tools(
        request.custom_mcp, gateway_base_url="https://minds.example/v1",
        turn_key=params["llm"]["api_key"], correlation_id=CORRELATION,
    )
    assert len(tools) == 2


# ── what the block carries ───────────────────────────────────────────────────


@pytest.mark.parametrize(("switches", "on"), [
    ({"COWORK_TURN_CUSTOM_MCP_ENABLED": "true"}, True),
    ({}, False),
], ids=["custom MCP alone", "all three off"])
async def test_the_custom_mcp_switch_alone_mints_the_prefix_and_sends_the_block(auth, monkeypatch, switches, on):
    monkeypatch.delenv("COWORK_TURN_CUSTOM_MCP_ENABLED", raising=False)
    for name, value in switches.items():
        monkeypatch.setenv(name, value)
    get_app_settings.cache_clear()

    params, _ = await _run(auth)

    assert TurnQueueSettings().needs_turn_key_id is on
    assert ("custom_mcp" in params) is on


async def test_the_block_carries_the_grants_versions_and_the_served_strings_minus_disabled_ones(auth, monkeypatch):
    third = "cm-" + "0" * 32
    served = '{"description":"été","input_schema":{"type":"object"},"name":"z"}'
    auth["listing"] = [httpx.Response(200, json=_listing(_second_server(tools=[served]),
                                                         _second_server(id=3, name=third)))]
    monkeypatch.setattr(prod, "_mint_oauth_block",
                        AsyncMock(return_value={"connections": [{"engine": "hubspot", "name": "main"}]}))

    params, _ = await _run(auth, disabled=[{"engine": "custom_mcp", "name": third},
                                           {"engine": "hubspot", "name": SECOND}])

    assert json.loads(next(r for r in auth["calls"] if r.url.path == _GRANT_PATH).content)["connection_ids"] == [1, 2]
    second = params["custom_mcp"]["connections"][1]
    assert (second["connection_id"], second["credential_version"], second["tools"]) == (2, 5, [served])
    assert [c["engine"] for c in params["oauth"]["connections"]] == ["hubspot"]


async def test_a_member_at_the_server_cap_gets_every_server(auth):
    auth["listing"] = [httpx.Response(200, json=_listing(*_more_servers(24)))]

    params, items = await _run(auth)

    assert items[-1][0] == "turn_completed"
    assert [c["connection_id"] for c in params["custom_mcp"]["connections"]] == list(range(1, 26))


async def test_a_member_with_no_server_gets_no_block_and_no_event(auth):
    auth["listing"] = [httpx.Response(200, json={"items": []})]

    params, _ = await _run(auth)

    assert "custom_mcp" not in params
    assert _paths(auth).count(_LISTING_PATH) == 1
    assert _GRANT_PATH not in _paths(auth)
    assert _events(auth) == []


def _definition(index: int, size: int) -> str:
    """A definition of exactly ``size`` UTF-8 bytes, padded mostly with three-byte characters."""
    head = json.dumps({"description": "", "input_schema": {"type": "object"}, "name": f"t{index}"},
                      separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    room = size - len(head.encode("utf-8"))
    padding = "文" * (room // 3) + "x" * (room % 3)
    return head.replace('"description":""', '"description":"' + padding + '"')


def _sixteen_kib_tools(extra_bytes: int = 0) -> list[str]:
    """16 definitions of 16 KiB: anton's block total exactly, plus ``extra_bytes`` on the last."""
    tools = [_definition(i, 16 * 1024) for i in range(16)]
    tools[-1] = _definition(15, 16 * 1024 + extra_bytes)
    return tools


async def test_a_block_at_antons_byte_total_is_sent_measured_unescaped(auth):
    tools = _sixteen_kib_tools()
    auth["listing"] = [httpx.Response(200, json={"items": [{**_listing()["items"][0], "tools": tools}]})]

    params, _ = await _run(auth)

    assert params["custom_mcp"]["connections"][0]["tools"] == tools
    assert sum(len(t.encode("utf-8")) for t in tools) == 262144
    assert len(json.dumps(tools).encode("utf-8")) > 262144
    assert _events(auth) == []


@pytest.mark.parametrize(
    "tools",
    [_sixteen_kib_tools(extra_bytes=1), [_definition(i, 100) for i in range(101)]],
    ids=["one byte past the total", "101 tools"],
)
async def test_a_block_past_either_anton_total_is_left_out(auth, tools):
    auth["listing"] = [httpx.Response(200, json={"items": [{**_listing()["items"][0], "tools": tools}]})]

    params, _ = await _run(auth)

    assert "custom_mcp" not in params
    assert _GRANT_PATH not in _paths(auth)
    assert [event["reason"] for event in _events(auth)] == ["over_backstop"]


# ── a grant answer that contradicts the turn ─────────────────────────────────


@pytest.mark.parametrize(("field", "wrong"), [
    ("user_id", "99999999-9999-4999-8999-999999999999"),
    ("organization_id", "99999999-9999-4999-8999-999999999999"),
    ("turn_key_id", "other"),
    ("correlation_id", "99999999-9999-4999-8999-999999999999"),
    ("audience", "custom-api"),
    ("purpose", "fetch"),
    ("grants", []),
])
async def test_a_grant_answer_bound_to_anything_else_fails_the_turn(auth, field, wrong):
    auth["grant"] = [lambda request: httpx.Response(
        200, json=_grant_answer(json.loads(request.content), **{field: wrong}))]

    params, items = await _run(auth)

    assert params is None
    assert items[-1][0] == "turn_failed"


async def test_a_refused_grant_answer_logs_its_route_and_reason(auth, caplog, owned_logger):
    """The line the console prints for a grant answer that fails the turn, as for a custom API grant."""
    logged = owned_logger("cowork.services.product_permissions", level=logging.WARNING)
    auth["grant"] = [lambda request: httpx.Response(
        200, json=_grant_answer(json.loads(request.content), turn_key_id="other"))]

    params, items = await _run(auth)

    assert params is None and items[-1][0] == "turn_failed"
    reason = "grant response binding does not match this turn"
    [record] = [
        r for r in caplog.records
        if r.name == "cowork.services.product_permissions" and getattr(r, "event", None) == "permission_unavailable"
    ]
    assert (record.route, record.reason) == (auth_keys.CUSTOM_MCP_GRANTS_ROUTE, reason)
    line = json.loads(logged.output().strip())
    assert (line["event"], line["route"], line["reason"]) == ("permission_unavailable", _GRANT_PATH, reason)


# ── leave-outs: the turn runs, one event names why ───────────────────────────


def _deep(depth: int) -> bytes:
    return b'{"items":' + b"[" * depth + b"]" * depth + b"}"


def _deep_error(depth: int) -> bytes:
    """An error body whose code nests ``depth`` arrays deep, past what the JSON parser can read."""
    return b'{"code":' + b"[" * depth + b"]" * depth + b"}"


def _more_servers(count: int) -> list[dict]:
    """``count`` more servers shaped like the contract's item, at ids 2 and up."""
    return [_second_server(id=index, name=f"cm-{index:032x}") for index in range(2, count + 2)]


def _over_backstop() -> dict:
    big = '{"description":"' + "x" * (600 * 1024) + '","input_schema":{"type":"object"},"name":"big"}'
    return _listing(_second_server(tools=[big, big.replace('"big"', '"big2"')]))


@pytest.mark.parametrize(("case", "listing", "grant", "reason", "listing_calls"), [
    ("listing times out", [httpx.ReadTimeout("slow")], None, "listing_failed", 1),
    ("listing answers 500", [httpx.Response(500)], None, "listing_failed", 1),
    ("listing answers 503 twice", [httpx.Response(503), httpx.Response(503)], None, "listing_failed", 2),
    ("listing names the key otherwise", [httpx.Response(200, json={"connections": []})], None,
     "listing_malformed", 1),
    ("a tool is an object", [httpx.Response(200, json=_listing(_second_server(tools=[{"name": "x"}])))], None,
     "listing_malformed", 1),
    # A JSON escape of a lone surrogate parses to a string that does not encode.
    ("a tool holds a lone surrogate", [httpx.Response(200, content=json.dumps(_listing(
        _second_server(tools=["LONE"]))).replace("LONE", "\\ud800").encode())], None,
     "listing_malformed", 1),
    ("listing nested 100,000 deep", [httpx.Response(200, content=_deep(100_000))], None, "listing_failed", 1),
    ("listing's 503 nested too deep", [httpx.Response(503, content=_deep_error(10_000))] * 2, None,
     "listing_failed", 2),
    ("26 servers", [httpx.Response(200, json=_listing(*_more_servers(25)))], None, "listing_malformed", 1),
    ("listing answers a list", [httpx.Response(200, json=[])], None, "listing_failed", 1),
    ("a 503 from a proxy is retried", [httpx.Response(503, text="<html>"), httpx.Response(503, text="<html>")], None,
     "listing_failed", 2),
    ("auth's other 503 is retried", [httpx.Response(503, json={"code": "storage_unavailable"})] * 2, None,
     "listing_failed", 2),
    ("over the backstop", [httpx.Response(200, json=_over_backstop())], None, "over_backstop", 1),
    ("grant answers 429", None, [httpx.Response(429, json={"code": "quota_exceeded"})], "grant_quota", 1),
    ("grant answers 409", None, [httpx.Response(409, json={"code": "conflict"})], "grant_conflict", 1),
    ("grant answers 404", None, [httpx.Response(404, json={"code": "capability_denied"})], "grant_refused", 1),
    ("grant answers a list", None, [httpx.Response(200, json=[])], "grant_refused", 1),
    ("grant's 404 nested too deep", None, [httpx.Response(404, content=_deep_error(10_000))], "grant_refused", 1),
])
async def test_a_turn_runs_without_the_block_and_logs_why(auth, case, listing, grant, reason, listing_calls):
    if listing is not None:
        auth["listing"] = listing
    if grant is not None:
        auth["grant"] = grant

    params, items = await _run(auth)

    assert params is not None and "custom_mcp" not in params, case
    assert items[-1][0] == "turn_completed"
    assert [(e["reason"], e["conversation_id"]) for e in _events(auth)] == [(reason, "conv-1")]
    assert _paths(auth).count(_LISTING_PATH) == listing_calls
    if grant is None:
        assert _GRANT_PATH not in _paths(auth)


@pytest.mark.parametrize("call", ["listing", "grant"])
async def test_auths_switch_off_leaves_the_block_out_at_info_without_a_retry(auth, call):
    auth[call] = [httpx.Response(503, json={"code": "disabled", "detail": "off", "request_id": "r"})]

    params, _ = await _run(auth)

    assert "custom_mcp" not in params
    assert _paths(auth).count(_LISTING_PATH if call == "listing" else _GRANT_PATH) == 1
    assert [e["reason"] for e in _events(auth, level="info")] == ["auth_disabled"]
    assert _events(auth) == []


async def test_with_custom_mcp_off_and_custom_api_on_nothing_asks_auth_about_custom_mcp(auth, monkeypatch):
    monkeypatch.setenv("COWORK_TURN_CUSTOM_MCP_ENABLED", "false")
    monkeypatch.setenv("COWORK_TURN_CUSTOM_API_ENABLED", "true")
    get_app_settings.cache_clear()

    params, _ = await _run(auth)

    assert TurnQueueSettings().needs_turn_key_id is True
    assert "custom_mcp" not in params
    assert _LISTING_PATH not in _paths(auth) and _GRANT_PATH not in _paths(auth)
    assert _events(auth) == [] and _events(auth, level="info") == []


async def test_a_listing_that_fails_fast_once_is_tried_again_and_the_block_is_sent(auth):
    auth["listing"] = [httpx.ConnectError("refused"), httpx.Response(200, json=_listing())]

    params, _ = await _run(auth)

    assert params["custom_mcp"] == _CONTRACT["turn_custom_mcp"]
    assert _paths(auth).count(_LISTING_PATH) == 2
    assert _events(auth) == []


async def test_without_the_producer_key_the_block_is_left_out(auth, monkeypatch):
    monkeypatch.setenv("COWORK_TURN_DATASOURCE_PRODUCER_KEY", "")
    get_app_settings.cache_clear()

    params, _ = await _run(auth)

    assert "custom_mcp" not in params
    assert auth["calls"] == []
    assert [e["reason"] for e in _events(auth)] == ["no_producer_key"]


async def test_a_turn_with_a_key_but_no_prefix_logs_it_at_info(auth):
    """A channel turn hands the producer a minted key and no prefix: every such turn has it, so not a warning."""
    block = {"provider": "minds-cloud", "api_key": "mdb_x.secret", "base_url": "https://minds.internal/v1"}

    params, _ = await _run(auth, llm=prod.TurnLlm(block=block, turn_key_id=None, ttl_seconds=1200))

    assert "custom_mcp" not in params
    assert auth["calls"] == []
    assert [e["reason"] for e in _events(auth, level="info")] == ["no_turn_key_prefix"]
    assert _events(auth) == []


@pytest.mark.parametrize("trusted", [False, True], ids=["local tenancy", "personal partition"])
async def test_outside_org_mode_nothing_is_listed_and_nothing_logged(auth, monkeypatch, trusted):
    """A personal partition has an org id and a user, but is one person's, not a member's org turn."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    if trusted:
        monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "true")
    else:
        monkeypatch.delenv("COWORK_TRUST_IDENTITY_HEADERS", raising=False)
    get_app_settings.cache_clear()

    params, _ = await _run(auth)

    assert "custom_mcp" not in params
    assert auth["calls"] == []
    assert _events(auth) == []
