"""A member's custom API connections in a cloud turn's job: listed by auth for the
turn key's owner, with fetch grants bound to that key, never a field value."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import test_producer as base
from cowork.common.settings.app_settings import get_app_settings
from cowork.services.product_permissions import ProductPermissionUnavailable
from cowork.turnqueue import producer as prod

FakeRedis = base.FakeRedis
_drain = base._drain
_reply = base._reply
_stub_llm_mint = base._stub_llm_mint

pytestmark = pytest.mark.usefixtures("granted_product_permissions")


def _listed(connection_id, connector_id="acme_crm", name="Prod", status="verified"):
    """Shaped like auth's InternalCustomApiConnectionsView items."""
    return {"id": connection_id, "connector_id": connector_id, "name": name, "status": status,
            "credential_version": 1}


def _granted(versions, **overrides):
    """Shaped like auth's custom API turn-grant answer, for user u1 in org o1."""
    response = {
        "user_id": "u1", "organization_id": "o1", "turn_key_id": "turnkey", "correlation_id": "r",
        "audience": "custom-api", "purpose": "fetch",
        "grants": [{"id": 100 + c, "connection_id": c, "credential_version": v, "expires_at": "x"}
                   for c, v in versions.items()],
    }
    response.update(overrides)
    return response


@pytest.fixture()
def turn(monkeypatch):
    """Org tenancy, the custom connector setting on, a producer key; auth faked."""
    from cowork.common.settings import user_settings

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_TURN_CONNECTOR_DRAFTS_ENABLED", "true")
    monkeypatch.setenv("COWORK_TURN_DATASOURCE_PRODUCER_KEY_ID", "producer")
    monkeypatch.setenv("COWORK_TURN_DATASOURCE_PRODUCER_KEY", "producer-secret")
    get_app_settings.cache_clear()
    state = {"redis": FakeRedis(replies=[("scratchpad:reply:conv-1", _reply("turn_completed", {}))])}
    monkeypatch.setattr(prod, "get_redis", lambda: state["redis"])
    monkeypatch.setattr(prod, "_new_correlation_id", lambda: "r")
    monkeypatch.setattr(prod, "_mint_oauth_block", AsyncMock(return_value=None))
    monkeypatch.setattr(user_settings, "get_user_settings",
                        lambda scope: SimpleNamespace(resolved_planning_model="m", hub_workspace_id=""))
    state["list"] = AsyncMock(return_value=[_listed(5), _listed(3, "beta_api", "Ops")])
    state["grant"] = AsyncMock(return_value=_granted({3: 1, 5: 2}))
    monkeypatch.setattr(prod, "list_custom_api_connections", state["list"])
    monkeypatch.setattr(prod, "register_custom_api_grants", state["grant"])
    yield state
    get_app_settings.cache_clear()


async def _run(state, **kwargs):
    """Run one turn; return its job params, or None and the yields when nothing was queued."""
    call = {"conversation_id": "conv-1", "org_id": "o1", "user_id": "u1", "input_text": "hi", "model": "m"}
    call.update(kwargs)
    items = await _drain(prod.stream_remote_replies(**call))
    if not state["redis"].added:
        return None, items
    return json.loads(state["redis"].added[0][1]["payload"])["params"], items


async def test_the_members_connections_travel_by_id_with_the_grants_versions_and_the_turn_key(turn):
    params, _ = await _run(turn)

    assert params["custom_api"] == {
        "turn_key": params["llm"]["api_key"],
        "connections": [
            {"connection_id": 3, "connector_id": "beta_api", "name": "Ops", "credential_version": 1},
            {"connection_id": 5, "connector_id": "acme_crm", "name": "Prod", "credential_version": 2},
        ],
    }
    listed = turn["list"].await_args.kwargs
    assert (listed["org_id"], listed["user_id"], listed["turn_key_id"]) == ("o1", "u1", "turnkey")
    granted = turn["grant"].await_args.kwargs
    assert granted["connection_ids"] == [3, 5]
    assert (granted["turn_key_id"], granted["correlation_id"]) == ("turnkey", "r")


async def test_a_routed_turn_binds_to_the_key_the_gate_minted(turn, monkeypatch):
    for name in ("mint_turn_key", "mint_turn_key_details"):
        monkeypatch.setattr(prod, name, AsyncMock(side_effect=AssertionError(f"{name} minted a second key")))
    block = {"provider": "minds-cloud", "api_key": "mdb_gate.secret", "base_url": "https://minds.internal/v1"}
    turn["grant"].return_value = _granted({3: 1, 5: 2}, turn_key_id="mdb_gate")

    params, _ = await _run(turn, correlation_id="r", llm=block, turn_key_id="mdb_gate")

    assert turn["list"].await_args.kwargs["turn_key_id"] == "mdb_gate"
    assert params["custom_api"]["turn_key"] == "mdb_gate.secret"


async def test_a_turn_that_arrives_with_a_key_but_no_prefix_runs_without_them(turn):
    """A channel turn hands the producer a minted key and no prefix to bind grants to."""
    block = {"provider": "minds-cloud", "api_key": "mdb_x.secret", "base_url": "https://minds.internal/v1"}

    params, _ = await _run(turn, llm=block)

    assert "custom_api" not in params
    turn["list"].assert_not_awaited()


@pytest.mark.parametrize("case", ["setting off", "local tenancy", "personal partition", "no user"])
async def test_outside_a_members_org_turn_nothing_is_listed(turn, monkeypatch, case):
    """Each case keeps a turn key prefix (the datasource setting is on), so only its own gate stands."""
    kwargs = {}
    monkeypatch.setenv("COWORK_TURN_DATASOURCE_ENABLED", "true")
    monkeypatch.setattr(prod, "list_verified_datasource_connections", AsyncMock(return_value=[]))
    if case == "setting off":
        monkeypatch.setenv("COWORK_TURN_CONNECTOR_DRAFTS_ENABLED", "false")
    elif case == "local tenancy":
        monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
        monkeypatch.delenv("COWORK_TRUST_IDENTITY_HEADERS", raising=False)
    elif case == "personal partition":
        # An org id and a user, but one person's partition, not org mode.
        monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
        monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "true")
    else:
        kwargs["user_id"] = None
    get_app_settings.cache_clear()

    params, _ = await _run(turn, **kwargs)

    assert "custom_api" not in params
    turn["list"].assert_not_awaited()


async def test_the_conversations_disabled_connections_are_left_out(turn):
    turn["grant"].return_value = _granted({5: 2})

    params, _ = await _run(turn, disabled=[{"engine": "beta_api", "name": "Ops"}])

    assert turn["grant"].await_args.kwargs["connection_ids"] == [5]
    assert [c["connection_id"] for c in params["custom_api"]["connections"]] == [5]


async def test_past_the_cap_the_lowest_ids_are_kept(turn):
    turn["list"].return_value = [_listed(i, name=f"n{i}") for i in range(30, 0, -1)]
    turn["grant"].return_value = _granted({i: 1 for i in range(1, 26)})

    params, _ = await _run(turn)

    assert turn["grant"].await_args.kwargs["connection_ids"] == list(range(1, 26))
    assert len(params["custom_api"]["connections"]) == 25


@pytest.mark.parametrize("case", ["no producer key", "listing unavailable", "grant refused", "nothing listed"])
async def test_the_turn_runs_without_them_when_auth_cannot_give_them(turn, monkeypatch, case):
    if case == "no producer key":
        monkeypatch.setenv("COWORK_TURN_DATASOURCE_PRODUCER_KEY", "")
    elif case == "listing unavailable":
        turn["list"].side_effect = ProductPermissionUnavailable()
    elif case == "grant refused":
        turn["grant"].side_effect = ProductPermissionUnavailable()
    else:
        turn["list"].return_value = []

    params, _ = await _run(turn)

    assert params is not None and "custom_api" not in params
    if case in ("no producer key", "nothing listed"):
        turn["grant"].assert_not_awaited()


@pytest.mark.parametrize("field", ["user_id", "organization_id", "turn_key_id", "correlation_id", "audience", "purpose"])
async def test_a_grant_answer_bound_to_anything_else_fails_the_turn(turn, field):
    wrong = {"user_id": "u2", "organization_id": "o2", "turn_key_id": "other", "correlation_id": "r2",
             "audience": "datasource-gateway", "purpose": "execute"}[field]
    turn["grant"].return_value = _granted({3: 1, 5: 2}, **{field: wrong})

    params, items = await _run(turn)

    assert params is None
    assert items[-1][0] == "turn_failed"


@pytest.mark.parametrize("grants", [
    {3: 1}, {3: 1, 5: 2, 9: 1}, "not a list", [{"connection_id": 3}], [{"connection_id": 3, "credential_version": 0}],
], ids=["missing one", "an extra one", "no list", "no version", "zero version"])
async def test_a_grant_set_that_differs_from_the_request_fails_the_turn(turn, grants):
    response = _granted({}) if isinstance(grants, (str, list)) else _granted(grants)
    if isinstance(grants, (str, list)):
        response["grants"] = grants
    turn["grant"].return_value = response

    params, items = await _run(turn)

    assert params is None
    assert items[-1][0] == "turn_failed"


@pytest.mark.parametrize("listing", [
    [_listed(3, status="pending")], [_listed(3), _listed(3)], [_listed(0)], ["not an object"], [_listed(3, name="")],
], ids=["not verified", "repeated", "zero id", "not an object", "no name"])
async def test_a_listing_that_contradicts_itself_fails_the_turn(turn, listing):
    turn["list"].return_value = listing

    params, items = await _run(turn)

    assert params is None
    assert items[-1][0] == "turn_failed"
    turn["grant"].assert_not_awaited()


@pytest.mark.parametrize(("org_id", "user_id"), [("o1", "u2"), ("o2", "u1")], ids=["same org", "another org"])
async def test_each_member_gets_only_what_auth_lists_for_their_own_key(turn, org_id, user_id):
    """auth derives the owner from the turn key; the producer passes this turn's identity and adds nothing."""
    turn["list"].return_value = [_listed(9, "own_api", "Mine")]
    turn["grant"].return_value = _granted({9: 1}, user_id=user_id, organization_id=org_id)

    params, _ = await _run(turn, org_id=org_id, user_id=user_id)

    listed = turn["list"].await_args.kwargs
    assert (listed["org_id"], listed["user_id"]) == (org_id, user_id)
    assert [c["connection_id"] for c in params["custom_api"]["connections"]] == [9]


async def test_a_grant_at_no_version_is_refused_by_the_producer_itself(turn):
    """The job model refuses it too; the producer must not rely on that."""
    from cowork.common.settings.app_settings import TurnQueueSettings
    from cowork.db.scoped import TenantScope

    turn["grant"].return_value = _granted({3: 1, 5: 0})

    with pytest.raises(ProductPermissionUnavailable):
        await prod._mint_custom_api_block(
            scope=TenantScope(org_mode=True, org_id="o1", user_id="u1"), correlation_id="r",
            turn_key_id="turnkey", disabled=None, settings=TurnQueueSettings(),
        )


# ── usage notes (step 2) ─────────────────────────────────────────────────────


@pytest.fixture()
def templates(monkeypatch):
    """A database holding acme_crm templates for two members of o1 and one of o2, saved the real way."""
    from anton.core.tools.connector_draft import parse_connector_draft
    from sqlalchemy.pool import StaticPool
    from sqlmodel import Session, SQLModel, create_engine

    from cowork.db.scoped import ScopedSession, TenantScope
    from cowork.services.connectors.custom_connectors import CustomConnectorService

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    for org, user, notes in (("o1", "u1", "Call /v1/me first."), ("o1", "u2", "Other member's notes."),
                             ("o2", "u1", "Other org's notes.")):
        draft = {"connector_id": "acme_crm", "label": "Acme CRM", "usage_notes": notes,
                 "fields": [{"name": "api_key", "label": "API key"}]}
        scope = TenantScope(org_mode=True, org_id=org, user_id=user)
        CustomConnectorService(ScopedSession(Session(engine), scope)).save_member_draft(parse_connector_draft(draft))

    async def run_db(fn, *, scope):
        return fn(ScopedSession(Session(engine), scope))

    monkeypatch.setattr(prod, "run_db", run_db)
    return engine


async def test_the_turn_carries_the_notes_of_the_members_own_template(turn, templates):
    params, _ = await _run(turn)

    assert params["connectors"]["acme_crm"] == {"usage_notes": "Call /v1/me first."}
    assert "beta_api" not in params["connectors"]


@pytest.mark.parametrize(("org_id", "user_id", "expected"), [
    ("o1", "u2", "Other member's notes."), ("o2", "u1", "Other org's notes."),
], ids=["another member", "another org"])
async def test_each_member_gets_only_their_own_templates_notes(turn, templates, org_id, user_id, expected):
    turn["grant"].return_value = _granted({3: 1, 5: 2}, user_id=user_id, organization_id=org_id)

    params, _ = await _run(turn, org_id=org_id, user_id=user_id)

    assert params["connectors"]["acme_crm"] == {"usage_notes": expected}


async def test_a_connection_whose_template_is_gone_gets_no_notes(turn, templates):
    turn["list"].return_value = [_listed(3, "beta_api", "Ops")]
    turn["grant"].return_value = _granted({3: 1})

    params, _ = await _run(turn)

    assert "connectors" not in params
    assert params["custom_api"]["connections"][0]["connector_id"] == "beta_api"


async def test_notes_are_scrubbed_and_cut(turn, templates):
    from sqlmodel import Session, select

    from cowork.models.custom_connector import CustomConnector
    from cowork.services.connectors.custom_connectors import USAGE_NOTES_MAX

    # A shape scrub_output knows; the template parser refuses long token runs at save anyway.
    planted = "sk-proj-" + "abcdefghijklmnopqrstuvwxyz0123456789ABCD"
    with Session(templates) as session:
        row = session.exec(select(CustomConnector).where(CustomConnector.created_by == "u1",
                                                         CustomConnector.org_id == "o1")).one()
        row.usage_notes = f"Use {planted}. " + "x" * 3000
        session.add(row)
        session.commit()

    params, _ = await _run(turn)

    notes = params["connectors"]["acme_crm"]["usage_notes"]
    assert planted not in notes
    assert len(notes) <= USAGE_NOTES_MAX


async def test_built_in_notes_stay_beside_the_custom_ones(turn, templates, monkeypatch):
    monkeypatch.setattr(prod, "_mint_oauth_block",
                        AsyncMock(return_value={"connections": [{"engine": "hubspot", "name": "main"}]}))
    monkeypatch.setattr(prod.registry, "usage_notes_for", lambda engines: {"hubspot": "Built-in notes."})

    params, _ = await _run(turn)

    assert params["connectors"] == {
        "acme_crm": {"usage_notes": "Call /v1/me first."}, "hubspot": {"usage_notes": "Built-in notes."},
    }


async def test_a_database_that_cannot_answer_costs_the_notes_not_the_turn(turn, monkeypatch):
    async def broken(fn, *, scope):
        raise RuntimeError("database down")

    monkeypatch.setattr(prod, "run_db", broken)

    params, _ = await _run(turn)

    assert params["custom_api"]["connections"]
    assert "connectors" not in params


def test_the_notes_read_nothing_outside_a_members_org_scope(templates):
    from sqlmodel import Session

    from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
    from cowork.services.connectors.custom_connectors import member_turn_usage_notes

    from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES
    from cowork.models.custom_connector import CustomConnector

    assert member_turn_usage_notes(ScopedSession(Session(templates), LOCAL_SCOPE), ["acme_crm"]) == {}
    no_user = TenantScope(org_mode=True, org_id="o1", user_id=None)
    assert member_turn_usage_notes(ScopedSession(Session(templates), no_user), ["acme_crm"]) == {}
    # One person's partition sees its own definitions, but is not an org member's turn.
    personal = TenantScope(personal=True, org_id="p1", user_id="u1", personal_resources=ALL_PERSONAL_RESOURCES)
    with Session(templates) as session:
        session.add(CustomConnector(connector_id="acme_crm", label="Acme", spec={"fields": []},
                                    usage_notes="Personal notes.", org_id="p1", created_by="u1"))
        session.commit()
    assert member_turn_usage_notes(ScopedSession(Session(templates), personal), ["acme_crm"]) == {}


# ── no value in the job or a log (step 6) ────────────────────────────────────


async def test_an_answer_carrying_values_puts_none_of_them_in_the_job_or_a_log(turn, monkeypatch):
    """Only ids and names are copied from auth's answers, whatever else they carry."""
    planted = "acct-77f3c9e2b1d4-private"
    lines: list[str] = []
    monkeypatch.setattr(prod.logger, "warning", lambda msg, *a, **k: lines.append(msg % a if a else msg))
    monkeypatch.setattr(prod.logger, "info", lambda msg, *a, **k: lines.append(msg % a if a else msg))
    turn["list"].return_value = [{**_listed(5), "fields": {"api_key": planted}, "secret": planted}]
    turn["grant"].return_value = {**_granted({5: 1}), "fields": {"api_key": planted}}

    params, _ = await _run(turn)

    assert planted not in json.dumps(params)
    assert not [line for line in lines if planted in line]
