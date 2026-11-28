"""The Microsoft Teams (Bot) connector in the registry: listed only where the Teams bot is set up,
labelled a bot, and never offered to the agent as a data source."""

from __future__ import annotations

import json

import pytest

from cowork.services.connectors.specs import _registry
from cowork.services.connectors.specs._registry import TEAMS_BOT_CONNECTOR, ConnectorSpecRegistry


@pytest.fixture
def registry():
    return ConnectorSpecRegistry()


def test_without_the_teams_bot_it_isnt_listed_or_served(registry, monkeypatch):
    monkeypatch.setattr(_registry, "_teams_bot_listed", lambda: False)

    assert TEAMS_BOT_CONNECTOR not in {c.id for c in registry.list_connectors()}
    assert registry.get_connector(TEAMS_BOT_CONNECTOR) is None
    assert "microsoft_teams" in {c.id for c in registry.list_connectors()}



def test_without_the_teams_bot_a_connector_added_to_the_registry_stays(registry, monkeypatch):
    monkeypatch.setattr(_registry, "_teams_bot_listed", lambda: False)

    registry.get_connectors()["added"] = {"id": "added", "label": "Added", "category": "database", "form": {}}

    assert "added" in registry.get_connectors()


def test_where_the_teams_bot_is_set_up_it_is_listed_as_a_bot_beside_the_graph_connector(registry, monkeypatch):
    monkeypatch.setattr(_registry, "_teams_bot_listed", lambda: True)

    listed = {c.id: c for c in registry.list_connectors()}
    bot = listed[TEAMS_BOT_CONNECTOR]
    assert (bot.label, bot.category, bot.bot) == ("Microsoft Teams (Bot)", "communication", True)
    assert "only sees the messages you send it or @mention it in" in bot.notice
    assert listed["microsoft_teams"].bot is False
    [method] = registry.get_connector(TEAMS_BOT_CONNECTOR).form.methods
    assert (method.submit_action, method.fields if hasattr(method, "fields") else []) == ("teams_connection", [])


@pytest.mark.parametrize("query", ["microsoft_teams_bot", "teams bot", "mindshub in teams", "teams"])
def test_the_agent_is_never_offered_the_bot_as_a_connection(registry, monkeypatch, query):
    monkeypatch.setattr(_registry, "_teams_bot_listed", lambda: True)

    assert TEAMS_BOT_CONNECTOR not in [c.id for c in registry.match_connector(query).candidates]


def test_it_is_listed_only_with_every_teams_setting_and_the_teams_packages(monkeypatch):
    for name in ("COWORK_TEAMS_BOT_APP_ID", "COWORK_TEAMS_BOT_APP_SECRET", "COWORK_TEAMS_TENANT_ID", "COWORK_TEAMS_INTERNAL_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert _registry._teams_bot_listed() is False
    monkeypatch.setenv("COWORK_TEAMS_BOT_APP_ID", "app")
    monkeypatch.setenv("COWORK_TEAMS_BOT_APP_SECRET", "secret")
    monkeypatch.setenv("COWORK_TEAMS_TENANT_ID", "tenant")
    monkeypatch.setenv("COWORK_TEAMS_INTERNAL_KEY", "k" * 40)
    monkeypatch.setattr("cowork.services.connectors.teams.client.teams_available", lambda: True)
    assert _registry._teams_bot_listed() is True
    monkeypatch.setattr("cowork.services.connectors.teams.client.teams_available", lambda: False)
    assert _registry._teams_bot_listed() is False


@pytest.fixture
def bot_listed(monkeypatch):
    """The shared registry, reloaded as it is where the Teams bot is set up."""
    from cowork.services.connectors.specs._registry import registry as shared

    monkeypatch.setattr(_registry, "_teams_bot_listed", lambda: True)
    shared.reload()
    assert TEAMS_BOT_CONNECTOR in shared.get_connectors()
    yield
    monkeypatch.undo()
    shared.reload()


@pytest.mark.parametrize("tc_input", [
    {"id": "microsoft_teams_bot"}, {"query": "teams bot"}, {"query": "teams"}, {"query": "mindshub in teams"},
])
@pytest.mark.asyncio
async def test_the_agents_own_lookup_never_offers_the_bot(bot_listed, tc_input):
    from cowork.harnesses.anton_harness.tools import _cowork_lookup_connector

    result = await _cowork_lookup_connector(None, tc_input)

    found = json.loads(result)
    assert found.get("id") != TEAMS_BOT_CONNECTOR
    assert (found.get("form") or {}).get("_connector_id") != TEAMS_BOT_CONNECTOR
    assert TEAMS_BOT_CONNECTOR not in found.get("available_ids", [])
    assert TEAMS_BOT_CONNECTOR not in [c.get("id") for c in found.get("candidates", [])]


@pytest.mark.parametrize("tc_input", [
    {"id": "microsoft_teams_bot"}, {"query": "teams bot"}, {"query": "Microsoft Teams (Bot)"},
    {"query": "mindshub in teams"},
])
@pytest.mark.asyncio
async def test_asking_for_the_bot_by_name_points_the_person_to_connect_apps_and_data(bot_listed, tc_input):
    from cowork.harnesses.anton_harness.tools import _cowork_lookup_connector

    found = json.loads(await _cowork_lookup_connector(None, tc_input))

    assert "Connect Apps and Data" in found["message"]
    assert "custom connector" not in found["message"]


@pytest.mark.asyncio
async def test_without_the_bot_listed_its_name_gets_the_usual_no_match(monkeypatch):
    from cowork.harnesses.anton_harness.tools import _cowork_lookup_connector

    monkeypatch.setattr(_registry, "_teams_bot_listed", lambda: False)
    _registry.registry.reload()
    try:
        found = json.loads(await _cowork_lookup_connector(None, {"id": "microsoft_teams_bot"}))
    finally:
        monkeypatch.undo()
        _registry.registry.reload()

    assert "Connect Apps and Data" not in found["message"]


@pytest.mark.parametrize("key", ["engine", "_connector_id"])
@pytest.mark.asyncio
async def test_the_agent_cant_ask_for_the_bot_in_chat(bot_listed, key):
    from cowork.harnesses.anton_harness.tools import _cowork_request_credentials

    result = await _cowork_request_credentials(None, {key: TEAMS_BOT_CONNECTOR, "title": "Connect", "fields": []})

    assert "data-vault-form" not in result and "Connect Apps and Data" in result
