"""A new connection that would overwrite a saved one is reported before it is submitted.

Roam names a connection after its graph, so a second token for the same graph
lands on the first connection's record and replaces it. The form asks
``/connectors/submissions/replaces`` first, and only a submission naming the
connection goes on to replace it.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from anton.core.datasources.data_vault import LocalDataVault
from fastapi.testclient import TestClient

from cowork.api.v1.endpoints.connectors import submissions as submissions_endpoint
from cowork.server import create_app
from cowork.services.connectors.persist import persist_connection

PATH = "/api/v1/connectors/submissions/replaces"
READ_EDIT = {"graph_name": "Mindshub_test", "api_token": "roam-graph-token-read-edit-1a2b"}
READ_ONLY = {"graph_name": "Mindshub_test", "api_token": "roam-graph-token-read-only-3c4d"}
OTHER_GRAPH = {"graph_name": "Other_graph", "api_token": "roam-graph-token-other-5e6f"}


@pytest.fixture()
def vault(tmp_path, monkeypatch) -> LocalDataVault:
    v = LocalDataVault(Path(tmp_path) / "vault")
    monkeypatch.setattr(submissions_endpoint, "vault_for_scope", lambda scope=None: v)
    return v


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    get_app_settings.cache_clear()
    return TestClient(create_app())


def _ask(client, values, *, name="", connector_id="roam_research", method="graph-api-token"):
    res = client.post(PATH, json={
        "connector_id": connector_id, "method": method, "name": name, "values": values,
    })
    assert res.status_code == 200, res.text
    return res.json()["replaces"]


def test_a_second_token_for_a_connected_graph_names_the_connection_it_replaces(client, vault):
    slug = persist_connection(
        "roam_research", "graph-api-token", "", {**READ_EDIT, "user_label": "Team graph"}, vault=vault,
    )

    replaces = _ask(client, {**READ_ONLY, "user_label": ""})

    assert replaces == {"name": slug, "label": "Team graph"}


def test_a_graph_name_in_another_case_is_another_graph(client, vault):
    """Roam graph names are case-sensitive; the save keeps them apart, so nothing is replaced."""
    persist_connection("roam_research", "graph-api-token", "", READ_EDIT, vault=vault)

    assert _ask(client, {**READ_ONLY, "graph_name": "mindshub_TEST"}) is None


def test_another_graph_is_a_new_connection(client, vault):
    persist_connection("roam_research", "graph-api-token", "", READ_EDIT, vault=vault)

    assert _ask(client, OTHER_GRAPH) is None


def test_a_first_connection_replaces_nothing(client, vault):
    assert _ask(client, READ_EDIT) is None


def test_a_submission_naming_its_connection_is_not_a_surprise_replacement(client, vault):
    slug = persist_connection("roam_research", "graph-api-token", "", READ_EDIT, vault=vault)

    assert _ask(client, READ_ONLY, name=slug) is None


def test_a_connector_without_an_identity_field_never_replaces(client, vault):
    persist_connection("notion", None, "", {"api_key": "secret_one"}, vault=vault)

    assert _ask(client, {"api_key": "secret_two"}, connector_id="notion", method=None) is None


def test_the_answer_matches_what_the_save_then_does(client, vault):
    """The check and persist_connection must agree: the named record is the one overwritten."""
    persist_connection("roam_research", "graph-api-token", "", READ_EDIT, vault=vault)
    replaces = _ask(client, READ_ONLY)

    slug = persist_connection("roam_research", "graph-api-token", "", READ_ONLY, vault=vault)

    assert slug == replaces["name"]
    assert [c["name"] for c in vault.list_connections()] == [slug]
    assert vault.read_record("roam_research", slug)["fields"]["api_token"] == READ_ONLY["api_token"]
