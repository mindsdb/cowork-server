"""The shipped SharePoint spec: the agent finds it, and its secret is marked.

The agent gets the Customize instruction instead of the form, and the spec
marks `client_secret` secret for the dedicated route's save, which passes no
method.
"""
from __future__ import annotations

import json

import pytest

from cowork.harnesses.anton_harness.tools import _cowork_lookup_connector
from cowork.schemas.connectors import ConnectorSpecResponse
from cowork.services.connectors.identity import secure_keys_for, spec_secret_fields
from cowork.services.connectors.specs._registry import registry


def test_the_spec_loads_and_validates():
    raw = registry.get_connectors()["sharepoint"]

    spec = ConnectorSpecResponse.model_validate(raw)

    assert spec.id == "sharepoint"
    assert [m.id for m in spec.form.methods] == ["app-key"]


@pytest.mark.parametrize("tc_input", [{"id": "sharepoint"}, {"query": "sharepoint"}, {"query": "microsoft sharepoint"}])
@pytest.mark.asyncio
async def test_lookup_finds_sharepoint_and_returns_the_instruction(tc_input):
    result = json.loads(await _cowork_lookup_connector(None, tc_input))

    assert result["id"] == "sharepoint"
    assert "form" not in result
    assert "Add files from SharePoint" in result["next_step"]


def test_only_the_client_secret_is_marked_secret():
    assert spec_secret_fields("sharepoint", None) == ["client_secret"]
    assert spec_secret_fields("sharepoint", "app-key") == ["client_secret"]


def test_secure_keys_for_a_route_save_cover_the_client_secret_only():
    fields = {"tenant_id": "t", "client_id": "c", "client_secret": "s", "site_urls": "https://x.sharepoint.com/sites/a"}

    assert secure_keys_for("sharepoint", None, fields) == ["client_secret"]
