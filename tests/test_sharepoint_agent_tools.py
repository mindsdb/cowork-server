"""The agent never collects a SharePoint key in chat.

`lookup_connector` returns SharePoint with an instruction and no form, so
there is nothing to hand `request_credentials`; and `request_credentials`
refuses a SharePoint form however it is stamped. Every other connector keeps
its form.
"""
from __future__ import annotations

import json

import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.harnesses.anton_harness.tools import _cowork_lookup_connector, _cowork_request_credentials
from cowork.services.connectors.specs._registry import registry


class _Spec:
    """A registry entry as `get_connector` returns it."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def model_dump(self) -> dict:
        return dict(self._data)


SHAREPOINT_SPEC = {
    "id": "sharepoint",
    "label": "SharePoint",
    "description": "Files from SharePoint sites.",
    "category": "files",
    "form": {
        "form_id": "sharepoint-connector",
        "title": "Connect SharePoint",
        "methods": [{
            "id": "app-key",
            "label": "App key",
            "fields": [{"name": "client_secret", "label": "Client secret", "type": "password", "secret": True}],
        }],
    },
}


@pytest.fixture
def sharepoint_in_registry(monkeypatch):
    real = registry.get_connector
    monkeypatch.setattr(
        registry, "get_connector", lambda cid: _Spec(SHAREPOINT_SPEC) if cid == "sharepoint" else real(cid),
    )


@pytest.fixture
def org_mode(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.mark.asyncio
async def test_lookup_returns_sharepoint_with_an_instruction_and_no_form(sharepoint_in_registry):
    result = json.loads(await _cowork_lookup_connector(None, {"id": "sharepoint"}))

    assert result["id"] == "sharepoint"
    assert "form" not in result
    assert "Customize" in result["next_step"]
    assert "Add files from SharePoint" in result["next_step"]
    assert "client_secret" not in json.dumps(result)


@pytest.mark.asyncio
async def test_in_org_mode_lookup_points_to_the_desktop_app(sharepoint_in_registry, org_mode):
    result = json.loads(await _cowork_lookup_connector(None, {"id": "sharepoint"}))

    assert "form" not in result
    assert "desktop app" in result["next_step"]
    assert "Customize" not in result["next_step"]


@pytest.mark.asyncio
async def test_lookup_still_returns_a_form_for_other_connectors():
    result = json.loads(await _cowork_lookup_connector(None, {"id": "postgres"}))

    assert result["form"]["_connector_id"] == "postgres"
    assert "request_credentials" in result["next_step"]


@pytest.mark.parametrize(
    "spec",
    [
        {"_connector_id": "sharepoint", "title": "Connect SharePoint", "fields": []},
        {"engine": "sharepoint", "title": "Connect SharePoint", "fields": [{"name": "client_secret", "type": "password"}]},
    ],
    ids=["stamped", "handcrafted"],
)
@pytest.mark.asyncio
async def test_request_credentials_renders_no_sharepoint_form(spec):
    result = await _cowork_request_credentials(None, spec)

    assert result.startswith("request_credentials: ")
    assert "data-vault-form" not in result
    assert "Customize" in result


@pytest.mark.asyncio
async def test_in_org_mode_request_credentials_points_to_the_desktop_app(org_mode):
    result = await _cowork_request_credentials(None, {"engine": "sharepoint", "title": "Connect", "fields": []})

    assert "data-vault-form" not in result
    assert "desktop app" in result
