"""The directory marks connectors that connect with a typed key or token, so
the web app offers a custom connector only where one could stand in."""
from __future__ import annotations

import pytest

from cowork.api.v1.endpoints.connectors import specs as specs_endpoints
from cowork.db.scoped import LOCAL_SCOPE
from cowork.services.connectors.specs._registry import connector_metadata
from tests._fakes import FakeRequest


@pytest.mark.asyncio
async def test_the_directory_marks_key_connectors_only():
    result = await specs_endpoints.list_connector_specs(LOCAL_SCOPE, FakeRequest())
    api_key = {c.id: c.api_key for c in result}

    assert api_key["hasdata"] is True and api_key["jira"] is True
    # A database, a vector database and a sign-in-only connector.
    assert api_key["postgres"] is False and api_key["pinecone"] is False and api_key["xero"] is False


def _spec(category="crm", **method):
    return {"id": "x", "label": "X", "category": category, "form": {"methods": [method]}}


@pytest.mark.asyncio
async def test_one_connector_answers_as_the_directory_does():
    hasdata = await specs_endpoints.get_connector_spec("hasdata", LOCAL_SCOPE)
    postgres = await specs_endpoints.get_connector_spec("postgres", LOCAL_SCOPE)

    assert hasdata.api_key is True and postgres.api_key is False


def test_a_secret_field_counts_as_a_key():
    assert connector_metadata(_spec(fields=[{"name": "api_key", "secret": True}])).api_key is True


def test_a_secret_field_behind_an_oauth_launch_is_not_a_key():
    spec = _spec(submit_action="oauth_launch", fields=[{"name": "client_secret", "secret": True}])

    assert connector_metadata(spec).api_key is False


def test_a_password_field_counts_as_a_key():
    assert connector_metadata(_spec(fields=[{"name": "token", "type": "password"}])).api_key is True


def test_a_method_with_no_secret_field_is_not_a_key():
    assert connector_metadata(_spec(fields=[{"name": "region"}])).api_key is False


def test_a_spec_without_a_form_is_not_a_key():
    assert connector_metadata({"id": "x", "category": "crm"}).api_key is False
