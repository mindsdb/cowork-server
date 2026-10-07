from __future__ import annotations

import json

import pytest

from cowork.harnesses.anton_harness.tools import _cowork_lookup_connector
from cowork.services.connectors.specs._registry import registry

IMAGE_QUERIES = [
    "generate an image",
    "create a skill for image generation",
    "make me a picture of a cat",
]


@pytest.mark.parametrize("query", IMAGE_QUERIES)
async def test_agent_lookup_returns_replicate_as_the_single_match(query):
    result = json.loads(await _cowork_lookup_connector(None, {"query": query}))
    assert result.get("id") == "replicate", result
    assert result["form"]["_connector_id"] == "replicate"


@pytest.mark.parametrize("query", IMAGE_QUERIES)
def test_picker_search_returns_replicate_as_the_single_match(query):
    result = registry.match_connector(query)
    assert [c.id for c in result.candidates] == ["replicate"]
    assert result.needs_clarification is False


def test_token_is_a_secret_field():
    spec = registry.get_connector("replicate")
    (method,) = spec.form.methods
    (field,) = method.fields
    assert field.name == "api_token"
    assert field.secret is True
    assert field.required is True


def test_probe_is_told_to_check_the_account_without_running_a_model():
    # The probe reads the method description; a generation call would bill the user.
    (method,) = registry.get_connector("replicate").form.methods
    assert "https://api.replicate.com/v1/account" in method.description
    assert "not by running a model" in method.description


def test_tile_says_the_service_is_paid():
    assert "Paid service" in registry.get_connector("replicate").notice


def test_other_connectors_lookup_by_name_is_unchanged():
    # Alias words score per token, so a generic word here ("to", "ai") would pull
    # Replicate into every "connect to <X>" lookup and make it ambiguous.
    leaked = []
    for cid, spec in registry.get_connectors().items():
        if cid == "replicate":
            continue
        for query in (f"connect to {spec['label']}", f"connect my {spec['label']}"):
            if "replicate" in [c.id for c in registry.match_connector(query).candidates]:
                leaked.append(query)
    assert not leaked
