"""PLANNING/CODING/ROUTER_MODEL_DEFAULTS and RECOMMENDED_PAIR are both derived
from MODEL_ROLE_DEFAULTS; this pins that they can't drift apart again."""
import pytest
from pydantic import SecretStr

from cowork.common.settings.app_settings import (
    AGENT_ROLE_ORDER,
    CODING_MODEL_DEFAULTS,
    DIRECT_EFFORT_CATALOG,
    MODEL_ROLE_DEFAULTS,
    PLANNING_MODEL_DEFAULTS,
    RECOMMENDED_MODELS,
    RECOMMENDED_PAIR,
    RETIRED_DIRECT_MODELS,
    ROUTER_MODEL_DEFAULTS,
)
from cowork.common.settings.user_settings import Provider, UserSettings


def test_role_default_dicts_match_the_source_table():
    for provider, roles in MODEL_ROLE_DEFAULTS.items():
        # AGENT_ROLE_ORDER is what the pair, the parse filter and the endpoint all
        # read, so a role in the table it does not name is a role nothing resolves.
        assert set(roles) == set(AGENT_ROLE_ORDER), provider
        assert PLANNING_MODEL_DEFAULTS[provider] == roles["planning"]
        assert CODING_MODEL_DEFAULTS[provider] == roles["coding"]
        assert ROUTER_MODEL_DEFAULTS[provider] == roles["router"]


def test_recommended_pair_matches_the_source_table():
    for provider, roles in MODEL_ROLE_DEFAULTS.items():
        ui_key = provider.replace("_", "-")
        assert RECOMMENDED_PAIR[ui_key] == tuple(roles[role] for role in AGENT_ROLE_ORDER)


# minds-cloud's picker list is fetched live, so it has no compiled list to check.
DIRECT_PROVIDERS = [p for p in MODEL_ROLE_DEFAULTS if p != "minds_cloud"]


@pytest.mark.parametrize("provider", DIRECT_PROVIDERS)
def test_compiled_defaults_are_models_the_picker_offers(provider):
    """A default the picker does not list is one nobody checked exists: the
    OpenAI coding/router default was "gpt-5.5-mini", which OpenAI does not
    serve, and the router default is what the gate runs every turn on."""
    offered = RECOMMENDED_MODELS[provider]
    for role, model in MODEL_ROLE_DEFAULTS[provider].items():
        assert model in offered, (provider, role, model)


def test_no_retired_model_is_offered_or_a_default():
    for provider, retired in RETIRED_DIRECT_MODELS.items():
        for old, new in retired.items():
            assert old not in RECOMMENDED_MODELS[provider]
            assert old not in MODEL_ROLE_DEFAULTS[provider].values()
            assert old not in DIRECT_EFFORT_CATALOG
            assert new in RECOMMENDED_MODELS[provider]


def test_every_effort_default_is_a_level_the_model_lists():
    for model, entry in DIRECT_EFFORT_CATALOG.items():
        assert entry["default"] in entry["efforts"], model


def test_openai_gpt5_models_do_not_offer_minimal():
    # OpenAI rejects it: "Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'".
    for model in ("gpt-5.5", "gpt-5.4-mini"):
        assert DIRECT_EFFORT_CATALOG[model]["efforts"] == ["none", "low", "medium", "high", "xhigh"]


def test_a_stored_retired_openai_pin_reads_back_as_its_replacement():
    """The picker wrote the old defaults back as explicit pins, so changing the
    default alone would leave those users on a model that 404s."""
    s = UserSettings(
        planning_provider=Provider.OPENAI, coding_provider=Provider.OPENAI,
        router_provider=Provider.OPENAI, openai_api_key=SecretStr("sk-openai"),
        planning_model="gpt-5.5", coding_model="gpt-5.5-mini", router_model="gpt-5.5-mini",
    )

    assert s.coding_model == s.resolved_coding_model == "gpt-5.4-mini"
    assert s.router_model == s.resolved_router_model == "gpt-5.4-mini"
    assert s.planning_model == "gpt-5.5"
    assert s.resolved_gate_model == "gpt-5.4-mini"


def test_a_retired_id_on_a_byo_endpoint_is_left_alone():
    # An openai-compatible endpoint serves whatever it serves; the OpenAI
    # catalog says nothing about it.
    s = UserSettings(
        planning_provider=Provider.OPENAI_COMPATIBLE, coding_provider=Provider.OPENAI_COMPATIBLE,
        router_provider=Provider.OPENAI_COMPATIBLE, openai_compatible_api_key=SecretStr("sk-compat"),
        openai_base_url="https://proxy.example.com/v1",
        planning_model="gpt-5.5-mini", coding_model="gpt-5.5-mini", router_model="gpt-5.5-mini",
    )

    assert (s.planning_model, s.coding_model, s.router_model) == ("gpt-5.5-mini",) * 3
