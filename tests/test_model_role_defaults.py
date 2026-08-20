"""PLANNING/CODING/ROUTER_MODEL_DEFAULTS and RECOMMENDED_PAIR are both derived
from MODEL_ROLE_DEFAULTS; this pins that they can't drift apart again."""
from cowork.common.settings.app_settings import (
    CODING_MODEL_DEFAULTS,
    MODEL_ROLE_DEFAULTS,
    PLANNING_MODEL_DEFAULTS,
    RECOMMENDED_PAIR,
    ROUTER_MODEL_DEFAULTS,
)


def test_role_default_dicts_match_the_source_table():
    for provider, roles in MODEL_ROLE_DEFAULTS.items():
        assert PLANNING_MODEL_DEFAULTS[provider] == roles["planning"]
        assert CODING_MODEL_DEFAULTS[provider] == roles["coding"]
        assert ROUTER_MODEL_DEFAULTS[provider] == roles["router"]


def test_recommended_pair_matches_the_source_table():
    for provider, roles in MODEL_ROLE_DEFAULTS.items():
        ui_key = provider.replace("_", "-")
        assert RECOMMENDED_PAIR[ui_key] == (roles["planning"], roles["coding"], roles["router"])
