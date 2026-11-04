"""PLANNING/CODING/ROUTER/SIDEKICK_MODEL_DEFAULTS and RECOMMENDED_PAIR are all
derived from MODEL_ROLE_DEFAULTS; this pins that they can't drift apart again."""
from cowork.common.settings.app_settings import (
    AGENT_ROLE_ORDER,
    CODING_MODEL_DEFAULTS,
    MODEL_ROLE_DEFAULTS,
    PLANNING_MODEL_DEFAULTS,
    RECOMMENDED_PAIR,
    ROUTER_MODEL_DEFAULTS,
    SIDEKICK_MODEL_DEFAULTS,
)


def test_role_default_dicts_match_the_source_table():
    for provider, roles in MODEL_ROLE_DEFAULTS.items():
        # AGENT_ROLE_ORDER is what the pair, the parse filter and the endpoint all
        # read, so a role in the table it does not name is a role nothing resolves.
        assert set(roles) == set(AGENT_ROLE_ORDER), provider
        assert PLANNING_MODEL_DEFAULTS[provider] == roles["planning"]
        assert CODING_MODEL_DEFAULTS[provider] == roles["coding"]
        assert ROUTER_MODEL_DEFAULTS[provider] == roles["router"]
        assert SIDEKICK_MODEL_DEFAULTS[provider] == roles["sidekick"]


def test_recommended_pair_matches_the_source_table():
    for provider, roles in MODEL_ROLE_DEFAULTS.items():
        ui_key = provider.replace("_", "-")
        assert RECOMMENDED_PAIR[ui_key] == tuple(roles[role] for role in AGENT_ROLE_ORDER)


def test_the_sidekick_starts_on_the_coding_model_everywhere():
    # anton falls back to the coding role when no sidekick is configured, so an
    # unset sidekick runs the same model on either side of the boundary.
    for provider, roles in MODEL_ROLE_DEFAULTS.items():
        assert roles["sidekick"] == roles["coding"], provider


def test_new_roles_are_appended_so_older_clients_keep_their_slots():
    assert AGENT_ROLE_ORDER[:3] == ("planning", "coding", "router")
    assert AGENT_ROLE_ORDER[-1] == "sidekick"
