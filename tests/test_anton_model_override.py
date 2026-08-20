"""ENG-1656 follow-up: the composer's per-conversation model pick must
actually drive Anton's planning/coding/router roles for that conversation's
turns, not just cosmetically label the SSE response.

Drives the REAL extracted function against a REAL AntonSettings, mirroring
test_agent_budget_settings.py's pattern for _overlay_user_settings — a
stand-in re-implementing the hasattr/setattr loop in the test body would not
catch a regression in the loop itself.
"""

from cowork.harnesses.anton_harness.harness import _apply_model_override


def test_model_override_sets_all_three_roles_on_a_real_anton_settings():
    from anton.config.settings import AntonSettings

    a = AntonSettings(_env_file=None)
    applied = _apply_model_override(a, "picked-model")

    assert a.planning_model == "picked-model"
    assert a.coding_model == "picked-model"
    assert a.router_model == "picked-model"
    assert set(applied) == {"planning_model", "coding_model", "router_model"}


def test_no_override_when_model_is_none():
    from anton.config.settings import AntonSettings

    a = AntonSettings(_env_file=None)
    before = (a.planning_model, a.coding_model, a.router_model)

    applied = _apply_model_override(a, None)

    assert applied == []
    assert (a.planning_model, a.coding_model, a.router_model) == before


def test_skew_guard_skips_fields_the_pinned_anton_does_not_have():
    """Version skew must degrade, never crash — anton is a git dep pinned to
    branch="main" (see _overlay_user_settings's docstring for why); pydantic
    raises ValueError on setattr of an unknown field."""
    from pydantic_settings import BaseSettings

    class PinnedAnton(BaseSettings):
        model_config = {"env_prefix": "ANTON_", "extra": "ignore"}
        planning_model: str = "default-planning"

    old = PinnedAnton()
    applied = _apply_model_override(old, "picked-model")

    assert applied == ["planning_model"]
    assert old.planning_model == "picked-model"
