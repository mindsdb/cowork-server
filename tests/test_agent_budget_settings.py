"""Agent budget settings (max_tool_rounds / max_continuations / max_turn_tokens).

Cowork defaults these HIGHER than anton's own CoreSettings defaults (25/3):
long knowledge-work tasks routinely need 30-50 tool rounds, and at anton's
defaults the agent stops mid-task to ask "want me to continue?" — which in
unattended contexts (evals, channels, scheduled work) is a dead end. The DB
value is overlaid onto AntonSettings per conversation, so it must round-trip
the settings service as an int and reject junk that would break sessions.
"""

import pytest

from cowork.common.settings.user_settings import UserSettings

# UserSettings has no env_prefix and CoreSettings reads ANTON_*; a developer
# shell (or an eval harness) exporting these must not leak into assertions.
_ENV_KEYS = (
    "MAX_TOOL_ROUNDS",
    "MAX_CONTINUATIONS",
    "ANTON_MAX_TOOL_ROUNDS",
    "ANTON_MAX_CONTINUATIONS",
    "COWORK_DEFAULT_MAX_TOOL_ROUNDS",
    "COWORK_DEFAULT_MAX_CONTINUATIONS",
    "MAX_TURN_TOKENS",
    "ANTON_MAX_TURN_TOKENS",
    "COWORK_DEFAULT_MAX_TURN_TOKENS",
)


@pytest.fixture(autouse=True)
def _clean_budget_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    _clear_app_settings_cache()
    yield
    _clear_app_settings_cache()


def _clear_app_settings_cache():
    from cowork.common.settings.app_settings import get_app_settings

    if hasattr(get_app_settings, "cache_clear"):
        get_app_settings.cache_clear()


def test_defaults_are_50_and_5():
    s = UserSettings()
    assert s.max_tool_rounds == 50
    assert s.max_continuations == 5


def test_hosted_deployments_can_lower_defaults_via_env(monkeypatch):
    # Hosted web builds have no Settings UI (sidebar entries are
    # desktop-only), and inference cost lands on the operator — the
    # COWORK_DEFAULT_* env vars are the deployment-level lever for users
    # who never set the per-user setting.
    monkeypatch.setenv("COWORK_DEFAULT_MAX_TOOL_ROUNDS", "30")
    monkeypatch.setenv("COWORK_DEFAULT_MAX_CONTINUATIONS", "2")
    _clear_app_settings_cache()
    s = UserSettings()
    assert s.max_tool_rounds == 30
    assert s.max_continuations == 2
    # An explicit per-user value still wins over the deployment default.
    assert UserSettings.model_validate({"max_tool_rounds": "80"}).max_tool_rounds == 80


def test_defaults_exceed_anton_core_defaults():
    # The entire point of these settings: Cowork runs with more headroom
    # than anton's CLI defaults. If anton raises its defaults above ours,
    # the overlay would silently *lower* budgets — revisit both together.
    from anton.core.settings import CoreSettings

    core = CoreSettings()
    assert UserSettings().max_tool_rounds >= core.max_tool_rounds
    assert UserSettings().max_continuations >= core.max_continuations


def test_string_values_coerce_to_int():
    # The settings DB stores strings; loading must coerce.
    s = UserSettings.model_validate({"max_tool_rounds": "80", "max_continuations": "2"})
    assert s.max_tool_rounds == 80
    assert s.max_continuations == 2


@pytest.mark.parametrize(
    "key,bad",
    [
        ("max_tool_rounds", "0"),       # below ge=5: would end every turn instantly
        ("max_tool_rounds", "10000"),   # above le=500: runaway-cost guard gone
        ("max_tool_rounds", "many"),    # not a number
        ("max_continuations", "-1"),    # below ge=0
        ("max_continuations", "999"),   # above le=25
    ],
)
def test_out_of_bounds_values_rejected(key, bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UserSettings.model_validate({key: bad})


def test_settings_service_round_trip():
    from cowork.db.session import get_open_session
    from cowork.services.settings import SettingService

    session = get_open_session()
    service = SettingService(session)
    try:
        service.upsert_setting("max_tool_rounds", "75")
        resp = service.get_setting("max_tool_rounds")
        assert resp.is_set
        assert int(resp.value) == 75
        assert service.load().max_tool_rounds == 75
        with pytest.raises(ValueError):
            service.upsert_setting("max_tool_rounds", "not-a-number")
    finally:
        # The test DB is shared session-wide — never leak the 75 row (or the
        # cache invalidation that goes with it) into later tests, even when
        # an assertion above fails.
        try:
            service.delete_setting("max_tool_rounds")
        except ValueError:
            pass
        session.close()
    assert UserSettings().max_tool_rounds == 50  # back to Cowork default


def test_budget_env_vars_not_synced_from_dotenv():
    # Regression guard for the credential-push bricker: anton's CoreSettings
    # accepts any int, so a stale anton-CLI line like
    # ANTON_MAX_TOOL_ROUNDS=1000 in ~/.cowork/.env is valid for the CLI but
    # out of bounds for UserSettings. sync_env_vars_to_db raises on the first
    # invalid mapped key — if these were mapped, one stale line would 400
    # every credential push / web token refresh (and a valid line would
    # silently revert the user's UI choice, the ENG-739 model re-pin bug).
    from cowork.migrations import _ENV_TO_SETTING

    assert "ANTON_MAX_TOOL_ROUNDS" not in _ENV_TO_SETTING
    assert "ANTON_MAX_CONTINUATIONS" not in _ENV_TO_SETTING
    assert "ANTON_MAX_TURN_TOKENS" not in _ENV_TO_SETTING


def test_anton_settings_accepts_overlay():
    # The harness bridge does setattr(anton_settings, attr, db_val) for these
    # keys; AntonSettings must expose them (inherited from CoreSettings) and
    # ChatSession reads them off the same object. _env_file=None keeps the
    # developer's real ~/.anton/.env out of the assertion.
    from anton.config.settings import AntonSettings

    a = AntonSettings(_env_file=None)
    a.max_tool_rounds = 75
    a.max_continuations = 4
    assert (a.max_tool_rounds, a.max_continuations) == (75, 4)


# ── Per-turn spend ceiling (ENG-1286) ──────────────────────────────────────


def test_turn_token_ceiling_default_is_1_25m():
    assert UserSettings().max_turn_tokens == 1_250_000


def test_turn_token_ceiling_default_matches_anton_rather_than_exceeding_it():
    """Deliberately EQUAL to anton's default, unlike the other two budgets.

    max_tool_rounds/max_continuations run looser here than anton's CLI defaults
    on purpose. This one must not: the per-turn distribution that produced
    1,250,000 was measured on Cowork traffic, so it already reflects those
    looser round budgets. Raising it here would loosen the ceiling against the
    exact population it was sized on.
    """
    from anton.core.settings import CoreSettings

    core = CoreSettings()
    if not hasattr(core, "max_turn_tokens"):
        pytest.skip(
            "pinned anton predates ENG-1286's max_turn_tokens — this assertion "
            "activates at the weekly release, and its passing is the signal "
            "that the merge-order note in the PR body is satisfied"
        )
    assert UserSettings().max_turn_tokens == core.max_turn_tokens


def test_hosted_deployments_can_lower_the_ceiling_via_env(monkeypatch):
    monkeypatch.setenv("COWORK_DEFAULT_MAX_TURN_TOKENS", "400000")
    _clear_app_settings_cache()
    assert UserSettings().max_turn_tokens == 400_000
    # An explicit per-user value still wins over the deployment default.
    assert UserSettings.model_validate(
        {"max_turn_tokens": "2000000"}
    ).max_turn_tokens == 2_000_000


@pytest.mark.parametrize(
    "bad",
    [
        "0",          # ge=100_000: the UI must not be able to switch the guard OFF
        "50000",      # below ge: would stop nearly every turn immediately
        "99999999",   # above le=50_000_000
        "lots",       # not a number
    ],
)
def test_turn_token_ceiling_out_of_bounds_rejected(bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UserSettings.model_validate({"max_turn_tokens": bad})


def test_overlay_skips_a_field_the_pinned_anton_does_not_have():
    """Version skew must degrade, never crash.

    anton is a git dep pinned to branch="main", so a budget can exist in
    cowork-server while the pinned anton still predates it — the field reaches
    main at the weekly release, not when the anton PR merges. pydantic raises
    `ValueError: object has no field "x"` on setattr of an unknown field, and
    the overlay loop runs on EVERY session build, so an unguarded overlay turns
    a one-week ordering gap into a total agent outage.

    Reproduces the raise on a stand-in model, then asserts the guard the
    harness uses (`hasattr` before `setattr`) is what makes it survivable.
    """
    from pydantic_settings import BaseSettings

    class PinnedAnton(BaseSettings):
        model_config = {"env_prefix": "ANTON_", "extra": "ignore"}
        max_tool_rounds: int = 25

    old = PinnedAnton()
    with pytest.raises(ValueError):
        setattr(old, "max_turn_tokens", 1_250_000)

    applied = []
    for attr, val in (("max_tool_rounds", 50), ("max_turn_tokens", 1_250_000)):
        if not hasattr(old, attr):
            continue
        setattr(old, attr, val)
        applied.append(attr)
    assert applied == ["max_tool_rounds"]
    assert old.max_tool_rounds == 50


def test_anton_settings_accepts_the_ceiling_overlay():
    """The live pinned anton — xfails until anton's ENG-1286 reaches main.

    Not a soft assertion: when this starts passing, the merge-order note in the
    PR body is satisfied and the guard above stops being load-bearing.
    """
    from anton.config.settings import AntonSettings

    a = AntonSettings(_env_file=None)
    if not hasattr(a, "max_turn_tokens"):
        pytest.skip("pinned anton predates ENG-1286's max_turn_tokens")
    a.max_turn_tokens = 900_000
    assert a.max_turn_tokens == 900_000
