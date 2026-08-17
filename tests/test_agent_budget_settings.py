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
    monkeypatch.setenv("COWORK_DEFAULT_MAX_TURN_TOKENS", "800000")
    _clear_app_settings_cache()
    assert UserSettings().max_turn_tokens == 800_000
    # An explicit per-user value still wins over the deployment default.
    assert UserSettings.model_validate(
        {"max_turn_tokens": "2000000"}
    ).max_turn_tokens == 2_000_000


@pytest.mark.parametrize("bad", ["400000", "99999999", "lots"])
def test_turn_token_ceiling_out_of_bounds_rejected(bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UserSettings.model_validate({"max_turn_tokens": bad})


def test_the_top_of_the_range_is_the_no_limit_setting():
    """"No limit" in the UI writes the MAX, not a sentinel.

    At 50_000_000 the ceiling is effectively — not literally — off. A turn makes
    roughly `max_tool_rounds x (max_continuations + 1)` LLM calls; at this
    repo's defaults (50 x 6 = ~306) that reaches 50M at ~163k per call, which is
    BELOW the ~190k a long conversation carries. It has never happened (largest
    turn in 30 days: 8.26M) because real turns end and compaction intervenes,
    but "the step cap always lands first" is not a guarantee.

    The top of the range being the off switch used to be a problem because it
    was undiscoverable; the checkbox fixes that, which is why the
    0-means-unlimited sentinel was removed rather than kept. Keeping the range
    contiguous means no validator guarding a hole and no special case in the
    client clamp.
    """
    assert UserSettings.model_validate(
        {"max_turn_tokens": "50000000"}
    ).max_turn_tokens == 50_000_000


@pytest.mark.parametrize("bad", ["0", "1", "100000", "749999"])
def test_below_the_floor_is_rejected(bad):
    """Including 0 — there is no sentinel, so 0 is just an illegal ceiling."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UserSettings.model_validate({"max_turn_tokens": bad})


def test_the_floor_is_above_one_call_s_worth_of_context():
    """Why 750_000 and not a rounder 100_000.

    A turn's first LLM call costs roughly the conversation's context (~190k on a
    long one). A ceiling below a couple of calls stops the turn before it has
    done anything — measured against anton's real `turn_stream`, a 100_000
    ceiling dispatched zero tools and still spent 400_000. anton now guarantees
    at least one tool round at any ceiling, so this floor is a usability bound,
    not the safety one; it is set where the tightest setting still does several
    rounds of work.

    Measured at the floor with 190k contexts: 2 tool rounds, 760_000 spent. So
    the floor buys roughly three calls, which is the bound asserted here — not
    four, which the first version of this test claimed and which fails by 10k.
    """
    from cowork.common.settings.user_settings import TURN_CEILING_FLOOR

    assert TURN_CEILING_FLOOR >= 3 * 190_000


def test_client_and_server_bounds_stay_in_lockstep():
    """Mirror of cowork's `BUDGET_FIELDS.maxTurnTokens`.

    The renderer clamps to its own range before writing; if the two drift, the
    client happily sends a value the server rejects — and the settings write is
    all-or-nothing, so it takes the user's unrelated changes down with it. The
    JS side already asserts the server's numbers; this is the missing direction.
    Update both together or not at all: cowork
    `src/renderer/cowork/lib/settingsTransform.js` -> BUDGET_FIELDS.
    """
    from cowork.common.settings.user_settings import TURN_CEILING_FLOOR

    field = UserSettings.model_fields["max_turn_tokens"]
    le = next(m.le for m in field.metadata if hasattr(m, "le"))
    assert (TURN_CEILING_FLOOR, le) == (750_000, 50_000_000)


def test_overlay_applies_every_budget_to_a_real_anton_settings():
    """Drives the REAL overlay against a REAL AntonSettings.

    The first version of this test declared its own stand-in model and
    re-implemented the hasattr/setattr loop in the test body, so it never
    imported `harness.py` — dropping `max_turn_tokens` from the tuple (silently
    disabling the whole user-facing setting) shipped green.
    """
    from anton.config.settings import AntonSettings

    from cowork.harnesses.anton_harness.harness import (
        _OVERLAID_SETTINGS,
        _overlay_user_settings,
    )

    # Asserted on the tuple, not only on the outcome: until anton's field
    # reaches the pinned `main`, the outcome assertions below take their
    # skew-guard branch, which is ALSO what dropping the key from the tuple
    # produces. Without this line that mutation ships green.
    assert "max_turn_tokens" in _OVERLAID_SETTINGS

    a = AntonSettings(_env_file=None)
    user = UserSettings.model_validate({
        "max_tool_rounds": "80", "max_continuations": "2",
        "max_turn_tokens": "900000",
    })
    applied = _overlay_user_settings(a, user)

    assert a.max_tool_rounds == 80
    assert a.max_continuations == 2
    assert "max_tool_rounds" in applied and "max_continuations" in applied
    if hasattr(AntonSettings(_env_file=None), "max_turn_tokens"):
        assert a.max_turn_tokens == 900_000
        assert "max_turn_tokens" in applied
    else:  # pinned anton predates ENG-1286 — the skew guard must have skipped it
        assert "max_turn_tokens" not in applied


def test_overlay_survives_a_field_the_pinned_anton_does_not_have():
    """Version skew must degrade, never crash — through the real function.

    anton is a git dep pinned to branch="main", so a setting can exist here
    while the pinned anton predates it; the field reaches main at the weekly
    release, not when the anton PR merges. pydantic raises on setattr of an
    unknown field, and the overlay runs on EVERY session build, so an unguarded
    overlay turns a one-week ordering gap into a total agent outage.

    Uses a stand-in for the OLD anton deliberately — we cannot install a past
    version here — but drives cowork-server's real loop over it, which is the
    half the previous test was missing.
    """
    from pydantic_settings import BaseSettings

    from cowork.harnesses.anton_harness.harness import _overlay_user_settings

    class PinnedAnton(BaseSettings):
        model_config = {"env_prefix": "ANTON_", "extra": "ignore"}
        max_tool_rounds: int = 25

    old = PinnedAnton()
    # The raise this guards, pinned so the test states its own premise.
    with pytest.raises(ValueError):
        setattr(old, "max_turn_tokens", 1_250_000)

    applied = _overlay_user_settings(
        old, UserSettings.model_validate({"max_tool_rounds": "80"})
    )
    assert applied == ["max_tool_rounds"]
    assert old.max_tool_rounds == 80


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
