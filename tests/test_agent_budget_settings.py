"""Agent tool-budget settings (max_tool_rounds / max_continuations).

Cowork defaults these HIGHER than anton's own CoreSettings defaults (25/3):
long knowledge-work tasks routinely need 30-50 tool rounds, and at anton's
defaults the agent stops mid-task to ask "want me to continue?" — which in
unattended contexts (evals, channels, scheduled work) is a dead end. The DB
value is overlaid onto AntonSettings per conversation, so it must round-trip
the settings service as an int and reject junk that would break sessions.
"""

import pytest

from cowork.common.settings.user_settings import UserSettings


def test_defaults_are_50_and_5():
    s = UserSettings()
    assert s.max_tool_rounds == 50
    assert s.max_continuations == 5


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
    try:
        service = SettingService(session)
        service.upsert_setting("max_tool_rounds", "75")
        resp = service.get_setting("max_tool_rounds")
        assert resp.is_set
        assert int(resp.value) == 75
        assert service.load().max_tool_rounds == 75
        with pytest.raises(ValueError):
            service.upsert_setting("max_tool_rounds", "not-a-number")
        service.delete_setting("max_tool_rounds")
        assert service.load().max_tool_rounds == 50  # back to Cowork default
    finally:
        session.close()


def test_anton_settings_accepts_overlay():
    # The harness bridge does setattr(anton_settings, attr, db_val) for these
    # keys; AntonSettings must expose them (inherited from CoreSettings) and
    # ChatSession reads them off the same object.
    from anton.config.settings import AntonSettings

    a = AntonSettings()
    a.max_tool_rounds = 75
    a.max_continuations = 4
    assert (a.max_tool_rounds, a.max_continuations) == (75, 4)
