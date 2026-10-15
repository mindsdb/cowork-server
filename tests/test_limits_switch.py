"""The limits switch: off by default, a turn runs to the deployment defaults.

With ``limits_enabled`` off, stored budgets are kept but ignored; with it on,
they apply. A budget saved before the switch existed turns it on at that
row's scope, so a user who chose a limit keeps it.
"""
from __future__ import annotations

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import (
    TurnLimits,
    UserSettings,
    default_turn_limits,
    effective_turn_limits,
    setting_is_org_scoped,
)
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.migrations import enable_limits_for_saved_budgets
from cowork.models.setting import Setting
from cowork.services.settings import SettingService

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "7ca7b810-9dad-11d1-80b4-00c04fd430c9"
ORG_C = "8da7b810-9dad-11d1-80b4-00c04fd430ca"
USER_A = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"

_ENV_KEYS = (
    "LIMITS_ENABLED",
    "MAX_TOOL_ROUNDS", "MAX_CONTINUATIONS", "MAX_TURN_TOKENS",
    "COWORK_DEFAULT_MAX_TOOL_ROUNDS", "COWORK_DEFAULT_MAX_CONTINUATIONS",
    "COWORK_DEFAULT_MAX_TURN_TOKENS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture()
def engine():
    import cowork.models.project, cowork.models.conversation  # noqa: F401
    import cowork.models.message, cowork.models.message_event  # noqa: F401
    import cowork.models.file, cowork.models.channel, cowork.models.setting  # noqa: F401
    import cowork.models.task_object, cowork.models.schedule, cowork.models.pin  # noqa: F401
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(eng)
    return eng


def _defaults() -> TurnLimits:
    app = get_app_settings()
    return TurnLimits(app.default_max_turn_tokens, app.default_max_tool_rounds, app.default_max_continuations)


_CHOSEN = {"max_turn_tokens": "900000", "max_tool_rounds": "12", "max_continuations": "1"}


def _switch_rows(engine) -> set[tuple]:
    with Session(engine) as s:
        return {
            (r.scope, r.org_id, r.user_id, r.value)
            for r in s.exec(select(Setting).where(Setting.key == "limits_enabled")).all()
        }


def test_the_switch_is_off_by_default_and_org_owned():
    assert UserSettings().limits_enabled is False
    # Org-scoped: in org mode only an org admin may write it, like the budgets.
    assert setting_is_org_scoped("limits_enabled")


def test_off_ignores_stored_budgets_and_on_applies_them():
    chosen = UserSettings.model_validate(_CHOSEN)
    assert effective_turn_limits(chosen) == _defaults()

    on = UserSettings.model_validate({**_CHOSEN, "limits_enabled": "true"})
    assert effective_turn_limits(on) == TurnLimits(900_000, 12, 1)


def test_off_follows_the_deployment_defaults(monkeypatch):
    # Self-hosted sets its budgets this way; the switch must not undo them.
    monkeypatch.setenv("COWORK_DEFAULT_MAX_TURN_TOKENS", "40000000")
    monkeypatch.setenv("COWORK_DEFAULT_MAX_TOOL_ROUNDS", "400")
    monkeypatch.setenv("COWORK_DEFAULT_MAX_CONTINUATIONS", "7")
    get_app_settings.cache_clear()
    assert effective_turn_limits(UserSettings.model_validate(_CHOSEN)) == TurnLimits(40_000_000, 400, 7)


def test_the_overlay_runs_a_turn_at_the_switchs_limits():
    from anton.config.settings import AntonSettings

    from cowork.harnesses.anton_harness.harness import _overlay_user_settings

    off = AntonSettings(_env_file=None)
    _overlay_user_settings(off, UserSettings.model_validate(_CHOSEN))
    defaults = _defaults()
    assert (off.max_tool_rounds, off.max_continuations) == (defaults.max_tool_rounds, defaults.max_continuations)

    on = AntonSettings(_env_file=None)
    _overlay_user_settings(on, UserSettings.model_validate({**_CHOSEN, "limits_enabled": "true"}))
    assert (on.max_tool_rounds, on.max_continuations) == (12, 1)
    if hasattr(on, "max_turn_tokens"):
        assert off.max_turn_tokens == defaults.max_turn_tokens
        assert on.max_turn_tokens == 900_000


def test_turning_the_switch_off_keeps_the_saved_values(engine):
    svc = SettingService(Session(engine), LOCAL_SCOPE)
    svc.upsert_setting("limits_enabled", "true")
    svc.upsert_setting("max_tool_rounds", "12")
    assert effective_turn_limits(svc.load()).max_tool_rounds == 12

    svc.upsert_setting("limits_enabled", "false")
    loaded = svc.load()
    assert loaded.max_tool_rounds == 12
    assert effective_turn_limits(loaded) == _defaults()

    svc.upsert_setting("limits_enabled", "true")
    assert effective_turn_limits(svc.load()).max_tool_rounds == 12


def test_a_saved_budget_turns_the_switch_on_at_its_scope(engine):
    with Session(engine) as s:
        # Org A saved two budgets: one switch row for the org.
        s.add(Setting(key="max_tool_rounds", value="12", scope="org", org_id=ORG_A))
        s.add(Setting(key="max_continuations", value="1", scope="org", org_id=ORG_A))
        s.add(Setting(key="max_tool_rounds", value="30", scope="user", org_id=ORG_A, user_id=USER_A))
        # Org B already turned the switch off: that sticks.
        s.add(Setting(key="max_turn_tokens", value="900000", scope="org", org_id=ORG_B))
        s.add(Setting(key="limits_enabled", value="False", scope="org", org_id=ORG_B))
        # Org C never saved a budget.
        s.add(Setting(key="greeting", value="hi", scope="org", org_id=ORG_C))
        s.commit()

    assert enable_limits_for_saved_budgets(Session(engine)) is True
    assert _switch_rows(engine) == {
        ("org", ORG_A, None, "True"),
        ("user", ORG_A, USER_A, "True"),
        ("org", ORG_B, None, "False"),
    }

    # Idempotent: a second boot writes nothing.
    assert enable_limits_for_saved_budgets(Session(engine)) is False

    org_a = SettingService(Session(engine), TenantScope(org_mode=True, org_id=ORG_A)).load()
    assert org_a.limits_enabled is True
    assert effective_turn_limits(org_a)[1:] == (12, 1)
    org_b = SettingService(Session(engine), TenantScope(org_mode=True, org_id=ORG_B)).load()
    assert effective_turn_limits(org_b) == _defaults()
    org_c = SettingService(Session(engine), TenantScope(org_mode=True, org_id=ORG_C)).load()
    assert org_c.limits_enabled is False


def test_an_installs_saved_budget_turns_its_switch_on(engine):
    """Desktop rows have no scope. A global row also reaches every org without
    its own, like the budget it sits next to, so both keep applying together."""
    with Session(engine) as s:
        s.add(Setting(key="max_turn_tokens", value="50000000"))  # "No limit"
        s.commit()

    assert enable_limits_for_saved_budgets(Session(engine)) is True
    assert _switch_rows(engine) == {(None, None, None, "True")}
    assert effective_turn_limits(SettingService(Session(engine), LOCAL_SCOPE).load()).max_turn_tokens == 50_000_000
    org = SettingService(Session(engine), TenantScope(org_mode=True, org_id=ORG_C)).load()
    assert effective_turn_limits(org).max_turn_tokens == 50_000_000


def test_a_replica_that_loses_the_race_skips_that_scope(engine):
    """Replicas boot together; the second insert for a scope must not fail the boot."""
    with Session(engine) as s:
        s.add(Setting(key="max_tool_rounds", value="12", scope="org", org_id=ORG_A))
        s.add(Setting(key="max_tool_rounds", value="20", scope="org", org_id=ORG_B))
        # Another replica already wrote org A's switch after this one read.
        s.add(Setting(key="limits_enabled", value="True", scope="org", org_id=ORG_A))
        s.commit()

    session = Session(engine)
    real_exec = session.exec

    class _Stale:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    def stale_exec(statement, *args, **kwargs):
        rows = real_exec(statement, *args, **kwargs).all()
        return _Stale([r for r in rows if r.key != "limits_enabled"])

    session.exec = stale_exec  # type: ignore[method-assign]
    assert enable_limits_for_saved_budgets(session) is True
    assert _switch_rows(engine) == {("org", ORG_A, None, "True"), ("org", ORG_B, None, "True")}


def test_the_settings_response_carries_the_safety_limits(engine, monkeypatch):
    """Settings shows these, greyed out, while the switch is off: the values in
    force, not the user's saved ones, which stay untouched."""
    monkeypatch.setenv("COWORK_DEFAULT_MAX_TOOL_ROUNDS", "300")
    get_app_settings.cache_clear()
    svc = SettingService(Session(engine), LOCAL_SCOPE)
    svc.upsert_setting("max_tool_rounds", "12")

    rows = {r.key: r for r in svc.list_settings()}
    assert rows["max_tool_rounds"].value == "12"
    assert rows["max_tool_rounds"].default == "300"
    assert rows["max_turn_tokens"].default == str(default_turn_limits().max_turn_tokens)
    assert rows["max_continuations"].default == str(default_turn_limits().max_continuations)
    # Only the budgets carry one.
    assert rows["limits_enabled"].default is None
    assert rows["greeting"].default is None
    assert svc.get_setting("max_tool_rounds").default == "300"
