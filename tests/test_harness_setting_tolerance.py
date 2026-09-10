"""A harness id we no longer ship must not break settings.

The value can arrive from a stored row at any scope, from ~/.cowork/.env or
process env (local mode), so the validator resolves unknown ids to anton
instead of raising. Writers persist the validated value.
"""
from __future__ import annotations

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from cowork.common.settings.user_settings import UserSettings
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.models.setting import Setting
from cowork.services.settings import SettingService

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
USER_A = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


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


def _svc(engine, scope: TenantScope = LOCAL_SCOPE) -> SettingService:
    return SettingService(Session(engine), scope)


def test_unknown_harness_values_resolve_to_anton():
    s = UserSettings(harness="hermes", channels_harness="hermes")
    assert s.harness == "anton"
    assert s.channels_harness == "anton"


def test_stale_rows_at_every_scope_resolve_to_anton(engine):
    with Session(engine) as session:
        session.add(Setting(key="harness", value="hermes"))
        session.add(Setting(key="greeting", value="hello there"))
        session.add(Setting(key="channels_harness", value="hermes", scope="org", org_id=ORG_A))
        session.add(Setting(key="harness", value="hermes", scope="user", org_id=ORG_A, user_id=USER_A))
        session.commit()

    local = _svc(engine).load()
    assert local.harness == "anton"
    assert local.greeting == "hello there"

    org = _svc(engine, TenantScope(org_mode=True, org_id=ORG_A)).load()
    assert org.channels_harness == "anton"

    user = _svc(engine, TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A)).load()
    assert user.harness == "anton"


def test_env_harness_resolves_to_anton_in_local_mode(monkeypatch):
    monkeypatch.setenv("HARNESS", "hermes")
    monkeypatch.setenv("COWORK_CHANNELS_HARNESS", "hermes")
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    try:
        s = UserSettings()
        assert s.harness == "anton"
        assert s.channels_harness == "anton"
    finally:
        get_app_settings.cache_clear()


def test_writing_an_unknown_harness_stores_anton(engine):
    _svc(engine).upsert_setting("harness", "hermes")
    with Session(engine) as session:
        row = session.exec(select(Setting).where(Setting.key == "harness")).one()
    assert row.value == "anton"
