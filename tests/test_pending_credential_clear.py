"""A desktop sign-out's pending credential clear, finished at server startup."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import uvicorn
from sqlmodel import Session, select
from uvicorn.lifespan.on import LifespanOn

from cowork import server
from cowork.api.v1.endpoints.health import health
from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import invalidate_user_settings_cache
from cowork.db.migrations import run_schema_migrations
from cowork.db.session import get_engine
from cowork.models.setting import Setting
from cowork.services import pending_credential_clear as pcc
from cowork.services.pending_credential_clear import (
    PENDING_CLEAR_MARKER,
    PendingCredentialClearFailed,
    consume_pending_credential_clear,
)
from cowork.services.settings import SettingService

CREDENTIAL_KEYS = ["anthropic_api_key", "openai_api_key", "providers_json"]
PREFERENCE_KEY = "planning_model"


def _seed(database_uri: str) -> None:
    with Session(get_engine(database_uri)) as session:
        for key in [*CREDENTIAL_KEYS, PREFERENCE_KEY]:
            session.add(Setting(key=key, value="seeded"))
        session.commit()


def _keys(database_uri: str) -> set[str]:
    with Session(get_engine(database_uri)) as session:
        rows = session.exec(select(Setting).where(Setting.scope.is_(None))).all()
        return {row.key for row in rows}


@pytest.fixture
def store(tmp_path: Path) -> tuple[str, Path]:
    """A migrated store outside COWORK_HOME, as an org store's DATABASE_URI is."""
    database_uri = f"sqlite:///{tmp_path / 'cowork.db'}"
    run_schema_migrations(get_engine(database_uri), database_uri)
    _seed(database_uri)
    return database_uri, tmp_path / PENDING_CLEAR_MARKER


def test_marker_clears_credentials_keeps_preferences_and_is_removed(store):
    database_uri, marker = store
    marker.touch()

    assert consume_pending_credential_clear(database_uri, "local") is True

    assert _keys(database_uri) == {PREFERENCE_KEY}
    assert not marker.exists()


def test_no_marker_leaves_every_row(store):
    database_uri, marker = store

    assert consume_pending_credential_clear(database_uri, "local") is False

    assert _keys(database_uri) == {*CREDENTIAL_KEYS, PREFERENCE_KEY}


def test_org_tenancy_ignores_the_marker(store):
    database_uri, marker = store
    marker.touch()

    assert consume_pending_credential_clear(database_uri, "org") is False

    assert _keys(database_uri) == {*CREDENTIAL_KEYS, PREFERENCE_KEY}
    assert marker.exists()


def test_failed_clear_keeps_the_marker_and_the_next_start_finishes_it(store, monkeypatch):
    database_uri, marker = store
    marker.touch()

    def failing_clear(self):
        raise RuntimeError("database is locked")

    with monkeypatch.context() as patch:
        patch.setattr(SettingService, "clear_credentials", failing_clear)
        with pytest.raises(PendingCredentialClearFailed):
            consume_pending_credential_clear(database_uri, "local")
    assert marker.exists()

    assert consume_pending_credential_clear(database_uri, "local") is True
    assert _keys(database_uri) == {PREFERENCE_KEY}
    assert not marker.exists()


def test_a_transient_unlink_failure_is_retried(store, monkeypatch):
    database_uri, marker = store
    marker.touch()
    real_unlink = Path.unlink
    failures = iter([PermissionError("in use"), PermissionError("in use")])

    def flaky_unlink(self, *args, **kwargs):
        error = next(failures, None)
        if error is not None:
            raise error
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    monkeypatch.setattr(pcc, "_UNLINK_RETRY_S", 0)

    assert consume_pending_credential_clear(database_uri, "local") is True
    assert not marker.exists()


def test_a_marker_that_cannot_be_removed_fails_the_start(store, monkeypatch):
    database_uri, marker = store
    marker.touch()

    def stuck_unlink(self, *args, **kwargs):
        raise PermissionError("in use")

    monkeypatch.setattr(Path, "unlink", stuck_unlink)
    monkeypatch.setattr(pcc, "_UNLINK_RETRY_S", 0)

    with pytest.raises(PendingCredentialClearFailed):
        consume_pending_credential_clear(database_uri, "local")
    assert marker.exists()


def test_failure_log_names_the_store_but_no_credential(store, monkeypatch, caplog):
    database_uri, marker = store
    marker.touch()

    def failing_clear(self):
        raise RuntimeError("DELETE FROM settings WHERE key = 'anthropic_api_key'")

    monkeypatch.setattr(SettingService, "clear_credentials", failing_clear)
    caplog.set_level(logging.ERROR, logger=pcc.__name__)

    with pytest.raises(PendingCredentialClearFailed) as raised:
        consume_pending_credential_clear(database_uri, "local")

    messages = [record.getMessage() for record in caplog.records if record.name == pcc.__name__]
    assert any("pending credential clear failed" in message for message in messages), messages
    for text in [*messages, str(raised.value)]:
        assert "api_key" not in text
    assert all(record.exc_info is None for record in caplog.records if record.name == pcc.__name__)
    assert raised.value.__context__ is None


def test_health_advertises_the_capability():
    assert "pending-credential-clear" in health()["capabilities"]


class _StopStartup(Exception):
    pass


async def _startup_failed(app) -> bool:
    """Drive the app's lifespan startup as Uvicorn does; True when it failed."""
    config = uvicorn.Config(app, lifespan="on", log_config=None)
    config.load()
    lifespan = LifespanOn(config)
    await lifespan.startup()
    return lifespan.should_exit


@pytest.fixture
def app_store():
    """The suite's own database, marked; its global rows for these keys are put
    back exactly as they were, and no other scope is touched."""
    database_uri = get_app_settings().database.uri
    marker = Path(get_engine(database_uri).url.database).parent / PENDING_CLEAR_MARKER
    keys = [*CREDENTIAL_KEYS, PREFERENCE_KEY]
    global_rows = select(Setting).where(Setting.key.in_(keys), Setting.scope.is_(None))
    with Session(get_engine(database_uri)) as session:
        before = {row.key: row.value for row in session.exec(global_rows).all()}
        for key in keys:
            if key not in before:
                session.add(Setting(key=key, value="seeded"))
        session.commit()
        # Encrypted, as a real saved key is, so the store reads as configured.
        SettingService(session, None).upsert_setting("anthropic_api_key", "test-key-not-real")
    marker.touch()
    yield database_uri, marker
    marker.unlink(missing_ok=True)
    with Session(get_engine(database_uri)) as session:
        for row in session.exec(global_rows).all():
            session.delete(row)
        session.flush()
        for key, value in before.items():
            session.add(Setting(key=key, value=value))
        session.commit()
    invalidate_user_settings_cache()


async def test_startup_clears_after_setup_and_before_anything_else_runs(app_store, monkeypatch):
    database_uri, marker = app_store
    seen: dict[str, object] = {}

    def recording_setup() -> None:
        seen["marker_during_setup"] = marker.exists()
        seen["config_ready_before"] = health()["config_ready"]

    def stop_after_clear() -> None:
        seen["keys_after_clear"] = _keys(database_uri)
        seen["config_ready"] = health()["config_ready"]
        raise _StopStartup()

    monkeypatch.setattr(server, "run_dev_setup", recording_setup)
    monkeypatch.setattr(server, "_warm_agent_runtime", stop_after_clear)

    await _startup_failed(server.create_app())

    assert seen["marker_during_setup"] is True
    keys_after_clear = seen["keys_after_clear"]
    assert isinstance(keys_after_clear, set)
    assert keys_after_clear.isdisjoint(CREDENTIAL_KEYS)
    assert PREFERENCE_KEY in keys_after_clear
    assert seen["config_ready_before"] is True
    assert seen["config_ready"] is False
    assert not marker.exists()


async def test_a_failed_clear_fails_startup(app_store, monkeypatch):
    database_uri, marker = app_store
    went_on: list[bool] = []

    def failing_clear(self):
        raise RuntimeError("database is locked")

    def stop_if_reached() -> None:
        went_on.append(True)
        raise _StopStartup()

    monkeypatch.setattr(server, "run_dev_setup", lambda: None)
    monkeypatch.setattr(server, "_warm_agent_runtime", stop_if_reached)
    monkeypatch.setattr(SettingService, "clear_credentials", failing_clear)

    assert await _startup_failed(server.create_app()) is True
    assert went_on == []
    assert marker.exists()
    assert set(CREDENTIAL_KEYS) <= _keys(database_uri)


async def test_a_fresh_store_is_migrated_before_its_pending_clear(tmp_path, monkeypatch):
    """An organization store a sign-out marked before its database existed."""
    database_uri = f"sqlite:///{tmp_path / 'cowork.db'}"
    marker = tmp_path / PENDING_CLEAR_MARKER
    marker.touch()
    reached: list[bool] = []

    def stop_after_clear() -> None:
        reached.append(True)
        raise _StopStartup()

    monkeypatch.setenv("DATABASE_URI", database_uri)
    get_app_settings.cache_clear()
    try:
        monkeypatch.setattr(server, "_warm_agent_runtime", stop_after_clear)
        await _startup_failed(server.create_app())
    finally:
        get_app_settings.cache_clear()

    assert reached == [True]
    assert not marker.exists()
    assert _keys(database_uri).isdisjoint(CREDENTIAL_KEYS)
