"""The MindsHub credential the desktop app hands over instead of storing.

The overlay lives in ``SettingService._raw_data``, which every reader of
``get_user_settings()`` goes through, so these tests exercise the resolved
``UserSettings`` rather than the holder alone: that is what the harnesses, the
publish path and the readiness gate actually read.
"""

import pytest
from fastapi.testclient import TestClient

from cowork.common.settings import runtime_credential
from cowork.common.settings.app_settings import get_app_settings

_CREDENTIAL_KEYS = (
    "minds_api_key",
    "anthropic_api_key",
    "openai_api_key",
    "gemini_api_key",
    "openai_compatible_api_key",
)


@pytest.fixture(autouse=True)
def clear_hand_over():
    """The holder is a module global, so a value left behind would change every
    later test's idea of whether MindsHub is configured."""
    runtime_credential.clear_minds_credential()
    yield
    runtime_credential.clear_minds_credential()


@pytest.fixture
def session():
    from cowork.db.session import get_open_session

    db = get_open_session()
    try:
        yield db
    finally:
        db.close()


def _clear_rows(session, *keys: str) -> None:
    from cowork.services.settings import SettingService

    service = SettingService(session)
    for key in keys:
        try:
            service.delete_setting(key)
        except ValueError:
            pass


@pytest.fixture
def no_stored_credentials(session):
    _clear_rows(session, *_CREDENTIAL_KEYS)
    yield
    _clear_rows(session, *_CREDENTIAL_KEYS)


def _loaded(session):
    from cowork.services.settings import SettingService

    return SettingService(session).load()


def _minds_key(session) -> str | None:
    value = _loaded(session).minds_api_key
    return value.get_secret_value() if value is not None else None


def test_hand_over_beats_a_stored_row(session, no_stored_credentials):
    # An install upgrading from a build that persisted its key still has that
    # row until the migration clears it. The live credential has to win, or the
    # first turn after the upgrade spends a key the app no longer manages.
    from cowork.services.settings import SettingService

    SettingService(session).upsert_setting("minds_api_key", "mdb_stored")
    runtime_credential.set_minds_credential("handed-over-token")

    assert _minds_key(session) == "handed-over-token"


def test_stored_row_answers_when_nothing_was_handed_over(session, no_stored_credentials):
    from cowork.services.settings import SettingService

    SettingService(session).upsert_setting("minds_api_key", "mdb_stored")

    assert _minds_key(session) == "mdb_stored"


def test_clearing_falls_back_to_the_stored_row(session, no_stored_credentials):
    from cowork.services.settings import SettingService

    SettingService(session).upsert_setting("minds_api_key", "mdb_stored")
    runtime_credential.set_minds_credential("handed-over-token")
    runtime_credential.clear_minds_credential()

    assert _minds_key(session) == "mdb_stored"


def test_a_blank_hand_over_clears_rather_than_storing_empty(session, no_stored_credentials):
    # Matches how _raw_data already treats a blank stored credential: no
    # credential, so the provider reads as unconfigured rather than configured
    # with something that cannot work.
    runtime_credential.set_minds_credential("handed-over-token")
    runtime_credential.set_minds_credential("")

    assert runtime_credential.get_minds_credential() is None
    assert _minds_key(session) is None


def test_the_hand_over_alone_makes_the_app_configured(session, no_stored_credentials):
    # config_status is what /health reports as config_ready, which decides
    # whether a launch lands on the app or on onboarding. With nothing stored,
    # the hand-over is the only thing that can answer.
    assert _loaded(session).config_status["config_ready"] is False

    runtime_credential.set_minds_credential("handed-over-token")

    status = _loaded(session).config_status
    assert status["config_ready"] is True
    assert status["config_error"] is None


def test_the_route_hands_over_and_clears():
    from cowork.server import create_app

    # No context manager: entering one runs the lifespan, which migrates the DB
    # the conftest already built from the models. Every TestClient in this
    # suite is built the same way.
    client = TestClient(create_app(), client=("127.0.0.1", 40000))

    response = client.put("/api/v1/runtime-credential/minds", json={"value": "from-the-route"})
    assert response.status_code == 200
    assert runtime_credential.get_minds_credential() == "from-the-route"

    response = client.put("/api/v1/runtime-credential/minds", json={"value": ""})
    assert response.status_code == 200
    assert runtime_credential.get_minds_credential() is None


def test_the_route_refuses_a_caller_that_is_not_loopback():
    # The route accepts a bearer token, so a network-exposed deployment must
    # not let a remote peer choose which credential the agent spends.
    from cowork.server import create_app

    client = TestClient(create_app(), client=("10.1.2.3", 40000))
    response = client.put("/api/v1/runtime-credential/minds", json={"value": "from-elsewhere"})

    assert response.status_code == 403
    assert runtime_credential.get_minds_credential() is None


def test_org_mode_ignores_a_hand_over(monkeypatch):
    # An org pod is handed a per-turn credential and never a stored one, so a
    # value accepted here would be one tenant's credential answering for every
    # tenant. The route guard refuses first; this is the layer under it.
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    try:
        runtime_credential.set_minds_credential("handed-over-token")
        assert runtime_credential.get_minds_credential() is None
    finally:
        get_app_settings.cache_clear()
