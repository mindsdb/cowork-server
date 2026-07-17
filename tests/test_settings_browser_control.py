"""A1: `browser_control_enabled` round-trip.

The renderer flips this on tab approval via
PUT /api/v1/settings/browser_control_enabled {"value": "true"}; the harness
tool gate (`_select_session_tools`) reads the TYPED value from
`SettingService.load()`, so the upsert must load back as a real bool.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.server import app
from cowork.services.settings import SettingService

client = TestClient(app, client=("127.0.0.1", 50000))


@pytest.fixture()
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def test_browser_control_enabled_put_round_trips_typed_bool(session):
    try:
        r = client.put(
            "/api/v1/settings/browser_control_enabled", json={"value": "true"}
        )
        assert r.status_code == 200
        assert SettingService(session).load().browser_control_enabled is True

        # Symmetric off-switch (SettingsView toggle).
        r = client.put(
            "/api/v1/settings/browser_control_enabled", json={"value": "false"}
        )
        assert r.status_code == 200
        assert SettingService(session).load().browser_control_enabled is False
    finally:
        try:
            SettingService(session).delete_setting("browser_control_enabled")
        except ValueError:
            pass
