"""A turn reads its settings once, in its first database unit, and every
later reader in the turn, in the gate and in the producer task, gets that
snapshot instead of opening another connection."""
from __future__ import annotations

import asyncio
import threading
from uuid import uuid4

import httpx
import pytest

import cowork.common.settings.user_settings as user_settings
from cowork.common.settings import runtime_credential
import cowork.handlers.responses as responses_mod
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.handlers.response_routing import DELEGATED_AGENTIC, RouteDecision
from cowork.server import create_app
from cowork.services.settings import SettingService
from cowork.streaming import registry

from _fakes import PausedHarness


@pytest.fixture(autouse=True)
def _forget_turns():
    yield
    registry.reset()


async def test_a_streamed_turn_loads_its_settings_once_off_the_event_loop(monkeypatch):
    loads: list[int] = []
    load = SettingService.load

    def counted_load(self):
        loads.append(threading.get_ident())
        return load(self)

    monkeypatch.setattr(SettingService, "load", counted_load)
    read_in_gate, read_in_producer = [], []

    async def decide(**_kwargs):
        # As the gate's own binding does (response_routing._settings_binding).
        read_in_gate.append(user_settings.get_user_settings())
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    class _ReadsSettings(PausedHarness):
        def stream_response(self, **kwargs):
            # As AntonHarness does while it builds the turn.
            read_in_producer.append(user_settings.get_user_settings())
            return super().stream_response(**kwargs)

    gate = _ReadsSettings()
    gate.release.set()
    monkeypatch.setattr(responses_mod, "decide_route", decide)
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: gate)

    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        answered = await asyncio.wait_for(
            client.post(
                "/api/v1/responses/",
                json={"input": "hi", "stream": True, "conversation": str(uuid4())},
            ),
            timeout=10,
        )

    assert answered.status_code == 200 and "response.completed" in answered.text, answered.text
    assert len(read_in_gate) == 1 and len(read_in_producer) == 1
    assert len(loads) == 1, f"settings loaded {len(loads)} times in one turn"
    assert loads[0] != threading.get_ident(), "settings were loaded on the event loop's thread"


def test_the_turn_snapshot_answers_only_for_its_own_scope(monkeypatch):
    loaded: list[TenantScope] = []
    monkeypatch.setattr(
        user_settings, "_load_from_db", lambda scope: loaded.append(scope) or user_settings.UserSettings(),
    )
    snapshot = user_settings.UserSettings(harness="anton")
    other_org = TenantScope(org_mode=True, org_id="org-b", user_id="user-b")

    with user_settings.use_turn_settings(LOCAL_SCOPE, snapshot):
        served = user_settings.get_user_settings()
        served.harness = "changed by one caller"
        served_again = user_settings.get_user_settings(LOCAL_SCOPE)
        assert loaded == []
        user_settings.get_user_settings(other_org)
        assert loaded == [other_org]
    user_settings.get_user_settings()

    assert served_again.harness == "anton"
    assert loaded == [other_org, LOCAL_SCOPE]


def test_the_turn_snapshot_serves_the_desktops_current_minds_credential(monkeypatch):
    """The desktop app hands over a short-lived MindsHub token and refreshes
    it while a turn runs. The snapshot keeps every other setting the turn
    started with, but serves the current token, as a fresh load would."""
    monkeypatch.setattr(
        user_settings, "_load_from_db", lambda scope: pytest.fail("the snapshot's scope loaded again"),
    )
    runtime_credential.set_minds_credential("token-at-turn-start")
    try:
        snapshot = user_settings.UserSettings(harness="anton", minds_api_key="token-at-turn-start")
        with user_settings.use_turn_settings(LOCAL_SCOPE, snapshot):
            runtime_credential.set_minds_credential("token-after-refresh")
            served = user_settings.get_user_settings()
    finally:
        runtime_credential.clear_minds_credential()

    assert served.minds_api_key.get_secret_value() == "token-after-refresh"
    assert served.harness == "anton"


def test_the_turn_snapshot_drops_a_runtime_token_that_is_cleared():
    runtime_credential.set_minds_credential("runtime-token")
    try:
        snapshot = SettingService._load([])
        with user_settings.use_turn_settings(LOCAL_SCOPE, snapshot):
            runtime_credential.clear_minds_credential()
            assert user_settings.get_user_settings().minds_api_key is None
    finally:
        runtime_credential.clear_minds_credential()


def test_the_turn_snapshot_preserves_a_stored_key_without_a_runtime_token():
    runtime_credential.clear_minds_credential()
    snapshot = user_settings.UserSettings(minds_api_key="stored-key")
    with user_settings.use_turn_settings(LOCAL_SCOPE, snapshot):
        assert user_settings.get_user_settings().minds_api_key.get_secret_value() == "stored-key"


def test_clearing_a_runtime_token_restores_the_stored_snapshot_key():
    from cowork.common.encryption import encrypt
    from cowork.models.setting import Setting

    runtime_credential.set_minds_credential("runtime-token")
    try:
        snapshot = SettingService._load([Setting(key="minds_api_key", value=encrypt("stored-key"))])
        with user_settings.use_turn_settings(LOCAL_SCOPE, snapshot):
            assert user_settings.get_user_settings().minds_api_key.get_secret_value() == "runtime-token"
            runtime_credential.clear_minds_credential()
            assert user_settings.get_user_settings().minds_api_key.get_secret_value() == "stored-key"
    finally:
        runtime_credential.clear_minds_credential()


def test_settings_validation_sees_the_runtime_token(monkeypatch):
    runtime_credential.set_minds_credential("runtime-token")
    enabled_map = user_settings.UserSettings._minds_enabled_map
    validated = []

    def check_runtime(self):
        assert self.minds_api_key is not None
        assert self.minds_api_key.get_secret_value() == "runtime-token"
        validated.append(True)
        return enabled_map(self)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(user_settings.UserSettings, "_minds_enabled_map", check_runtime)
            SettingService._load([])
        assert validated
    finally:
        runtime_credential.clear_minds_credential()
