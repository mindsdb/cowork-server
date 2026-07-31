"""ENG-1126: atomic, side-effect-free settings writes.

save_all is all-or-nothing (a Settings save can't half-apply), and test-providers
is read-only (a "test" no longer silently writes provider_status).
"""
import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints.settings import bulk_upsert_settings
from cowork.db.session import get_open_session
from cowork.schemas.settings import SettingsBulkUpsertRequest
from cowork.services.settings import SettingService


def _cleanup(session, *keys):
    svc = SettingService(session)
    for k in keys:
        try:
            svc.delete_setting(k)
        except ValueError:
            pass


def test_save_all_writes_every_key_in_one_transaction():
    session = get_open_session()
    try:
        _cleanup(session, "greeting", "tone", "planning_provider")
        svc = SettingService(session)
        written = svc.save_all(
            {"greeting": "hello", "tone": "formal", "planning_provider": "openai"}
        )
        assert set(written) == {"greeting", "tone", "planning_provider"}
        loaded = svc.load()
        assert loaded.greeting == "hello"
        assert loaded.tone == "formal"
        assert loaded.planning_provider.value == "openai"
    finally:
        _cleanup(session, "greeting", "tone", "planning_provider")
        session.close()


def test_save_all_is_all_or_nothing_on_an_invalid_value():
    session = get_open_session()
    try:
        _cleanup(session, "greeting", "planning_provider")
        svc = SettingService(session)
        with pytest.raises(ValueError):
            svc.save_all(
                {"greeting": "written?", "planning_provider": "not-a-provider"}
            )
        # The valid key in the same batch must NOT have been written.
        assert svc._fetch_row("greeting") is None
    finally:
        _cleanup(session, "greeting", "planning_provider")
        session.close()


def test_save_all_skips_masked_and_none():
    session = get_open_session()
    try:
        _cleanup(session, "anthropic_api_key", "greeting")
        svc = SettingService(session)
        written = svc.save_all({"anthropic_api_key": "***", "greeting": "hi"})
        assert written == ["greeting"]
        assert svc._fetch_row("anthropic_api_key") is None
    finally:
        _cleanup(session, "anthropic_api_key", "greeting")
        session.close()


def test_bulk_endpoint_400s_on_invalid_and_writes_nothing():
    session = get_open_session()
    try:
        _cleanup(session, "tone", "planning_provider")
        with pytest.raises(HTTPException) as exc:
            bulk_upsert_settings(
                SettingsBulkUpsertRequest(
                    values={"tone": "casual", "planning_provider": "nope"}
                ),
                session,
            )
        assert exc.value.status_code == 400
        assert SettingService(session)._fetch_row("tone") is None
    finally:
        _cleanup(session, "tone", "planning_provider")
        session.close()


async def test_test_providers_does_not_persist(monkeypatch):
    from cowork.api.v1.endpoints import settings as ep

    async def fake_ping(providers):
        return {"anthropic": "ok"}, {"anthropic": "connected"}

    monkeypatch.setattr(ep, "ping_providers", fake_ping)

    session = get_open_session()
    try:
        _cleanup(session, "anthropic_api_key", "provider_status", "provider_status_details")
        SettingService(session).upsert_setting("anthropic_api_key", "sk-test")

        result = await ep.test_providers(
            session,
            ep._TestProvidersBody(providers=[{"type": "anthropic", "apiKey": "***"}]),
        )
        assert result["providerStatus"] == {"anthropic": "ok"}
        # The point of ENG-1126: a test writes nothing.
        svc = SettingService(session)
        assert svc._fetch_row("provider_status") is None
        assert svc._fetch_row("provider_status_details") is None
    finally:
        _cleanup(session, "anthropic_api_key", "provider_status", "provider_status_details")
        session.close()
