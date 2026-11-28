"""Unattended turns refresh a stale saved model list first (cowork.services.model_list_refresh).

A channel message or a scheduled run resolves its model from the saved
MindsHub model list. When that copy is older than the window it is refreshed
from MindsHub first, so a model zero data retention now blocks is not picked.
A failed refresh never stops the turn.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.common.settings import user_settings as us
from cowork.common.settings.app_settings import MINDS_FREE_MODEL
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.services import model_list_refresh as refresh
from cowork.services.providers import MindsModelListing
from cowork.services.settings import SettingService
from _fakes import units_on_engine

ORG = "org-refresh"
MEMBER = "u-refresh"
OTHER_MODEL = "kimi"


def _listing(*, zdr: bool) -> MindsModelListing:
    enabled = {MINDS_FREE_MODEL: not zdr, OTHER_MODEL: True}
    reasons = {MINDS_FREE_MODEL: "zdr_restricted"} if zdr else {}
    return MindsModelListing([MINDS_FREE_MODEL, OTHER_MODEL], {}, enabled, {}, {}, {}, {}, reasons, {}, {})


@pytest.fixture()
def session(monkeypatch):
    import cowork.models.setting  # noqa: F401

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    units_on_engine(monkeypatch, engine)
    refresh.reset_for_tests()
    with Session(engine) as s:
        yield s
    refresh.reset_for_tests()
    us.get_app_settings.cache_clear()


def _mode(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("COWORK_TENANCY_MODE", mode)
    us.get_app_settings.cache_clear()


@pytest.fixture()
def org(session, monkeypatch):
    """Org mode on MindsHub; MindsHub's listing is ``state.listing``, minted keys are recorded."""
    from types import SimpleNamespace

    from cowork.turnqueue import producer

    _mode(monkeypatch, "org")
    SettingService(session, TenantScope(org_mode=True, org_id=ORG)).upsert_setting("planning_provider", "minds_cloud")
    state = SimpleNamespace(listing=_listing(zdr=True), minted=[], revoked=[], fetched=[])

    async def mint(*, org_id, user_id, correlation_id, **_kw):
        state.minted.append((org_id, user_id))
        return {"api_key": f"turn-key-{correlation_id}", "base_url": "https://minds.example/v1"}

    async def revoke(correlation_id):
        state.revoked.append(correlation_id)

    async def fetch(base_url, api_key, **kw):
        state.fetched.append((api_key, kw.get("tenant_key"), kw.get("user_id")))
        if isinstance(state.listing, Exception):
            raise state.listing
        return state.listing

    monkeypatch.setattr(producer, "mint_llm_block", mint)
    monkeypatch.setattr(producer, "revoke_turn_key_quietly", revoke)
    monkeypatch.setattr(refresh, "fetch_minds_models", fetch)
    return state


def _blocked(session, scope) -> list[str]:
    return json.loads(SettingService(session, scope).load().minds_model_zdr_blocked)


@pytest.mark.parametrize("user_id", [MEMBER, None], ids=["scheduled-run", "channel-turn"])
def test_a_stale_list_is_refreshed_on_a_key_minted_for_the_call(session, org, user_id):
    scope = TenantScope(org_mode=True, org_id=ORG, user_id=user_id)
    asyncio.run(refresh.refresh_saved_model_list_if_stale(scope))

    assert _blocked(session, TenantScope(org_mode=True, org_id=ORG, user_id="u-other")) == [MINDS_FREE_MODEL]
    enabled = json.loads(SettingService(session, scope).load().minds_model_enabled)
    assert enabled[MINDS_FREE_MODEL] is False
    assert org.minted == [(ORG, user_id)]
    ((key, tenant, user),) = org.fetched
    assert key.startswith("turn-key-") and (tenant, user) == (ORG, user_id)
    assert len(org.revoked) == 1 and key.endswith(org.revoked[0])


def test_a_fresh_list_is_not_refreshed_again(session, org):
    scope = TenantScope(org_mode=True, org_id=ORG, user_id=MEMBER)
    asyncio.run(refresh.refresh_saved_model_list_if_stale(scope))
    asyncio.run(refresh.refresh_saved_model_list_if_stale(scope))
    assert len(org.fetched) == 1


def test_a_list_older_than_the_window_is_refreshed_again(session, org, monkeypatch):
    scope = TenantScope(org_mode=True, org_id=ORG, user_id=MEMBER)
    asyncio.run(refresh.refresh_saved_model_list_if_stale(scope))
    stamped = refresh._refreshed_at[(ORG, MEMBER)]
    monkeypatch.setattr(refresh.time, "monotonic", lambda: stamped + refresh.SAVED_MODEL_LIST_MAX_AGE_S + 1)
    org.listing = _listing(zdr=False)
    asyncio.run(refresh.refresh_saved_model_list_if_stale(scope))
    assert len(org.fetched) == 2
    assert _blocked(session, scope) == []


def test_a_settings_load_counts_as_a_refresh(session, org):
    scope = TenantScope(org_mode=True, org_id=ORG, user_id=MEMBER)
    refresh.note_saved_model_list_refreshed(scope)
    asyncio.run(refresh.refresh_saved_model_list_if_stale(scope))
    assert org.fetched == []


@pytest.mark.parametrize("failure", [RuntimeError("MindsHub down"), "empty"])
def test_a_failed_refresh_keeps_the_saved_copy_and_never_raises(session, org, failure):
    scope = TenantScope(org_mode=True, org_id=ORG, user_id=MEMBER)
    SettingService(session, scope).upsert_setting("minds_model_zdr_blocked", json.dumps([OTHER_MODEL]))
    org.listing = MindsModelListing(None, {}, {}, {}, {}, {}, {}, {}, {}, {}) if failure == "empty" else failure

    asyncio.run(refresh.refresh_saved_model_list_if_stale(scope))

    assert _blocked(session, scope) == [OTHER_MODEL]
    assert len(org.revoked) == 1
    # Stamped all the same, so the next message doesn't ask again at once.
    asyncio.run(refresh.refresh_saved_model_list_if_stale(scope))
    assert len(org.fetched) == 1


def test_an_org_off_mindshub_mints_nothing(session, org):
    scope = TenantScope(org_mode=True, org_id=ORG, user_id=MEMBER)
    service = SettingService(session, TenantScope(org_mode=True, org_id=ORG))
    service.upsert_setting("anthropic_api_key", "sk-ant-org")
    for role in ("planning_provider", "coding_provider", "router_provider"):
        service.upsert_setting(role, "anthropic")
    asyncio.run(refresh.refresh_saved_model_list_if_stale(scope))
    assert org.minted == [] and org.fetched == []


def test_the_desktop_refreshes_with_its_stored_key(session, monkeypatch):
    _mode(monkeypatch, "local")
    asked = []

    async def fetch(base_url, api_key, **kw):
        asked.append(api_key)
        return _listing(zdr=True)

    monkeypatch.setattr(refresh, "fetch_minds_models", fetch)
    service = SettingService(session, LOCAL_SCOPE)
    service.upsert_setting("minds_api_key", "mdb_desktop")
    asyncio.run(refresh.refresh_saved_model_list_if_stale(LOCAL_SCOPE))

    assert asked == ["mdb_desktop"]
    assert _blocked(session, LOCAL_SCOPE) == [MINDS_FREE_MODEL]
