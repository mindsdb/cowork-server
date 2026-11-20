"""Models zero data retention blocks are never a fallback while another is served.

MindsHub marks them ``enabled: false`` with ``disabled_reason: zdr_restricted``.
Unlike a wallet lock, a top-up never lifts that, so ``recommended_models`` caches
their ids in ``minds_model_zdr_blocked`` and resolution skips them when it would
otherwise keep a locked model.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import SecretStr
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints import settings as ep
from cowork.common.settings import user_settings as us
from cowork.common.settings.app_settings import IMAGE_MODEL_DEFAULT, MINDS_FREE_MODEL
from cowork.common.settings.user_settings import Provider, UserSettings, setting_is_org_scoped
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.services.providers import MindsModelListing
from cowork.services.settings import SettingService
from _fakes import FakeRequest, units_on_engine

ORG = "org-zdr"
MEMBER = "u-zdr"
ZDR_MODEL = "kimi"  # served on a ZDR provider, locked by the wallet
FLARE = IMAGE_MODEL_DEFAULT
SUNBURST = "gpt-image-2.6-sunburst"


def _listing(*, zdr: bool) -> MindsModelListing:
    """Air is refused under ZDR; the ZDR model is wallet-locked either way."""
    enabled = {MINDS_FREE_MODEL: not zdr, ZDR_MODEL: False}
    reasons = {ZDR_MODEL: "wallet_empty"}
    if zdr:
        reasons[MINDS_FREE_MODEL] = "zdr_restricted"
    return MindsModelListing([MINDS_FREE_MODEL, ZDR_MODEL], {}, enabled, {}, {}, {}, {}, reasons, {})


def _mode(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("COWORK_TENANCY_MODE", mode)
    us.get_app_settings.cache_clear()


@pytest.fixture()
def session(monkeypatch):
    import cowork.models.setting  # noqa: F401

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    units_on_engine(monkeypatch, engine)
    with Session(engine) as s:
        yield s
    us.get_app_settings.cache_clear()


def _minds(**kw) -> UserSettings:
    return UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        router_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb_test"),
        **kw,
    )


# -- the cache -------------------------------------------------------------------


def test_org_mode_caches_the_blocked_ids_for_the_org_and_locks_them_in_the_shared_map(session, monkeypatch):
    _mode(monkeypatch, "org")
    zdr = {"on": True}

    async def fake_org_catalog(*, org_id, user_id, bearer_token, refresh=False):
        return _listing(zdr=zdr["on"])

    monkeypatch.setattr(ep, "fetch_org_model_catalog", fake_org_catalog)
    scope = TenantScope(org_mode=True, org_id=ORG, user_id=MEMBER)
    asyncio.run(ep.recommended_models(FakeRequest({"Authorization": "Bearer jwt"}), scope))

    other = SettingService(session, TenantScope(org_mode=True, org_id=ORG, user_id="u-other")).load()
    assert json.loads(other.minds_model_zdr_blocked) == [MINDS_FREE_MODEL]
    assert json.loads(other.minds_model_enabled)[MINDS_FREE_MODEL] is False

    zdr["on"] = False
    asyncio.run(ep.recommended_models(FakeRequest({"Authorization": "Bearer jwt"}), scope))
    assert json.loads(SettingService(session, scope).load().minds_model_zdr_blocked) == []


def test_desktop_caches_the_blocked_ids(session, monkeypatch):
    _mode(monkeypatch, "local")

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None, **_kw):
        return _listing(zdr=True)

    monkeypatch.setattr(ep, "fetch_minds_models", fake_fetch)
    service = SettingService(session, LOCAL_SCOPE)
    service.upsert_setting("minds_api_key", "mdb_desktop")
    service.upsert_setting("planning_provider", "minds_cloud")
    asyncio.run(ep.recommended_models(FakeRequest(), LOCAL_SCOPE))

    assert json.loads(SettingService(session, LOCAL_SCOPE).load().minds_model_zdr_blocked) == [MINDS_FREE_MODEL]


def test_the_blocked_list_is_an_org_setting():
    assert setting_is_org_scoped("minds_model_zdr_blocked") is True


# -- resolution ------------------------------------------------------------------

LOCKED = json.dumps({MINDS_FREE_MODEL: False, ZDR_MODEL: False})
BLOCKED = json.dumps([MINDS_FREE_MODEL])


@pytest.mark.parametrize("mode", ["org", "local"])
def test_a_blocked_router_pin_moves_to_a_locked_zdr_model(monkeypatch, mode):
    # Desktop summaries run on the router model.
    _mode(monkeypatch, mode)
    s = _minds(router_model=MINDS_FREE_MODEL, minds_model_enabled=LOCKED, minds_model_zdr_blocked=BLOCKED)
    assert s.resolved_router_model == ZDR_MODEL


def test_a_wallet_locked_pin_without_zdr_is_kept_for_its_top_up_path(monkeypatch):
    _mode(monkeypatch, "local")
    s = _minds(router_model=MINDS_FREE_MODEL, minds_model_enabled=LOCKED)
    assert s.resolved_router_model == MINDS_FREE_MODEL


def test_org_defaults_skip_a_blocked_free_model(monkeypatch):
    _mode(monkeypatch, "org")
    s = _minds(minds_model_enabled=LOCKED, minds_model_zdr_blocked=BLOCKED)
    assert s.planning_model == s.coding_model == s.router_model == ZDR_MODEL


def test_org_defaults_keep_the_free_model_when_only_the_wallet_locks_it(monkeypatch):
    _mode(monkeypatch, "org")
    s = _minds(minds_model_enabled=LOCKED)
    assert s.planning_model == MINDS_FREE_MODEL


def test_an_enabled_model_still_wins_over_any_fallback(monkeypatch):
    _mode(monkeypatch, "org")
    enabled = json.dumps({MINDS_FREE_MODEL: False, ZDR_MODEL: True})
    s = _minds(router_model=MINDS_FREE_MODEL, minds_model_enabled=enabled, minds_model_zdr_blocked=BLOCKED)
    assert s.resolved_router_model == ZDR_MODEL


@pytest.mark.parametrize(
    "blocked, handed",
    [
        pytest.param([FLARE, SUNBURST], None, id="every-image-model-blocked"),
        pytest.param([FLARE], SUNBURST, id="blocked-pick-moves-to-a-wallet-locked-model"),
    ],
)
def test_a_blocked_image_model_is_never_handed_to_anton(blocked, handed):
    s = _minds(
        image_model=FLARE,
        minds_image_model_enabled=json.dumps({FLARE: False, SUNBURST: False}),
        minds_model_zdr_blocked=json.dumps(blocked),
    )
    assert s.image_model_for_agent == handed


def test_an_unreadable_blocked_list_blocks_nothing(monkeypatch):
    _mode(monkeypatch, "local")
    s = _minds(router_model=MINDS_FREE_MODEL, minds_model_enabled=LOCKED, minds_model_zdr_blocked="{bad")
    assert s.resolved_router_model == MINDS_FREE_MODEL
