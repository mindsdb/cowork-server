"""Warm the MindsHub availability map before the first turn (ENG-748, desktop).

On desktop the minds-cloud role defaults stay the premium canonical models
(`sonnet`/`haiku`); free-tier users are steered off them only by the cached
`minds_model_enabled` availability map. That map is refreshed lazily on GET
/recommended-models, so a brand-new sign-in that sends a message before the
picker ever loads resolves the planning default against an EMPTY map —
`_enabled_aware_default` returns canonical `sonnet` and MindsHub denies the
empty free-tier wallet (`wallet_empty` 402) on message one. That is the live
first-contact cohort in ENG-748.

`warm_enabled_model_map` closes the race at the two guaranteed-pre-first-turn
seams — server startup with a stored key, and immediately after a credential
sync (POST /settings/raw) — by fetching /v1/models and persisting the map
through the same guarded writer the recommended-models endpoint uses
(`persist_enabled_model_map`), so the invariants can't drift.

The org surface is fixed separately by `role_defaults` (see
`test_first_turn_default_floor`); this file covers the desktop surface.
"""
import asyncio
import json

from cowork.common.settings import user_settings as us
from cowork.db.scoped import LOCAL_SCOPE
from cowork.services.settings import SettingService

# Realistic free-tier gateway shape: whole catalog listed, paid models locked,
# the free model enabled and first.
FREE_ENABLED = {"mindshub_air": True, "sonnet": False, "haiku": False, "kimi": False}
PAID_ENABLED = {"mindshub_air": True, "sonnet": True, "haiku": True, "kimi": True}


def _listing(enabled):
    from cowork.services.providers import MindsModelListing

    return MindsModelListing(list(enabled), {}, enabled, {}, {}, {})


def _empty():
    from cowork.services.providers import _empty_listing

    return _empty_listing()


def _set(session, **values):
    svc = SettingService(session)
    for k, v in values.items():
        svc.upsert_setting(k, v)


def _clear(session, *keys):
    svc = SettingService(session)
    for k in keys:
        try:
            svc.delete_setting(k)
        except ValueError:
            pass


def _fresh_session():
    us.get_app_settings.cache_clear()  # ensure local tenancy, uncontaminated
    from cowork.db.session import get_open_session

    return get_open_session()


# ── warm_enabled_model_map ────────────────────────────────────────────

def test_warm_populates_empty_map(monkeypatch):
    from cowork.services import providers

    async def fake_fetch(url, key, *, force_refresh=False, tenant_key=None):
        return _listing(FREE_ENABLED)

    monkeypatch.setattr(providers, "fetch_minds_models", fake_fetch)
    session = _fresh_session()
    try:
        _set(session, minds_api_key="mdb_test", minds_url="https://minds.example/v1")
        changed = asyncio.run(providers.warm_enabled_model_map(session))
        assert changed is True
        assert json.loads(SettingService(session).load().minds_model_enabled) == FREE_ENABLED
    finally:
        _clear(session, "minds_api_key", "minds_url", "minds_model_enabled")
        session.close()


def test_warm_noops_without_a_minds_key(monkeypatch):
    from cowork.services import providers

    called = False

    async def fake_fetch(url, key, *, force_refresh=False, tenant_key=None):
        nonlocal called
        called = True
        return _listing(FREE_ENABLED)

    monkeypatch.setattr(providers, "fetch_minds_models", fake_fetch)
    session = _fresh_session()
    try:
        _clear(session, "minds_api_key", "minds_url", "minds_model_enabled")
        changed = asyncio.run(providers.warm_enabled_model_map(session))
        assert changed is False
        assert called is False  # never hits the network without a key
    finally:
        session.close()


def test_warm_is_fail_open_on_fetch_failure(monkeypatch):
    from cowork.services import providers

    async def fake_fetch(url, key, *, force_refresh=False, tenant_key=None):
        return _empty()  # fetch_minds_models swallows errors -> empty listing

    monkeypatch.setattr(providers, "fetch_minds_models", fake_fetch)
    session = _fresh_session()
    try:
        _set(session, minds_api_key="mdb_test", minds_url="https://minds.example/v1",
             minds_model_enabled=json.dumps(FREE_ENABLED))
        changed = asyncio.run(providers.warm_enabled_model_map(session))
        assert changed is False
        # A known-good map is never clobbered by a failed fetch.
        assert json.loads(SettingService(session).load().minds_model_enabled) == FREE_ENABLED
    finally:
        _clear(session, "minds_api_key", "minds_url", "minds_model_enabled")
        session.close()


def test_warm_is_a_noop_when_map_already_current(monkeypatch):
    from cowork.services import providers

    async def fake_fetch(url, key, *, force_refresh=False, tenant_key=None):
        return _listing(FREE_ENABLED)

    monkeypatch.setattr(providers, "fetch_minds_models", fake_fetch)
    session = _fresh_session()
    try:
        _set(session, minds_api_key="mdb_test", minds_url="https://minds.example/v1",
             minds_model_enabled=json.dumps(FREE_ENABLED))
        assert asyncio.run(providers.warm_enabled_model_map(session)) is False
    finally:
        _clear(session, "minds_api_key", "minds_url", "minds_model_enabled")
        session.close()


# ── the payoff: warm -> DB -> free-tier default resolves affordable ───

def test_warm_flips_the_free_tier_default_off_the_paid_canonical(monkeypatch):
    from cowork.services import providers

    async def fake_fetch(url, key, *, force_refresh=False, tenant_key=None):
        return _listing(FREE_ENABLED)

    monkeypatch.setattr(providers, "fetch_minds_models", fake_fetch)
    session = _fresh_session()
    try:
        _set(session, minds_api_key="mdb_test", minds_url="https://minds.example/v1",
             planning_provider="minds_cloud", coding_provider="minds_cloud")

        # Cold map (the first-turn state): planning default is the paid canonical
        # that 402s a free-tier wallet.
        assert SettingService(session).load().resolved_planning_model == "sonnet"

        asyncio.run(providers.warm_enabled_model_map(session))

        # After the warm the same default resolves to the free-allowance model.
        assert SettingService(session).load().resolved_planning_model == "mindshub_air"
    finally:
        _clear(session, "minds_api_key", "minds_url", "minds_model_enabled",
               "planning_provider", "coding_provider")
        session.close()


# ── persist_enabled_model_map guards ──────────────────────────────────

def test_persist_skips_empty_map():
    from cowork.services.providers import persist_enabled_model_map

    session = _fresh_session()
    try:
        assert persist_enabled_model_map(session, LOCAL_SCOPE, "{}", {}) is False
    finally:
        session.close()


def test_persist_writes_only_on_change():
    from cowork.services.providers import persist_enabled_model_map

    session = _fresh_session()
    try:
        prior = json.dumps(FREE_ENABLED)
        assert persist_enabled_model_map(session, LOCAL_SCOPE, prior, FREE_ENABLED) is False
        assert persist_enabled_model_map(session, LOCAL_SCOPE, prior, PAID_ENABLED) is True
        assert json.loads(SettingService(session).load().minds_model_enabled) == PAID_ENABLED
    finally:
        _clear(session, "minds_model_enabled")
        session.close()


def test_persist_preserves_map_order():
    from cowork.services.providers import persist_enabled_model_map

    session = _fresh_session()
    try:
        # Deliberately NOT alphabetical, so a `sort_keys=True` regression would
        # reorder to [fable, mindshub_air, sonnet] and fail this — mirrors prod,
        # where /v1/models ranks a paid alias first and the first *enabled* model
        # is what the fallback must land on.
        ordered = {"sonnet": False, "fable": True, "mindshub_air": True}
        persist_enabled_model_map(session, LOCAL_SCOPE, "{}", ordered)
        stored = SettingService(session).load().minds_model_enabled
        assert list(json.loads(stored)) == ["sonnet", "fable", "mindshub_air"]
    finally:
        _clear(session, "minds_model_enabled")
        session.close()


# ── credential-sync seam: POST /settings/raw warms after the sync ─────

def test_write_raw_settings_warms_after_sync(monkeypatch, tmp_path):
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.db.session import get_open_session

    warmed = []

    async def spy_warm(session, scope=None):
        warmed.append(True)
        return False

    monkeypatch.setattr(settings_endpoint, "warm_enabled_model_map", spy_warm)
    monkeypatch.setattr(settings_endpoint, "_ENV_PATH", tmp_path / ".env")
    us.get_app_settings.cache_clear()  # local tenancy

    class _Req:
        client = type("C", (), {"host": "127.0.0.1"})()
        headers: dict = {}

    body = settings_endpoint._RawSettingsBody(content="ANTON_MINDS_API_KEY=mdb_test\n")
    session = get_open_session()
    try:
        result = asyncio.run(settings_endpoint.write_raw_settings(body, session, _Req()))
        assert result == {"ok": True}
        assert warmed == [True]  # the sync path warmed the map exactly once
    finally:
        session.close()


# ── boot seam: server startup warms the map (the load-bearing desktop seam) ──
#
# On desktop the /settings/raw seam above never fires (Electron writes .env and
# syncs via per-key PUTs), so the boot warm in server.py is the only seam that
# actually closes the ENG-748 first-turn gap. These lock it: deleting the warm
# block or inverting its `tenancy_mode` gate must turn one of them red.

def test_boot_warm_populates_map_on_desktop(monkeypatch):
    from cowork import server
    from cowork.services import providers

    async def fake_fetch(url, key, *, force_refresh=False, tenant_key=None):
        return _listing(FREE_ENABLED)

    monkeypatch.setattr(providers, "fetch_minds_models", fake_fetch)
    session = _fresh_session()  # local tenancy
    try:
        _set(session, minds_api_key="mdb_test", minds_url="https://minds.example/v1")
        assert asyncio.run(server._warm_model_map_on_boot()) is True
        assert json.loads(SettingService(session).load().minds_model_enabled) == FREE_ENABLED
    finally:
        _clear(session, "minds_api_key", "minds_url", "minds_model_enabled")
        session.close()


def test_boot_warm_is_gated_off_in_org_mode(monkeypatch):
    """Org mode stores no key and is floored by role_defaults; the boot warm
    must not fetch. A stored key is present so that inverting the gate would
    reach the (spied) fetch and fail this test."""
    from cowork import server
    from cowork.services import providers

    called = False

    async def fake_fetch(url, key, *, force_refresh=False, tenant_key=None):
        nonlocal called
        called = True
        return _listing(FREE_ENABLED)

    monkeypatch.setattr(providers, "fetch_minds_models", fake_fetch)
    session = _fresh_session()
    try:
        _set(session, minds_api_key="mdb_test", minds_url="https://minds.example/v1")
        monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
        us.get_app_settings.cache_clear()
        assert asyncio.run(server._warm_model_map_on_boot()) is False
        assert called is False  # the org gate short-circuited before the fetch
    finally:
        _clear(session, "minds_api_key", "minds_url", "minds_model_enabled")
        session.close()
        us.get_app_settings.cache_clear()


def test_boot_warm_is_bounded_and_fail_open(monkeypatch):
    """The warm runs before uvicorn binds the port, so a degraded MindsHub must
    not stall it. A fetch that hangs past the ceiling returns without raising
    and leaves the stored map untouched. The outer wait_for proves the bound:
    without it, an unbounded warm would make this test hang, not fail fast."""
    from cowork import server
    from cowork.services import providers

    async def hanging_fetch(url, key, *, force_refresh=False, tenant_key=None):
        await asyncio.sleep(30)
        return _listing(FREE_ENABLED)

    monkeypatch.setattr(providers, "fetch_minds_models", hanging_fetch)
    monkeypatch.setattr(server, "_BOOT_WARM_TIMEOUT_S", 0.1)
    session = _fresh_session()
    try:
        _set(session, minds_api_key="mdb_test", minds_url="https://minds.example/v1")

        async def _run():
            return await asyncio.wait_for(server._warm_model_map_on_boot(), timeout=3.0)

        assert asyncio.run(_run()) is False
        # A hung fetch never writes the map.
        assert SettingService(session).load().minds_model_enabled in (None, "{}")
    finally:
        _clear(session, "minds_api_key", "minds_url", "minds_model_enabled")
        session.close()
