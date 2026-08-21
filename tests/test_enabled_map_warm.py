"""Warm the MindsHub availability map before the first turn (ENG-748, desktop).

ENG-1652 made the minds-cloud role defaults the free model in both tenancy
modes, so the UNSET default no longer 402s a free-tier wallet on turn one — the
floor covers it (see `test_first_turn_default_floor`). What the availability map
still governs is a stored PAID pin: a user who explicitly picked a premium model
(e.g. `kimi`) is steered back off it only by the cached `minds_model_enabled`
map, via the wallet-aware fallback in `_resolved_model`. That map is refreshed
lazily on GET /recommended-models, so a brand-new sign-in that sends a message
before the picker ever loads resolves that pin against an EMPTY map — an empty
map is no evidence, so the pin is kept and MindsHub denies the empty free-tier
wallet (`wallet_empty` 402) on message one. That is the residual first-contact
cohort in ENG-748.

`warm_enabled_model_map` closes the race at the two guaranteed-pre-first-turn
seams — server startup with a stored key, and immediately after a credential
sync (POST /settings/raw) — by fetching /v1/models and persisting the map
through the same guarded writer the recommended-models endpoint uses
(`persist_enabled_model_map`), so the invariants can't drift.

The unset default is floored by `role_defaults` in both modes (see
`test_first_turn_default_floor`); this file covers the desktop warm that heals a
stored pin before the picker loads.
"""
import asyncio
import json

import httpx

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


# ── the payoff: warm -> DB -> a free-tier user's paid pin resolves affordable ─

def test_warm_flips_a_stored_paid_pin_off_the_paid_canonical(monkeypatch):
    from cowork.services import providers

    async def fake_fetch(url, key, *, force_refresh=False, tenant_key=None):
        return _listing(FREE_ENABLED)

    monkeypatch.setattr(providers, "fetch_minds_models", fake_fetch)
    session = _fresh_session()
    try:
        # ENG-1652 made the UNSET default the free model, so the warm no longer
        # rescues the default — the floor does (see test_first_turn_default_floor).
        # What still 402s on turn one is a stored PAID pin: this user explicitly
        # picked `kimi`.
        _set(session, minds_api_key="mdb_test", minds_url="https://minds.example/v1",
             planning_provider="minds_cloud", coding_provider="minds_cloud",
             planning_model="kimi")

        # Cold map (the first-turn state): an empty map is no evidence, so the
        # pin is kept and a free-tier wallet 402s on `kimi`.
        assert SettingService(session).load().resolved_planning_model == "kimi"

        asyncio.run(providers.warm_enabled_model_map(session))

        # After the warm the map marks `kimi` disabled, so the wallet-aware
        # fallback resolves to the free-allowance model. The stored pin itself is
        # never rewritten — a topped-up wallet restores it on the next load.
        assert SettingService(session).load().resolved_planning_model == "mindshub_air"
        assert SettingService(session).load().planning_model == "kimi"
    finally:
        _clear(session, "minds_api_key", "minds_url", "minds_model_enabled",
               "planning_provider", "coding_provider", "planning_model")
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


def test_persist_densifies_sparse_enabled_over_full_catalogue():
    # `fetch_minds_models` only records rows that publish `enabled`, so a served
    # model that omits the flag (missing = available) is ABSENT from the raw
    # map. The resolution logic reads absence from a non-empty map as "retired"
    # and steers off it — so the sparse map must be densified over the full
    # served catalogue (`live_ids`) before it is stored, or a working free model
    # gets misread as retired. Here only `sonnet` publishes a flag; `mindshub_air`
    # and `haiku` are served but unflagged and must persist as available.
    from cowork.services.providers import persist_enabled_model_map

    session = _fresh_session()
    try:
        sparse = {"sonnet": False}
        catalogue = ["mindshub_air", "sonnet", "haiku"]
        assert persist_enabled_model_map(
            session, LOCAL_SCOPE, "{}", sparse, catalogue
        ) is True
        stored = json.loads(SettingService(session).load().minds_model_enabled)
        # Full catalogue, in gateway order, with the unflagged rows defaulted to
        # available and the published lock preserved.
        assert stored == {"mindshub_air": True, "sonnet": False, "haiku": True}
        assert list(stored) == catalogue
    finally:
        _clear(session, "minds_model_enabled")
        session.close()


def test_persist_prunes_retired_ids_when_enabled_flags_absent():
    # A gateway can return a real catalogue WITHOUT any `enabled` flags (version
    # skew / a plain OpenAI-compatible endpoint) -> live_enabled == {}. We can't
    # re-derive the locks, but the id list is still authoritative membership, so
    # a retired id must be PRUNED and the surviving locks preserved. Without the
    # prune the retired id lingers as "still served" and resolution keeps
    # selecting a model that 404s (ENG-1820).
    from cowork.services.providers import persist_enabled_model_map

    session = _fresh_session()
    try:
        prior = json.dumps({"mindshub_air": True, "opus": False})
        catalogue = ["opus", "sonnet"]  # mindshub_air retired; sonnet new
        assert persist_enabled_model_map(
            session, LOCAL_SCOPE, prior, {}, catalogue
        ) is True
        stored = json.loads(SettingService(session).load().minds_model_enabled)
        # mindshub_air pruned, opus's stored lock preserved, sonnet defaulted on.
        assert stored == {"opus": False, "sonnet": True}
        assert list(stored) == catalogue
    finally:
        _clear(session, "minds_model_enabled")
        session.close()


def test_persist_holds_map_when_no_catalogue_and_no_flags():
    # A fetch FAILURE yields neither a catalogue nor flags (live_ids None,
    # live_enabled {}). With no evidence at all we must NOT prune — hold the
    # known-good map rather than wipe it.
    from cowork.services.providers import persist_enabled_model_map

    session = _fresh_session()
    try:
        prior = json.dumps({"mindshub_air": True, "opus": False})
        assert persist_enabled_model_map(session, LOCAL_SCOPE, prior, {}, None) is False
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


# ── fetch_minds_models total budget (the layer under the boot bound) ──
#
# The boot seam's own ceiling is covered by test_boot_warm_is_bounded_and_fail_open,
# but that stubs fetch_minds_models out entirely. This exercises the REAL fetch and
# pins its total budget — `asyncio.wait_for(_fetch(), _MINDS_MODELS_TIMEOUT_S)` — which
# is the only thing that bounds a *successful-but-trickled* response: httpx.Timeout is
# per-operation, so a body dribbled in under-timeout chunks (or a redirect chain) has no
# ceiling without the outer wait_for. Deleting that wrapper leaves the whole suite green
# otherwise. A slow transport reproduces the unbounded case: the fetch would "succeed"
# after the sleep (so the negative cache never engages), and only the wait_for stops it —
# hence the hang→fail outer guard, matching the boot test above.

def test_fetch_minds_models_total_budget_bounds_a_slow_success(monkeypatch):
    from cowork.services import providers

    async def slow_get(self, url, *args, **kwargs):
        # A successful 200 that arrives only after the sleep. httpx's per-op
        # timeout is bypassed here (we replace .get), so nothing but the outer
        # wait_for can cut it off — exactly the trickle case with no per-op cap.
        await asyncio.sleep(30)
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "mindshub_air", "enabled": True}]},
            request=httpx.Request("GET", str(url)),
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", slow_get)
    monkeypatch.setattr(providers, "_MINDS_MODELS_TIMEOUT_S", 0.1)
    providers._minds_models_cache.clear()
    try:
        async def _run():
            # Guard well above the 0.1s budget but far below the 30s sleep: with
            # the budget in place this returns fast; remove it and the sleep runs
            # past the guard and the test times out (hang) rather than passing.
            return await asyncio.wait_for(
                providers.fetch_minds_models("https://minds.example/v1", "mdb_test"),
                timeout=3.0,
            )

        listing = asyncio.run(_run())
        # The budget expired mid-fetch: the failure falls through to an empty
        # listing (ids is None), never the slow gateway's real data.
        assert listing.ids is None
        # ...and that empty result is negatively cached so a degraded gateway
        # isn't re-probed on every boot/open.
        assert any(v.ids is None for _, v in providers._minds_models_cache.values())
    finally:
        providers._minds_models_cache.clear()


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
