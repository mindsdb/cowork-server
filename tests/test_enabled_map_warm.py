"""Turn-path cold-start warm of the org availability map (ENG-748).

``minds_model_enabled`` is written only when the client calls
`/settings/recommended-models`, which on web boot is fire-and-forget and is NOT
serialized before the first `POST /responses`. A brand-new org that sends its
first message before that fetch lands resolves the *planning* model against an
empty map — ``_enabled_aware_default`` then hands out the canonical (paid)
default and the turn is denied on an empty wallet (the 33 first-contact
wallet-402s Langfuse saw). ``warm_org_enabled_model_map`` closes that race at the
one point guaranteed to run before resolution.

The persist guards themselves (never clobber a good map with {}, order-preserving
JSON, change-only write) are shared with the recommended-models endpoint via
``persist_enabled_model_map`` and covered there; these tests pin the warm's own
contract: org-only, no-op once warm, fail-open, and that a warmed map actually
flips the planning default off the paid canonical.
"""
import asyncio
import json

from cowork.common.settings.user_settings import Provider, get_user_settings
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.db.session import get_open_session
from cowork.services import providers as pv
from cowork.services.settings import SettingService

# Free-tier catalog shape: whole catalog listed, baseline first and enabled,
# paid models locked — exactly what a brand-new org's /v1/models returns.
FREE_ENABLED = {"mindshub_air": True, "sonnet": False, "opus": False}


def _listing(enabled: dict) -> "pv.MindsModelListing":
    return pv.MindsModelListing(list(enabled) or None, {}, enabled, {}, {}, {})


def _org(org_id: str = "org-748") -> TenantScope:
    return TenantScope(org_mode=True, org_id=org_id, user_id="u-1")


def _clear(session, scope, *keys: str) -> None:
    svc = SettingService(session, scope)
    for key in keys:
        try:
            svc.delete_setting(key)
        except ValueError:
            pass


def _stored(session, scope) -> str | None:
    return SettingService(session, scope).load().minds_model_enabled


def test_warm_populates_an_empty_org_map(monkeypatch):
    async def fake_catalog(*, org_id, bearer_token, refresh=False):
        assert (org_id, bearer_token) == ("org-748", "jwt-abc")  # per-org + forwarded bearer
        return _listing(FREE_ENABLED)

    monkeypatch.setattr(pv, "fetch_org_model_catalog", fake_catalog)
    session, scope = get_open_session(), _org()
    try:
        _clear(session, scope, "minds_model_enabled")
        wrote = asyncio.run(
            pv.warm_org_enabled_model_map(session, scope, bearer_token="jwt-abc")
        )
        assert wrote is True
        assert json.loads(_stored(session, scope)) == FREE_ENABLED
    finally:
        _clear(session, scope, "minds_model_enabled")
        session.close()


def test_warm_is_a_noop_once_the_map_is_populated(monkeypatch):
    called = False

    async def fake_catalog(*, org_id, bearer_token, refresh=False):
        nonlocal called
        called = True
        return _listing({"other": True})

    monkeypatch.setattr(pv, "fetch_org_model_catalog", fake_catalog)
    session, scope = get_open_session(), _org()
    try:
        SettingService(session, scope).upsert_setting(
            "minds_model_enabled", json.dumps(FREE_ENABLED)
        )
        wrote = asyncio.run(
            pv.warm_org_enabled_model_map(session, scope, bearer_token="jwt-abc")
        )
        assert wrote is False
        assert called is False  # a warm org never pays the catalog fetch
        assert json.loads(_stored(session, scope)) == FREE_ENABLED  # untouched
    finally:
        _clear(session, scope, "minds_model_enabled")
        session.close()


def test_warm_fails_open_on_a_catalog_error(monkeypatch):
    async def boom(*, org_id, bearer_token, refresh=False):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(pv, "fetch_org_model_catalog", boom)
    session, scope = get_open_session(), _org()
    try:
        _clear(session, scope, "minds_model_enabled")
        wrote = asyncio.run(
            pv.warm_org_enabled_model_map(session, scope, bearer_token="jwt-abc")
        )
        assert wrote is False  # no crash — the turn proceeds as it does today
        assert json.loads(_stored(session, scope) or "{}") == {}
    finally:
        _clear(session, scope, "minds_model_enabled")
        session.close()


def test_warm_never_clobbers_a_known_good_map_on_a_flagless_catalog(monkeypatch):
    """A gateway that returns ids without `enabled` flags (version skew) must not
    wipe a previously-good map. The map is non-empty so the warm short-circuits
    before fetching at all — the strongest form of "preserve known-good"."""
    called = False

    async def fake_catalog(*, org_id, bearer_token, refresh=False):
        nonlocal called
        called = True
        return _listing({})  # ids-only skew would yield an empty enabled map

    monkeypatch.setattr(pv, "fetch_org_model_catalog", fake_catalog)
    session, scope = get_open_session(), _org()
    try:
        good = json.dumps(FREE_ENABLED)
        SettingService(session, scope).upsert_setting("minds_model_enabled", good)
        asyncio.run(pv.warm_org_enabled_model_map(session, scope, bearer_token="jwt-abc"))
        assert called is False
        assert json.loads(_stored(session, scope)) == FREE_ENABLED
    finally:
        _clear(session, scope, "minds_model_enabled")
        session.close()


def test_warm_skips_local_mode(monkeypatch):
    called = False

    async def fake_catalog(*, org_id, bearer_token, refresh=False):
        nonlocal called
        called = True
        return _listing(FREE_ENABLED)

    monkeypatch.setattr(pv, "fetch_org_model_catalog", fake_catalog)
    session = get_open_session()
    try:
        wrote = asyncio.run(
            pv.warm_org_enabled_model_map(session, LOCAL_SCOPE, bearer_token="jwt-abc")
        )
        assert wrote is False and called is False
    finally:
        session.close()


def test_warm_skips_without_a_bearer(monkeypatch):
    called = False

    async def fake_catalog(*, org_id, bearer_token, refresh=False):
        nonlocal called
        called = True
        return _listing(FREE_ENABLED)

    monkeypatch.setattr(pv, "fetch_org_model_catalog", fake_catalog)
    session, scope = get_open_session(), _org()
    try:
        _clear(session, scope, "minds_model_enabled")
        wrote = asyncio.run(
            pv.warm_org_enabled_model_map(session, scope, bearer_token=None)
        )
        assert wrote is False and called is False
    finally:
        _clear(session, scope, "minds_model_enabled")
        session.close()


def test_warm_flips_the_planning_default_off_the_paid_canonical(monkeypatch):
    """The payoff: before the warm, an empty map resolves planning to the paid
    canonical default (the 402 shape); after, to the free baseline model."""
    async def fake_catalog(*, org_id, bearer_token, refresh=False):
        return _listing(FREE_ENABLED)  # sonnet locked, mindshub_air first + enabled

    monkeypatch.setattr(pv, "fetch_org_model_catalog", fake_catalog)
    session, scope = get_open_session(), _org()
    try:
        # Brand-new org: minds-cloud provider, no explicit planning pick, empty map.
        svc = SettingService(session, scope)
        _clear(session, scope, "minds_model_enabled", "planning_model")
        svc.upsert_setting("planning_provider", Provider.MINDS_CLOUD.value)
        svc.upsert_setting("minds_api_key", "mdb_free")

        # Empty map → canonical (paid) default: what the first turn 402s on today.
        assert get_user_settings(scope).planning_model == "sonnet"

        asyncio.run(pv.warm_org_enabled_model_map(session, scope, bearer_token="jwt-abc"))

        # Warmed map → the first enabled (free) model, so the first turn is served.
        assert get_user_settings(scope).planning_model == "mindshub_air"
    finally:
        _clear(session, scope, "minds_model_enabled", "planning_provider", "minds_api_key")
        session.close()
