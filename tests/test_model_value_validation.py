"""Write-time validation of model settings (ENG-1358).

A model id MindsHub cannot serve could be written into `planning_model` /
`coding_model` and nothing caught it: the settings API validated the key and the
type but never the value, the connection test is deliberately model-blind, and
the only signal was a 404 per turn. These cover the write-time gate.

The bias under test is SOFT-FAIL: the gate may only reject when it holds a real
catalog that definitively lacks the id. Every other state — no MindsHub, no
credentials, an unreachable or empty catalog — must allow the write, or an
offline user can't change their own settings.
"""
import asyncio

import pytest

import cowork.services.providers as providers
from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.services.providers import MODEL_VALUE_SETTINGS, model_value_rejection

CATALOG = ["mindshub_air", "claude-sonnet-5", "deepseek-v4-pro"]


def _listing(ids):
    return providers.MindsModelListing(
        ids=ids, efforts={}, enabled={}, labels={}, providers={}, families={}, role_defaults={}
    )


def _minds_settings(**overrides):
    """Settings with MindsHub configured as the provider for every model role."""
    base = dict(
        minds_api_key="sk-test-not-a-real-key",
        minds_url="https://api.mindshub.ai",
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        router_provider=Provider.MINDS_CLOUD,
    )
    base.update(overrides)
    return UserSettings.model_validate(base)


@pytest.fixture
def catalog(monkeypatch):
    """Install a fake /v1/models catalog; returns a list of fetch call counts."""
    calls = []

    async def _fake(minds_url, api_key, *, force_refresh=False, tenant_key=None):
        calls.append((minds_url, api_key))
        return _listing(list(CATALOG))

    monkeypatch.setattr(providers, "fetch_minds_models", _fake)
    return calls


def _reject(settings, key, value):
    return asyncio.run(model_value_rejection(settings, key, value))


# ── The ENG-1358 case ────────────────────────────────────────────────


def test_the_eng_1358_id_is_rejected(catalog):
    """`deepseek-v4-flash` is a Fireworks provider name; the live catalog has
    `deepseek-v4-pro`. This is the exact value that reached the reported user's
    settings and produced a 404 on every turn."""
    reason = _reject(_minds_settings(), "planning_model", "deepseek-v4-flash")
    assert reason is not None
    assert "deepseek-v4-flash" in reason


def test_rejection_names_real_models_and_leaks_no_credentials(catalog):
    """The string is returned to the client as a 400 body."""
    reason = _reject(_minds_settings(), "planning_model", "not-a-model")
    assert "mindshub_air" in reason  # points at something usable
    assert "sk-test-not-a-real-key" not in reason
    assert "api.mindshub.ai" not in reason


@pytest.mark.parametrize("key", sorted(MODEL_VALUE_SETTINGS))
def test_every_model_valued_setting_is_gated(catalog, key):
    """planning/coding/router share the bare `str | None` shape and the failure."""
    assert _reject(_minds_settings(), key, "not-a-model") is not None


def test_a_catalog_model_is_allowed(catalog):
    for model in CATALOG:
        assert _reject(_minds_settings(), "planning_model", model) is None


# ── Soft-fail: everything below MUST allow the write ─────────────────


def test_catalog_unreachable_allows_the_write(monkeypatch):
    """ids is None on any fetch failure — an offline MindsHub must never stop
    someone changing settings."""
    async def _fail(*a, **k):
        return _listing(None)

    monkeypatch.setattr(providers, "fetch_minds_models", _fail)
    assert _reject(_minds_settings(), "planning_model", "anything-at-all") is None


def test_catalog_raising_allows_the_write(monkeypatch):
    async def _boom(*a, **k):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(providers, "fetch_minds_models", _boom)
    assert _reject(_minds_settings(), "planning_model", "anything-at-all") is None


def test_empty_catalog_allows_the_write(monkeypatch):
    async def _empty(*a, **k):
        return _listing([])

    monkeypatch.setattr(providers, "fetch_minds_models", _empty)
    assert _reject(_minds_settings(), "planning_model", "anything-at-all") is None


def test_no_credentials_allows_the_write(catalog):
    """Onboarding writes a model before credentials exist; blocking that would
    deadlock a fresh install."""
    s = _minds_settings(minds_api_key=None)
    assert _reject(s, "planning_model", "anything-at-all") is None
    assert not catalog, "must not attempt a fetch with no key"


def test_byok_provider_is_not_gated(catalog):
    """Anthropic publishes no /v1/models; a BYOK endpoint has its own catalog.
    Checking a MindsHub catalog would reject every valid Anthropic model."""
    s = _minds_settings(
        planning_provider=Provider.ANTHROPIC,
        anthropic_api_key="sk-ant-test-not-a-real-key",
    )
    assert _reject(s, "planning_model", "claude-opus-4-20250514") is None
    assert not catalog, "must not fetch the MindsHub catalog for a BYOK provider"


def test_gated_on_where_the_turn_ACTUALLY_goes_not_the_declared_provider(catalog):
    """`planning_provider` is a preference, not a destination: with no Anthropic
    key, `_resolve_provider` falls back to MindsHub and the turn is served there.
    Keying on the DECLARED provider would leave that user ungated — the exact
    hole this check exists to close — so it keys on the resolved one."""
    s = _minds_settings(planning_provider=Provider.ANTHROPIC, anthropic_api_key=None)
    assert s.resolved_planning_provider == Provider.MINDS_CLOUD
    assert _reject(s, "planning_model", "not-a-model") is not None


def test_clearing_a_model_is_allowed(catalog):
    """Empty means "fall back to the provider default" — always legal."""
    for value in ("", "   "):
        assert _reject(_minds_settings(), "planning_model", value) is None
    assert not catalog


def test_non_model_settings_are_untouched(catalog):
    assert _reject(_minds_settings(), "memory_mode", "not-a-model") is None
    assert not catalog, "must not fetch a catalog for a non-model setting"


# ── Endpoint level: the gate every real writer lands on ──────────────


def _seed_minds(svc):
    svc.save_all({
        "minds_api_key": "sk-test-not-a-real-key",
        "minds_url": "https://api.mindshub.ai",
        "planning_provider": "minds_cloud",
    })


async def test_put_settings_key_400s_on_an_unservable_model(monkeypatch):
    """`PUT /settings/planning_model` — the path the reported user's value took,
    and the one `syncModelsToDb` (desktop onboarding) writes through."""
    from fastapi import HTTPException
    from cowork.api.v1.endpoints.settings import upsert_setting
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.db.session import get_open_session
    from cowork.schemas.settings import SettingUpsertRequest
    from cowork.services.settings import SettingService

    async def _fake(*a, **k):
        return _listing(list(CATALOG))

    monkeypatch.setattr(providers, "fetch_minds_models", _fake)

    session = get_open_session()
    svc = SettingService(session)
    try:
        _seed_minds(svc)
        with pytest.raises(HTTPException) as exc:
            await upsert_setting(
                "planning_model",
                SettingUpsertRequest(value="deepseek-v4-flash"),
                session,
                LOCAL_SCOPE,
                None,
            )
        assert exc.value.status_code == 400
        assert "deepseek-v4-flash" in exc.value.detail
        # Nothing persisted — the bad id must not reach the DB at all.
        assert svc._fetch_row("planning_model") is None

        # A real catalog model saves normally through the same path.
        await upsert_setting(
            "planning_model",
            SettingUpsertRequest(value="mindshub_air"),
            session,
            LOCAL_SCOPE,
            None,
        )
        assert svc.load().planning_model == "mindshub_air"
    finally:
        for k in ("planning_model", "minds_api_key", "minds_url", "planning_provider"):
            try:
                svc.delete_setting(k)
            except ValueError:
                pass
        session.close()


async def test_an_unreachable_catalog_never_blocks_a_settings_save(monkeypatch):
    """The soft-fail contract, at the endpoint: MindsHub down must not make the
    Settings form unsaveable."""
    from cowork.api.v1.endpoints.settings import upsert_setting
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.db.session import get_open_session
    from cowork.schemas.settings import SettingUpsertRequest
    from cowork.services.settings import SettingService

    async def _down(*a, **k):
        return _listing(None)

    monkeypatch.setattr(providers, "fetch_minds_models", _down)

    session = get_open_session()
    svc = SettingService(session)
    try:
        _seed_minds(svc)
        await upsert_setting(
            "planning_model",
            SettingUpsertRequest(value="some-model-we-cannot-verify"),
            session,
            LOCAL_SCOPE,
            None,
        )
        assert svc.load().planning_model == "some-model-we-cannot-verify"
    finally:
        for k in ("planning_model", "minds_api_key", "minds_url", "planning_provider"):
            try:
                svc.delete_setting(k)
            except ValueError:
                pass
        session.close()


# ── Pending-state resolution (ENG-1358 review, blocking) ─────────────
#
# The Settings form ships provider + credential + model in ONE bulk PUT, and no
# UI path saves a provider without its role's model. A gate that resolves the
# provider from the PRE-write DB therefore asks about the config being replaced:
# it checked the new model against the OLD provider's catalog.


async def test_switching_minds_to_byok_in_one_put_is_not_blocked(monkeypatch):
    """The escape route this ticket exists to provide. Pre-write the provider is
    still minds_cloud, so a pre-write gate validated Anthropic's own recommended
    model against the MindsHub catalog and 400'd — taking the API key down with
    it, since save_all is all-or-nothing."""
    from cowork.api.v1.endpoints.settings import bulk_upsert_settings
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.db.session import get_open_session
    from cowork.schemas.settings import SettingsBulkUpsertRequest
    from cowork.services.settings import SettingService

    async def _fake(*a, **k):
        return _listing(list(CATALOG))

    monkeypatch.setattr(providers, "fetch_minds_models", _fake)

    session = get_open_session()
    svc = SettingService(session, LOCAL_SCOPE)
    keys = ("planning_model", "planning_provider", "minds_api_key",
            "minds_url", "anthropic_api_key")
    try:
        svc.save_all({
            "minds_api_key": "sk-minds", "minds_url": "https://api.mindshub.ai",
            "planning_provider": "minds_cloud", "planning_model": "mindshub_air",
        })
        await bulk_upsert_settings(
            SettingsBulkUpsertRequest(values={
                "planning_provider": "anthropic",
                "anthropic_api_key": "sk-ant-xyz",
                "planning_model": "claude-sonnet-4-6",
            }), session, LOCAL_SCOPE, None,
        )
        s = svc.load()
        assert s.planning_model == "claude-sonnet-4-6"
        assert s.resolved_planning_provider == Provider.ANTHROPIC
        assert s.anthropic_api_key is not None, "the credential must not roll back"
    finally:
        for k in keys:
            try:
                svc.delete_setting(k)
            except ValueError:
                pass
        session.close()


async def test_switching_byok_to_minds_in_one_put_is_gated(monkeypatch):
    """The inverse hole: pre-write the provider is anthropic, so a pre-write gate
    no-op'd and persisted this ticket's own id through the new check."""
    from fastapi import HTTPException
    from cowork.api.v1.endpoints.settings import bulk_upsert_settings
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.db.session import get_open_session
    from cowork.schemas.settings import SettingsBulkUpsertRequest
    from cowork.services.settings import SettingService

    fetched = []

    async def _fake(minds_url, api_key, *, force_refresh=False, tenant_key=None):
        fetched.append(minds_url)
        return _listing(list(CATALOG))

    monkeypatch.setattr(providers, "fetch_minds_models", _fake)

    session = get_open_session()
    svc = SettingService(session, LOCAL_SCOPE)
    keys = ("planning_model", "planning_provider", "minds_api_key",
            "minds_url", "anthropic_api_key")
    try:
        svc.save_all({
            "anthropic_api_key": "sk-ant-xyz", "planning_provider": "anthropic",
            "minds_api_key": "sk-minds", "minds_url": "https://api.mindshub.ai",
            "planning_model": "claude-sonnet-4-6",
        })
        with pytest.raises(HTTPException) as exc:
            await bulk_upsert_settings(
                SettingsBulkUpsertRequest(values={
                    "planning_provider": "minds_cloud",
                    "planning_model": "deepseek-v4-flash",
                }), session, LOCAL_SCOPE, None,
            )
        assert exc.value.status_code == 400
        assert fetched, "the gate must consult the catalog for the PENDING provider"
        assert svc.load().planning_model == "claude-sonnet-4-6"
    finally:
        for k in keys:
            try:
                svc.delete_setting(k)
            except ValueError:
                pass
        session.close()


def test_first_run_bulk_save_carrying_its_own_credential_is_gated(catalog):
    """Onboarding sends key + url + model together. Resolving pre-write found no
    credential and soft-passed, making the gate a no-op on a fresh install."""
    from cowork.services.settings import SettingService
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.db.session import get_open_session

    session = get_open_session()
    svc = SettingService(session, LOCAL_SCOPE)
    try:
        pending = svc.load_pending({
            "minds_api_key": "sk-minds",
            "minds_url": "https://api.mindshub.ai",
            "planning_provider": "minds_cloud",
            "planning_model": "deepseek-v4-flash",
        })
        assert pending.resolved_planning_provider == Provider.MINDS_CLOUD
        assert pending.minds_api_key is not None
        assert _reject(pending, "planning_model", "deepseek-v4-flash") is not None
    finally:
        session.close()


def test_load_pending_skips_write_diff_sentinels(catalog):
    """None / *** mean "unchanged" to the writers; the gate must agree or a
    masked-secret save would read as clearing the credential."""
    from cowork.services.settings import SettingService
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.db.session import get_open_session

    session = get_open_session()
    svc = SettingService(session, LOCAL_SCOPE)
    try:
        svc.save_all({"minds_api_key": "sk-minds", "minds_url": "https://api.mindshub.ai"})
        pending = svc.load_pending({"minds_api_key": "***", "minds_url": None})
        assert pending.minds_api_key is not None
        assert pending.minds_url == "https://api.mindshub.ai"
    finally:
        for k in ("minds_api_key", "minds_url"):
            try:
                svc.delete_setting(k)
            except ValueError:
                pass
        session.close()


# ── The deprecated-but-live `latest:` namespace ──────────────────────


def test_latest_prefixed_aliases_prod_still_resolves_are_allowed(catalog):
    """/v1/models is a LISTING, not the servable set: the gateway still resolves
    a `latest:` prefix (app_settings.py:23), minds-auth.ts preserves such a pin
    as a user choice, and every shipped cowork_evals run spec carries one."""
    for model in CATALOG:
        assert _reject(_minds_settings(), "planning_model", f"latest:{model}") is None


def test_latest_prefix_does_not_become_a_bypass(catalog):
    """A bare "starts with latest:" allowance would let `latest:nonsense` through,
    which prod 404s identically to deepseek-v4-flash — reopening this bug inside
    the legacy namespace."""
    assert _reject(_minds_settings(), "planning_model", "latest:nonsense") is not None
    assert _reject(_minds_settings(), "planning_model", "latest:deepseek-v4-flash") is not None


# ── Org (hosted) tenancy ─────────────────────────────────────────────
#
# Hosted stores no MindsHub key — a per-turn key is minted, so `_has_key`
# returns True for minds_cloud with nothing stored. The gate's
# "no credential → allow" soft-fail therefore made it a permanent no-op in the
# shipping tenancy mode, silently. The catalog comes from the operator endpoint
# keyed by org id + the caller's own bearer, exactly as recommended_models does.


def _org_settings():
    return UserSettings.model_validate({
        "planning_provider": Provider.MINDS_CLOUD,
        "coding_provider": Provider.MINDS_CLOUD,
        "router_provider": Provider.MINDS_CLOUD,
    })


def test_org_mode_gates_via_the_operator_catalog(monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    seen = {}

    async def _fake_org(*, org_id, bearer_token, refresh=False):
        seen["org_id"], seen["bearer"] = org_id, bearer_token
        return _listing(list(CATALOG))

    monkeypatch.setattr(providers, "fetch_org_model_catalog", _fake_org)
    try:
        s = _org_settings()
        assert s.minds_api_key is None, "hosted stores no key — the old soft-pass"
        reason = asyncio.run(providers.model_value_rejection(
            s, "planning_model", "deepseek-v4-flash",
            org_id="org-123", bearer_token="jwt-abc",
        ))
        assert reason is not None
        assert seen == {"org_id": "org-123", "bearer": "jwt-abc"}
        assert asyncio.run(providers.model_value_rejection(
            s, "planning_model", "mindshub_air",
            org_id="org-123", bearer_token="jwt-abc",
        )) is None
    finally:
        get_app_settings.cache_clear()


def test_org_mode_without_a_bearer_still_soft_fails(monkeypatch):
    """No bearer → no catalog → no evidence. Never block the write."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()

    async def _boom(**k):
        raise AssertionError("must not fetch without a bearer")

    monkeypatch.setattr(providers, "fetch_org_model_catalog", _boom)
    try:
        assert asyncio.run(providers.model_value_rejection(
            _org_settings(), "planning_model", "deepseek-v4-flash",
            org_id="org-123", bearer_token="",
        )) is None
    finally:
        get_app_settings.cache_clear()


# ── Org mode through the real HTTP stack (ENG-1358 re-review) ────────
#
# The tests above call model_value_rejection directly with org_id/bearer as
# literal arguments, which leaves the ENDPOINT wiring unpinned: with a direct
# call `request is None`, so the org branch is dead and both
# `_bearer_token(request)` and `scope.org_id` could be gutted with the suite
# still green. Hosted is the shipping mode and its gate rests on that wiring,
# so drive it over HTTP once.


def test_org_mode_gate_is_wired_through_the_endpoint(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cowork.api.v1.endpoints import settings as settings_ep
    from cowork.db.scoped import TenantScope, get_tenant_scope
    from cowork.db.session import get_open_session, get_session
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()

    seen = {}

    async def _fake_org(*, org_id, bearer_token, refresh=False):
        seen["org_id"], seen["bearer"] = org_id, bearer_token
        return _listing(list(CATALOG))

    async def _no_local(*a, **k):
        raise AssertionError("org mode must not use the stored-key catalog")

    monkeypatch.setattr(providers, "fetch_org_model_catalog", _fake_org)
    monkeypatch.setattr(providers, "fetch_minds_models", _no_local)

    app = FastAPI()
    app.include_router(settings_ep.router, prefix="/api/v1/settings")
    session = get_open_session()
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_tenant_scope] = lambda: TenantScope(
        org_mode=True, org_id="org-123", user_id="user-1"
    )
    try:
        client = TestClient(app)
        res = client.put(
            "/api/v1/settings/planning_model",
            json={"value": "deepseek-v4-flash"},
            headers={"Authorization": "Bearer jwt-abc"},
        )
        assert res.status_code == 400, res.text
        assert "deepseek-v4-flash" in res.json()["detail"]
        # Both halves of the wiring: the scope's org id and the CALLER's bearer
        # must reach the operator catalog. Gutting either one restores the
        # silent no-op this fix removed.
        assert seen == {"org_id": "org-123", "bearer": "jwt-abc"}

        assert client.put(
            "/api/v1/settings/planning_model",
            json={"value": "mindshub_air"},
            headers={"Authorization": "Bearer jwt-abc"},
        ).status_code == 200
    finally:
        get_app_settings.cache_clear()
        session.close()
