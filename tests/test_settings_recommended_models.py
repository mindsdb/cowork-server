"""Tests for the /recommended-models overlay of custom OpenAI-compatible models.

The endpoint reads the openai-compatible provider card's own baseUrl from
providers_json (not the shared openai_base_url, which gemini/openai reuse) and
overlays its live model list. fetch_minds_models is stubbed so no network is hit.
"""
import asyncio
import json

from cowork.db.scoped import LOCAL_SCOPE
from tests._fakes import FakeRequest


def _listing(ids, efforts=None, enabled=None, labels=None, providers=None, families=None):
    """A MindsModelListing with everything a test doesn't care about left empty.

    Keeps a stub to the fields under test while still returning the real named
    tuple, so a field added to the listing shows up as a failing assertion here
    rather than as an attribute error inside the endpoint.
    """
    from cowork.services.providers import MindsModelListing

    return MindsModelListing(
        ids, efforts or {}, enabled or {}, labels or {}, providers or {}, families or {}
    )


def _delete_settings(session, *keys: str) -> None:
    from cowork.services.settings import SettingService

    service = SettingService(session)
    for key in keys:
        try:
            service.delete_setting(key)
        except ValueError:
            pass


def _set_settings(session, **values: str) -> None:
    from cowork.services.settings import SettingService

    service = SettingService(session)
    for key, value in values.items():
        service.upsert_setting(key, value)


def test_recommended_models_overlays_openai_compatible(monkeypatch):
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session

    calls: list[tuple[str, str]] = []

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        calls.append((base_url, api_key))
        return _listing(
            ["model-a", "model-b"],
            {"model-a": {"efforts": ["low", "high"], "default": "low"}},
            {"model-b": False},
        )

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        # No minds key, so only the openai-compatible branch can fire.
        _delete_settings(session, "minds_api_key")
        _set_settings(
            session,
            providers_json=json.dumps(
                [{"type": "openai-compatible", "baseUrl": "https://llm.staging.example/v1", "apiKey": "***"}]
            ),
            openai_api_key="sk-test",
        )

        result = asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))

        assert result["recommendedModels"]["openai-compatible"] == ["model-a", "model-b"]
        assert result["modelEfforts"]["model-a"] == {"efforts": ["low", "high"], "default": "low"}
        # enabled:false surfaces so the picker can render the model as locked.
        assert result["modelEnabled"] == {"model-b": False}
        # minds-cloud bucket untouched (its static default), confirming only the
        # openai-compatible branch ran.
        assert result["recommendedModels"]["minds-cloud"] == []
        # Fetched against the card's baseUrl + the stored OpenAI key.
        assert calls == [("https://llm.staging.example/v1", "sk-test")]
    finally:
        _delete_settings(session, "minds_api_key", "providers_json", "openai_api_key")
        session.close()


def test_custom_endpoint_cannot_override_a_minds_model(monkeypatch):
    """A colliding model id must not let a BYO base URL restyle a MindsHub model.

    modelEfforts / modelEnabled / modelLabels are keyed by model id alone, with
    no provider dimension, so both branches write into one namespace. MindsHub
    is fetched first and owns any id it describes; the custom endpoint fills
    only the ids MindsHub didn't. Without that, pointing an openai-compatible
    card at an endpoint that happens to serve `sonnet` would rename MindsHub's
    sonnet in the picker and could render it locked.
    """
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        if "mindshub" in base_url:
            return _listing(
                ["sonnet", "haiku"],
                {"sonnet": {"efforts": ["low", "high"], "default": "high"}},
                {"sonnet": True, "haiku": True},
                {"sonnet": "Claude Sonnet 5", "haiku": "Claude Haiku 4.5"},
                providers={"sonnet": "anthropic", "haiku": "anthropic"},
                families={"sonnet": "sonnet", "haiku": "haiku"},
            )
        # Same alias, different everything — and locked.
        return _listing(
            ["sonnet", "local-llama"],
            {"sonnet": {"efforts": ["none"], "default": "none"}},
            {"sonnet": False, "local-llama": True},
            {"sonnet": "Some Other Sonnet", "local-llama": "Local Llama"},
            providers={"sonnet": "someone-else", "local-llama": "someone-else"},
            families={"sonnet": "not-sonnet", "local-llama": "local-llama"},
        )

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _set_settings(
            session,
            minds_api_key="mdb_test",
            minds_url="https://api.mindshub.ai",
            providers_json=json.dumps(
                [{"type": "openai-compatible", "baseUrl": "https://llm.local/v1", "apiKey": "***"}]
            ),
            openai_api_key="sk-test",
        )

        result = asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))

        # MindsHub's description of `sonnet` survives on every id-keyed map.
        assert result["modelLabels"]["sonnet"] == "Claude Sonnet 5"
        assert result["modelEnabled"]["sonnet"] is True
        assert result["modelEfforts"]["sonnet"] == {"efforts": ["low", "high"], "default": "high"}
        # Including the grouping metadata: a BYO base URL must not be able to move
        # a MindsHub model into another vendor's section, or relabel it as an old
        # version of something it isn't.
        assert result["modelProviders"]["sonnet"] == "anthropic"
        assert result["modelFamilies"]["sonnet"] == "sonnet"
        # The custom endpoint still contributes ids MindsHub never mentioned.
        assert result["modelLabels"]["local-llama"] == "Local Llama"
        assert result["modelEnabled"]["local-llama"] is True
        # Both buckets still list their own models; only the id-keyed maps merge.
        assert result["recommendedModels"]["minds-cloud"] == ["sonnet", "haiku"]
        assert result["recommendedModels"]["openai-compatible"] == ["sonnet", "local-llama"]
    finally:
        _delete_settings(
            session, "minds_api_key", "minds_url", "providers_json", "openai_api_key", "minds_model_enabled"
        )
        session.close()


def test_custom_endpoint_cannot_describe_a_model_minds_listed_undescribed(monkeypatch):
    """A gateway older than provider/family describes nothing, and that still wins.

    While this server runs ahead of the MindsHub release that publishes the two
    fields, both minds maps come back empty for models MindsHub does serve. Reading
    "the target has no entry for this id" as "MindsHub didn't claim this one" would
    then let a custom endpoint answering `{"id": "sonnet", "family": "haiku"}` file
    MindsHub's own sonnet under haiku as an older version of it. What is reserved is
    every id MindsHub listed, described or not.
    """
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        if "mindshub" in base_url:
            # The pre-grouping gateway: ids and enabled flags, no metadata.
            return _listing(["sonnet", "haiku"], enabled={"sonnet": True, "haiku": True})
        # A custom endpoint that happens to serve the same alias, and files it
        # under another one. No provider, so `families` is its only claim.
        return _listing(["sonnet", "local-llama"], families={"sonnet": "haiku"})

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _set_settings(
            session,
            minds_api_key="mdb_test",
            minds_url="https://api.mindshub.ai",
            providers_json=json.dumps(
                [{"type": "openai-compatible", "baseUrl": "https://llm.local/v1", "apiKey": "***"}]
            ),
            openai_api_key="sk-test",
        )

        result = asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))

        # No family for sonnet at all is the right answer here: the app lists it
        # without a "latest" tag rather than under a head it was never pinned to.
        assert "sonnet" not in result["modelFamilies"]
        assert result["modelFamilies"] == {}
        assert result["modelProviders"] == {}
    finally:
        _delete_settings(
            session, "minds_api_key", "minds_url", "providers_json", "openai_api_key",
            "minds_model_enabled",
        )
        session.close()


def test_custom_endpoint_cannot_supply_a_provider_minds_could_not_place(monkeypatch):
    """A minds row with a family but no usable provider is in one map, not both.

    fetch_minds_models records such a row in `families` only, so a check against
    `modelProviders` alone finds no entry and would hand the model's section to the
    custom endpoint: a frozen anthropic snapshot rendered under someone else's
    heading.
    """
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        if "mindshub" in base_url:
            return _listing(["sonnet-4-5"], families={"sonnet-4-5": "sonnet"})
        return _listing(
            ["sonnet-4-5"],
            providers={"sonnet-4-5": "someone-else"},
            families={"sonnet-4-5": "sonnet-4-5"},
        )

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _set_settings(
            session,
            minds_api_key="mdb_test",
            minds_url="https://api.mindshub.ai",
            providers_json=json.dumps(
                [{"type": "openai-compatible", "baseUrl": "https://llm.local/v1", "apiKey": "***"}]
            ),
            openai_api_key="sk-test",
        )

        result = asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))

        assert result["modelProviders"] == {}
        # And the pin MindsHub did publish survives, so the app still knows this is
        # a frozen version rather than a moving alias.
        assert result["modelFamilies"] == {"sonnet-4-5": "sonnet"}
    finally:
        _delete_settings(
            session, "minds_api_key", "minds_url", "providers_json", "openai_api_key",
            "minds_model_enabled",
        )
        session.close()


def test_grouping_maps_stay_partial_across_two_listings(monkeypatch):
    """MindsHub describes its models, the custom endpoint describes none of its own.

    The served maps are partial, not empty: `sonnet` is described and `local-llama`
    is not, in the same response. Filling every listed id in (because some id in the
    map has metadata) would have the app tag `local-llama` "latest" and group it
    under a vendor neither gateway named.
    """
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        if "mindshub" in base_url:
            return _listing(
                ["sonnet"],
                providers={"sonnet": "anthropic"},
                families={"sonnet": "sonnet"},
            )
        return _listing(["sonnet", "local-llama"])  # ids only: a plain BYO endpoint

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _set_settings(
            session,
            minds_api_key="mdb_test",
            minds_url="https://api.mindshub.ai",
            providers_json=json.dumps(
                [{"type": "openai-compatible", "baseUrl": "https://llm.local/v1", "apiKey": "***"}]
            ),
            openai_api_key="sk-test",
        )

        result = asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))

        assert result["modelProviders"] == {"sonnet": "anthropic"}
        assert result["modelFamilies"] == {"sonnet": "sonnet"}
        # Both buckets still list everything their own gateway serves.
        assert result["recommendedModels"]["minds-cloud"] == ["sonnet"]
        assert result["recommendedModels"]["openai-compatible"] == ["sonnet", "local-llama"]
    finally:
        _delete_settings(
            session, "minds_api_key", "minds_url", "providers_json", "openai_api_key",
            "minds_model_enabled",
        )
        session.close()


def test_recommended_models_no_openai_compatible_card(monkeypatch):
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session

    called = False

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        nonlocal called
        called = True
        return _listing(["x"])

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _delete_settings(session, "minds_api_key", "providers_json", "openai_api_key")

        result = asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))

        assert result["recommendedModels"]["openai-compatible"] == []
        assert result["modelEnabled"] == {}
        assert called is False
    finally:
        session.close()


def test_recommended_models_surfaces_minds_locked_upsells(monkeypatch):
    """A free user's minds-cloud bucket lists paid models flagged enabled:false."""
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        # MindsHub lists the whole picker catalog; paid models come back
        # enabled:false for a free caller so the UI can show them as locked.
        return _listing(
            ["mindshub_air", "opus", "gpt"],
            enabled={"mindshub_air": True, "opus": False, "gpt": False},
            labels={"opus": "Claude Opus"},
        )

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _set_settings(session, minds_api_key="mdb_free", minds_url="https://api.mindshub.ai")
        _delete_settings(session, "providers_json")

        result = asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))

        assert result["recommendedModels"]["minds-cloud"] == ["mindshub_air", "opus", "gpt"]
        assert result["modelEnabled"] == {"mindshub_air": True, "opus": False, "gpt": False}
        # Display-only label passthrough — a model missing here (mindshub_air,
        # gpt) is the client's job to derive a fallback, not this endpoint's.
        assert result["modelLabels"] == {"opus": "Claude Opus"}
    finally:
        _delete_settings(session, "minds_api_key", "minds_url")
        session.close()


def test_recommended_models_surfaces_the_grouping_metadata(monkeypatch):
    """The two maps the picker groups by and tags "latest" from."""
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        return _listing(
            ["mindshub_air", "sonnet", "sonnet-4-5"],
            providers={"mindshub_air": "openai", "sonnet": "anthropic", "sonnet-4-5": "anthropic"},
            families={"mindshub_air": "mindshub_air", "sonnet": "sonnet", "sonnet-4-5": "sonnet"},
        )

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _set_settings(session, minds_api_key="mdb_free", minds_url="https://api.mindshub.ai")
        _delete_settings(session, "providers_json")

        result = asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))

        assert result["modelProviders"]["sonnet"] == "anthropic"
        # A moving alias names itself; a pin names its head. That difference is the
        # entire "which of these is the latest" signal the app renders from.
        assert result["modelFamilies"]["sonnet"] == "sonnet"
        assert result["modelFamilies"]["sonnet-4-5"] == "sonnet"
    finally:
        _delete_settings(session, "minds_api_key", "minds_url", "minds_model_enabled")
        session.close()


def test_recommended_models_keeps_serving_the_pre_existing_keys(monkeypatch):
    """An app whose UI bundle predates the new maps must keep working verbatim.

    The renderer updates over the air independently of this server, so the keys it
    already reads are a contract: new metadata is added alongside them, never in
    place of them.
    """
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        return _listing(["mindshub_air"], enabled={"mindshub_air": True})

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _set_settings(session, minds_api_key="mdb_free", minds_url="https://api.mindshub.ai")
        _delete_settings(session, "providers_json")

        result = asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))

        assert {
            "recommendedModels", "recommendedPair", "modelEfforts", "modelEnabled", "modelLabels",
        } <= set(result)
        assert result["recommendedModels"]["minds-cloud"] == ["mindshub_air"]
    finally:
        _delete_settings(session, "minds_api_key", "minds_url", "minds_model_enabled")
        session.close()


def test_recommended_models_grouping_maps_empty_for_a_byok_endpoint(monkeypatch):
    """A BYOK gateway publishes neither field, and that is not an error.

    The maps come back empty, which is the app's signal to render one ungrouped
    list rather than to group everything under an invented vendor.
    """
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        return _listing(["model-a", "model-b"])

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _delete_settings(session, "minds_api_key")
        _set_settings(
            session,
            providers_json=json.dumps(
                [{"type": "openai-compatible", "baseUrl": "https://llm.example/v1", "apiKey": "***"}]
            ),
            openai_api_key="sk-test",
        )

        result = asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))

        assert result["recommendedModels"]["openai-compatible"] == ["model-a", "model-b"]
        assert result["modelProviders"] == {}
        assert result["modelFamilies"] == {}
    finally:
        _delete_settings(session, "minds_api_key", "providers_json", "openai_api_key")
        session.close()


def test_recommended_models_empty_enabled_does_not_wipe_map(monkeypatch):
    """A fetch returning ids but no enabled flags (gateway version skew) must
    NOT overwrite a previously-good availability map with {} — that would
    re-lock the canonical default, the exact ENG-597 bug. Guard is on
    live_enabled, not the id list."""
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session
    from cowork.services.settings import SettingService

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        return _listing(["mindshub_air", "opus"])  # ids present, enabled EMPTY

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        good = json.dumps({"mindshub_air": True, "opus": False})
        _set_settings(
            session,
            minds_api_key="mdb_free",
            minds_url="https://api.mindshub.ai",
            minds_model_enabled=good,
        )
        _delete_settings(session, "providers_json")

        asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))

        stored = SettingService(session).get_setting("minds_model_enabled").value
        assert json.loads(stored) == {"mindshub_air": True, "opus": False}
    finally:
        _delete_settings(session, "minds_api_key", "minds_url", "minds_model_enabled")
        session.close()


def test_recommended_models_writes_map_only_on_change(monkeypatch):
    """upsert_setting commits a row + invalidates the settings cache, and this
    endpoint runs on every boot/settings-open — so the map is written only when
    it actually changed, not unconditionally."""
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session
    from cowork.services.settings import SettingService

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        return _listing(["mindshub_air", "opus"], enabled={"mindshub_air": True, "opus": False})

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _set_settings(session, minds_api_key="mdb_free", minds_url="https://api.mindshub.ai")
        _delete_settings(session, "providers_json", "minds_model_enabled")

        # Spy AFTER seeding settings so only the endpoint's writes are counted.
        writes: list[str] = []
        real_upsert = SettingService.upsert_setting

        def spy_upsert(self, key, value):
            if key == "minds_model_enabled":
                writes.append(value)
            return real_upsert(self, key, value)

        monkeypatch.setattr(SettingService, "upsert_setting", spy_upsert)

        asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))  # map absent → 1 write
        asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))  # identical map stored → no write

        assert len(writes) == 1, writes
    finally:
        _delete_settings(session, "minds_api_key", "minds_url", "minds_model_enabled")
        session.close()


def test_recommended_models_write_preserves_map_order(monkeypatch):
    """The persisted map must keep /v1/models order (baseline model first) —
    the first-enabled default fallback iterates in insertion order. A sorted
    write would alphabetize it and could silently promote the wrong model
    (e.g. an enabled 'air-mini' sorting before the baseline)."""
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session
    from cowork.services.settings import SettingService

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        # Baseline listed FIRST by the gateway, but sorting alphabetically
        # would put 'air-mini' ahead of it.
        return _listing(
            ["zephyr_base", "air-mini", "sonnet"],
            enabled={"zephyr_base": True, "air-mini": True, "sonnet": False},
        )

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _set_settings(session, minds_api_key="mdb_free", minds_url="https://api.mindshub.ai")
        _delete_settings(session, "providers_json", "minds_model_enabled")

        asyncio.run(recommended_models(FakeRequest(), session, LOCAL_SCOPE))

        stored = SettingService(session).get_setting("minds_model_enabled").value
        assert list(json.loads(stored).keys()) == ["zephyr_base", "air-mini", "sonnet"]
    finally:
        _delete_settings(session, "minds_api_key", "minds_url", "minds_model_enabled")
        session.close()


def test_org_mode_uses_operator_catalog_with_caller_bearer(monkeypatch):
    """Org mode: the catalog comes from fetch_org_model_catalog (operator URL +
    caller bearer, per-org), NOT the stored key or the tenant-settable s.minds_url."""
    from cowork.api.v1.endpoints import settings as ep
    from cowork.db.scoped import TenantScope
    from cowork.db.session import get_open_session

    # An admin-set minds_url must be ignored on this path (would otherwise be a
    # JWT-forwarding sink). Prove it by making the endpoint's own fetch explode.
    _set_settings(get_open_session(), minds_url="https://attacker.example/v1")

    seen = {}

    async def fake_org_catalog(*, org_id, bearer_token, refresh=False):
        seen["org_id"] = org_id
        seen["token"] = bearer_token
        return _listing(["mindshub_air", "sonnet"], enabled={"mindshub_air": True, "sonnet": False})

    async def boom_fetch(*a, **k):  # the direct fetch must NOT run in org mode
        raise AssertionError("org path must not call fetch_minds_models directly")

    monkeypatch.setattr(ep, "fetch_org_model_catalog", fake_org_catalog)
    monkeypatch.setattr(ep, "fetch_minds_models", boom_fetch)

    class Req:
        headers = {"Authorization": "Bearer jwt-abc"}

    org = TenantScope(org_mode=True, org_id="org-1", user_id="u-1")
    result = asyncio.run(ep.recommended_models(Req(), get_open_session(), org))

    assert seen == {"org_id": "org-1", "token": "jwt-abc"}   # per-org, forwarded bearer
    assert result["recommendedModels"]["minds-cloud"] == ["mindshub_air", "sonnet"]
    assert result["modelEnabled"]["sonnet"] is False          # wallet-lock surfaced


def test_providers_cache_is_scoped_by_tenant(monkeypatch):
    """Exercise the real _minds_models_cache: same URL, different tenant_key must
    not share an entry (the enabled map is wallet-specific)."""
    from cowork.services import providers as pv

    calls = []

    async def fake_get(url, headers=None, **kw):
        calls.append(headers.get("Authorization"))
        class R:
            status_code = 200
            def json(self):
                # tenant A locks sonnet, tenant B does not — keyed off the bearer
                locked = "A" in headers["Authorization"]
                return {"data": [
                    {"id": "mindshub_air", "enabled": True},
                    {"id": "sonnet", "enabled": not locked},
                ]}
        return R()

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        get = staticmethod(fake_get)

    monkeypatch.setattr(pv.httpx, "AsyncClient", lambda *a, **k: FakeClient())
    pv._minds_models_cache.clear()

    url = "https://api.staging.example/v1"
    a = asyncio.run(pv.fetch_minds_models(url, "tok-A", tenant_key="org-A"))
    b = asyncio.run(pv.fetch_minds_models(url, "tok-B", tenant_key="org-B"))
    a_again = asyncio.run(pv.fetch_minds_models(url, "tok-A", tenant_key="org-A"))

    assert a.enabled["sonnet"] is False and b.enabled["sonnet"] is True  # no bleed
    assert len(calls) == 2                       # A cached, B cached, A re-served from cache
    keys = {k[1] for k in pv._minds_models_cache}
    assert keys == {"org-A", "org-B"}            # one entry per tenant
