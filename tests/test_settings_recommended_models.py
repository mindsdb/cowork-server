"""Tests for the /recommended-models overlay of custom OpenAI-compatible models.

The endpoint reads the openai-compatible provider card's own baseUrl from
providers_json (not the shared openai_base_url, which gemini/openai reuse) and
overlays its live model list. fetch_minds_models is stubbed so no network is hit.
"""
import asyncio
import json

from cowork.db.scoped import LOCAL_SCOPE


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

    async def fake_fetch(base_url, api_key, force_refresh=False):
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

        result = asyncio.run(recommended_models(session, LOCAL_SCOPE))

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

    async def fake_fetch(base_url, api_key, force_refresh=False):
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

        result = asyncio.run(recommended_models(session, LOCAL_SCOPE))

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


def test_recommended_models_no_openai_compatible_card(monkeypatch):
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session

    called = False

    async def fake_fetch(base_url, api_key, force_refresh=False):
        nonlocal called
        called = True
        return _listing(["x"])

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _delete_settings(session, "minds_api_key", "providers_json", "openai_api_key")

        result = asyncio.run(recommended_models(session, LOCAL_SCOPE))

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

    async def fake_fetch(base_url, api_key, force_refresh=False):
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

        result = asyncio.run(recommended_models(session, LOCAL_SCOPE))

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

    async def fake_fetch(base_url, api_key, force_refresh=False):
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

        result = asyncio.run(recommended_models(session, LOCAL_SCOPE))

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

    async def fake_fetch(base_url, api_key, force_refresh=False):
        return _listing(["mindshub_air"], enabled={"mindshub_air": True})

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _set_settings(session, minds_api_key="mdb_free", minds_url="https://api.mindshub.ai")
        _delete_settings(session, "providers_json")

        result = asyncio.run(recommended_models(session, LOCAL_SCOPE))

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

    async def fake_fetch(base_url, api_key, force_refresh=False):
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

        result = asyncio.run(recommended_models(session, LOCAL_SCOPE))

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

    async def fake_fetch(base_url, api_key, force_refresh=False):
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

        asyncio.run(recommended_models(session, LOCAL_SCOPE))

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

    async def fake_fetch(base_url, api_key, force_refresh=False):
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

        asyncio.run(recommended_models(session, LOCAL_SCOPE))  # map absent → 1 write
        asyncio.run(recommended_models(session, LOCAL_SCOPE))  # identical map stored → no write

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

    async def fake_fetch(base_url, api_key, force_refresh=False):
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

        asyncio.run(recommended_models(session, LOCAL_SCOPE))

        stored = SettingService(session).get_setting("minds_model_enabled").value
        assert list(json.loads(stored).keys()) == ["zephyr_base", "air-mini", "sonnet"]
    finally:
        _delete_settings(session, "minds_api_key", "minds_url", "minds_model_enabled")
        session.close()
