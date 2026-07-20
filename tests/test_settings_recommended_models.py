"""Tests for the /recommended-models overlay of custom OpenAI-compatible models.

The endpoint reads the openai-compatible provider card's own baseUrl from
providers_json (not the shared openai_base_url, which gemini/openai reuse) and
overlays its live model list. fetch_minds_models is stubbed so no network is hit.
"""
import asyncio
import json


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

    async def fake_fetch(base_url, api_key):
        calls.append((base_url, api_key))
        return (
            ["model-a", "model-b"],
            {"model-a": {"efforts": ["low", "high"], "default": "low"}},
            {"model-b": False},
            {},
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

        result = asyncio.run(recommended_models(session))

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


def test_recommended_models_no_openai_compatible_card(monkeypatch):
    from cowork.api.v1.endpoints import settings as settings_endpoint
    from cowork.api.v1.endpoints.settings import recommended_models
    from cowork.db.session import get_open_session

    called = False

    async def fake_fetch(base_url, api_key):
        nonlocal called
        called = True
        return ["x"], {}, {}, {}

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _delete_settings(session, "minds_api_key", "providers_json", "openai_api_key")

        result = asyncio.run(recommended_models(session))

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

    async def fake_fetch(base_url, api_key):
        # MindsHub lists the whole picker catalog; paid models come back
        # enabled:false for a free caller so the UI can show them as locked.
        return (
            ["mindshub_air", "opus", "gpt"],
            {},
            {"mindshub_air": True, "opus": False, "gpt": False},
            {},
        )

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _set_settings(session, minds_api_key="mdb_free", minds_url="https://api.mindshub.ai")
        _delete_settings(session, "providers_json")

        result = asyncio.run(recommended_models(session))

        assert result["recommendedModels"]["minds-cloud"] == ["mindshub_air", "opus", "gpt"]
        assert result["modelEnabled"] == {"mindshub_air": True, "opus": False, "gpt": False}
    finally:
        _delete_settings(session, "minds_api_key", "minds_url")
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

    async def fake_fetch(base_url, api_key):
        return (["mindshub_air", "opus"], {}, {})  # ids present, enabled EMPTY

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

        asyncio.run(recommended_models(session))

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

    async def fake_fetch(base_url, api_key):
        return (["mindshub_air", "opus"], {}, {"mindshub_air": True, "opus": False})

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

        asyncio.run(recommended_models(session))  # map absent → 1 write
        asyncio.run(recommended_models(session))  # identical map stored → no write

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

    async def fake_fetch(base_url, api_key):
        # Baseline listed FIRST by the gateway, but sorting alphabetically
        # would put 'air-mini' ahead of it.
        return (
            ["zephyr_base", "air-mini", "sonnet"],
            {},
            {"zephyr_base": True, "air-mini": True, "sonnet": False},
        )

    monkeypatch.setattr(settings_endpoint, "fetch_minds_models", fake_fetch)

    session = get_open_session()
    try:
        _set_settings(session, minds_api_key="mdb_free", minds_url="https://api.mindshub.ai")
        _delete_settings(session, "providers_json", "minds_model_enabled")

        asyncio.run(recommended_models(session))

        stored = SettingService(session).get_setting("minds_model_enabled").value
        assert list(json.loads(stored).keys()) == ["zephyr_base", "air-mini", "sonnet"]
    finally:
        _delete_settings(session, "minds_api_key", "minds_url", "minds_model_enabled")
        session.close()
