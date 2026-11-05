"""The image model (ENG-4024): the setting, its default, the catalog, and how it
reaches anton, its scratchpads and the UI.

The image role is not a chat role. MindsHub is the only gateway that serves image
models, so there is no image provider; image rows travel on ``/v1/models`` with
``kind: "image"`` and are kept out of everything the chat roles read. A chat role
that resolved onto an image model, or the image role onto a chat model, would fail
every call, so most of what is pinned here is that the two never cross.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from anton.core.llm.provider import StreamToolResult
from pydantic import SecretStr
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.common.settings import user_settings as us
from cowork.common.settings.app_settings import IMAGE_MODEL_DEFAULT, MINDS_FREE_MODEL
from cowork.common.settings.user_settings import (
    SETTING_ENV_ALIASES,
    Provider,
    UserSettings,
    setting_is_operator_pinned,
    setting_is_org_scoped,
)
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.services.providers import MindsModelListing
from _fakes import FakeRequest

FLARE = IMAGE_MODEL_DEFAULT
SUNBURST = "gpt-image-2.6-sunburst"
CHAT_ENABLED = json.dumps({MINDS_FREE_MODEL: True, "sonnet": True})


def _minds(**kw) -> UserSettings:
    return UserSettings(
        planning_provider=Provider.MINDS_CLOUD,
        coding_provider=Provider.MINDS_CLOUD,
        minds_api_key=SecretStr("mdb_test"),
        **kw,
    )


def _mode(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("COWORK_TENANCY_MODE", mode)
    us.get_app_settings.cache_clear()


@pytest.fixture
def org_mode(monkeypatch):
    _mode(monkeypatch, "org")
    yield
    us.get_app_settings.cache_clear()


# ── The setting ───────────────────────────────────────────────────────


def test_a_fresh_install_starts_on_the_compiled_default():
    """Nothing fetched yet: every first message of a fresh install."""
    s = _minds()

    assert s.image_model == FLARE
    assert s.resolved_image_model == FLARE


def test_a_byok_install_gets_the_compiled_default_too():
    s = UserSettings(
        planning_provider=Provider.ANTHROPIC,
        coding_provider=Provider.ANTHROPIC,
        anthropic_api_key=SecretStr("sk-ant-test"),
    )

    assert s.resolved_image_model == FLARE


def test_the_catalogs_declared_default_wins_over_the_compiled_one():
    s = _minds(minds_role_defaults=json.dumps({"image": SUNBURST}))

    assert s.image_model == SUNBURST
    assert s.resolved_image_model == SUNBURST


def test_an_explicit_pick_is_never_overwritten_by_the_declared_default():
    s = _minds(
        image_model=FLARE,
        minds_role_defaults=json.dumps({"image": SUNBURST}),
        minds_image_model_enabled=json.dumps({FLARE: True, SUNBURST: True}),
    )

    assert s.image_model == FLARE
    assert s.resolved_image_model == FLARE


def test_a_locked_default_moves_to_an_enabled_image_model():
    s = _minds(
        minds_role_defaults=json.dumps({"image": SUNBURST}),
        minds_image_model_enabled=json.dumps({FLARE: True, SUNBURST: False}),
    )

    assert s.image_model == FLARE


def test_with_every_image_model_locked_the_default_stays_put():
    """No image model is free, so there is nothing to fall back to: the locked
    default is refused with a top-up path, never swapped for a chat model the
    chat availability map does affirm."""
    s = _minds(
        minds_model_enabled=CHAT_ENABLED,
        minds_image_model_enabled=json.dumps({FLARE: False, SUNBURST: False}),
    )

    assert s.image_model == FLARE
    assert s.resolved_image_model == FLARE


def test_a_retired_default_moves_to_a_served_image_model():
    s = _minds(minds_image_model_enabled=json.dumps({SUNBURST: False}))

    assert s.image_model == SUNBURST


def test_org_mode_never_hands_the_image_role_the_free_chat_model(org_mode):
    """The chat roles' org-mode fallback is the free chat model. The image role
    must keep the desktop rule instead, most of all before any catalog is cached,
    where the chat rule hands out the free model with no evidence at all."""
    fresh = UserSettings()
    assert fresh.planning_model == MINDS_FREE_MODEL
    assert fresh.image_model == fresh.resolved_image_model == FLARE

    s = UserSettings(minds_image_model_enabled=json.dumps({FLARE: False, SUNBURST: False}))

    assert s.planning_model == MINDS_FREE_MODEL
    assert s.image_model == FLARE
    assert s.resolved_image_model == FLARE

    unlocked = UserSettings(minds_image_model_enabled=json.dumps({FLARE: False, SUNBURST: True}))
    assert unlocked.resolved_image_model == SUNBURST


def test_a_pick_the_wallet_cannot_pay_for_resolves_to_one_it_can():
    s = _minds(
        image_model=SUNBURST,
        minds_image_model_enabled=json.dumps({FLARE: True, SUNBURST: False}),
    )

    assert s.image_model == SUNBURST  # the stored row is never rewritten
    assert s.resolved_image_model == FLARE


def test_a_chat_id_stored_as_the_image_model_resolves_to_an_image_model():
    s = _minds(
        image_model="sonnet",
        minds_model_enabled=CHAT_ENABLED,
        minds_image_model_enabled=json.dumps({FLARE: True, SUNBURST: True}),
    )

    assert s.resolved_image_model == FLARE


def test_a_members_restricted_image_model_resolves_to_another():
    s = _minds(
        image_model=SUNBURST,
        minds_image_model_enabled=json.dumps({FLARE: True, SUNBURST: True}),
        minds_model_restricted=json.dumps([SUNBURST]),
    )

    assert s._minds_image_enabled_map() == {FLARE: True, SUNBURST: False}
    assert s.resolved_image_model == FLARE


def test_image_models_never_reach_the_chat_roles():
    """The chat roles' first-enabled fallback reads only the chat map."""
    s = _minds(
        planning_model="retired-chat-model",
        minds_model_enabled=json.dumps({MINDS_FREE_MODEL: False, "sonnet": False}),
        minds_image_model_enabled=json.dumps({FLARE: True}),
    )

    assert s.resolved_planning_model != FLARE
    assert s.coding_model != FLARE


def test_the_setting_syncs_like_the_coding_model():
    """ENG-739: a model never rides the bulk .env sync, the operator pins it like
    the other models, and it is a personal setting, not an org one."""
    assert "image_model" not in SETTING_ENV_ALIASES
    assert setting_is_operator_pinned("image_model") is setting_is_operator_pinned("coding_model") is True
    assert setting_is_org_scoped("image_model") is setting_is_org_scoped("coding_model") is False
    assert setting_is_org_scoped("minds_image_model_enabled") is True
    assert setting_is_operator_pinned("minds_image_model_enabled") is True


# ── The catalog ───────────────────────────────────────────────────────


def _parse(monkeypatch, rows):
    from cowork.services import providers

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"data": rows}

    class _Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **kw):
            return _Response()

    monkeypatch.setattr(providers.httpx, "AsyncClient", _Client)
    providers._minds_models_cache.clear()
    return asyncio.run(providers.fetch_minds_models("https://api.mindshub.ai", "mdb_test"))


def test_image_rows_are_listed_apart_from_the_chat_models(monkeypatch):
    listing = _parse(monkeypatch, [
        {"id": MINDS_FREE_MODEL, "kind": "chat", "enabled": True,
         "default_for": ["planning", "coding", "router"]},
        {"id": FLARE, "kind": "image", "enabled": True, "label": "GPT Image 2.6 Flare",
         "provider": "openai", "default_for": ["image"]},
        {"id": SUNBURST, "kind": "image", "enabled": False, "disabled_reason": "wallet_empty"},
        {"id": "embed-small", "kind": "embedding", "embedding": True},
        {"id": "sonnet"},  # no kind: a chat model, as from an older MindsHub
    ])

    assert listing.ids == [MINDS_FREE_MODEL, "sonnet"]
    assert listing.enabled == {MINDS_FREE_MODEL: True}
    assert listing.image_models == {FLARE: True, SUNBURST: False}
    assert listing.role_defaults == {
        "planning": MINDS_FREE_MODEL, "coding": MINDS_FREE_MODEL, "router": MINDS_FREE_MODEL,
        "image": FLARE,
    }
    assert listing.labels[FLARE] == "GPT Image 2.6 Flare"
    assert listing.providers[FLARE] == "openai"
    assert listing.disabled_reasons == {SUNBURST: "wallet_empty"}


def test_an_image_row_without_an_enabled_flag_is_available(monkeypatch):
    listing = _parse(monkeypatch, [{"id": "sonnet"}, {"id": FLARE, "kind": "image"}])

    assert listing.image_models == {FLARE: True}


def test_only_an_image_row_may_claim_the_image_role_and_no_other(monkeypatch):
    listing = _parse(monkeypatch, [
        {"id": "sonnet", "default_for": ["image", "planning"]},
        {"id": FLARE, "kind": "image", "default_for": ["coding", "image"]},
    ])

    assert listing.role_defaults == {"planning": "sonnet", "image": FLARE}


def test_the_image_role_survives_the_stored_map_and_the_write_gate():
    from cowork.api.v1.endpoints.settings import _reject_malformed_role_defaults

    stored = json.dumps({"planning": "sonnet", "image": SUNBURST})
    assert _minds(minds_role_defaults=stored)._minds_role_default_map()["image"] == SUNBURST
    _reject_malformed_role_defaults({"minds_role_defaults": stored})


# ── The picker endpoint ───────────────────────────────────────────────


def _listing(*, image_models, enabled=None, role_defaults=None, disabled_reasons=None):
    return MindsModelListing(
        [MINDS_FREE_MODEL, "sonnet"], {}, enabled or {MINDS_FREE_MODEL: True, "sonnet": True},
        {FLARE: "GPT Image 2.6 Flare"}, {}, {}, role_defaults or {}, disabled_reasons or {},
        image_models,
    )


@pytest.fixture()
def session():
    import cowork.models.setting  # noqa: F401

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _desktop_picker(session, monkeypatch, listing) -> dict:
    from cowork.api.v1.endpoints import settings as ep
    from cowork.services.settings import SettingService

    async def fake_fetch(base_url, api_key, force_refresh=False, tenant_key=None):
        return listing

    monkeypatch.setattr(ep, "fetch_minds_models", fake_fetch)
    SettingService(session, LOCAL_SCOPE).upsert_setting("minds_api_key", "mdb_desktop")
    return asyncio.run(ep.recommended_models(FakeRequest(), session, LOCAL_SCOPE))


def test_the_chat_pickers_never_list_an_image_model(session, monkeypatch):
    payload = _desktop_picker(session, monkeypatch, _listing(image_models={FLARE: True, SUNBURST: True}))

    assert payload["recommendedModels"]["minds-cloud"] == [MINDS_FREE_MODEL, "sonnet"]
    assert len(payload["recommendedPair"]["minds-cloud"]) == 3
    assert FLARE not in payload["recommendedPair"]["minds-cloud"]


def test_the_image_picker_lists_only_image_models_with_their_state(session, monkeypatch):
    payload = _desktop_picker(session, monkeypatch, _listing(
        image_models={FLARE: True, SUNBURST: False},
        role_defaults={"image": FLARE},
        disabled_reasons={SUNBURST: "wallet_empty"},
    ))

    assert payload["imageModels"] == [FLARE, SUNBURST]
    assert payload["imageModelDefault"] == FLARE
    assert payload["modelEnabled"][SUNBURST] is False
    assert payload["modelDisabledReasons"][SUNBURST] == "wallet_empty"
    assert payload["modelLabels"][FLARE] == "GPT Image 2.6 Flare"


def test_the_picker_caches_what_resolution_reads(session, monkeypatch):
    from cowork.services.settings import SettingService

    payload = _desktop_picker(session, monkeypatch, _listing(
        image_models={FLARE: False, SUNBURST: True}, role_defaults={"image": FLARE},
    ))

    stored = SettingService(session, LOCAL_SCOPE).load()
    assert json.loads(stored.minds_image_model_enabled) == {FLARE: False, SUNBURST: True}
    assert json.loads(stored.minds_model_enabled) == {MINDS_FREE_MODEL: True, "sonnet": True}
    assert json.loads(stored.minds_role_defaults)["image"] == FLARE
    # The default the picker shows is the one a turn resolves to.
    assert payload["imageModelDefault"] == stored.resolved_image_model == SUNBURST


def test_a_catalog_without_image_models_keeps_the_stored_image_map(session, monkeypatch):
    from cowork.services.settings import SettingService

    _desktop_picker(session, monkeypatch, _listing(image_models={FLARE: True}))
    payload = _desktop_picker(session, monkeypatch, _listing(image_models={}))

    assert payload["imageModels"] == []
    assert json.loads(SettingService(session, LOCAL_SCOPE).load().minds_image_model_enabled) == {FLARE: True}


def test_an_org_members_restricted_image_model_is_theirs_alone(session, monkeypatch, org_mode):
    from cowork.api.v1.endpoints import settings as ep
    from cowork.services.settings import SettingService

    restricted_for = {"u-a"}

    async def fake_org_catalog(*, org_id, user_id, bearer_token, refresh=False):
        restricted = user_id in restricted_for
        return _listing(
            image_models={FLARE: not restricted, SUNBURST: True},
            disabled_reasons={FLARE: "model_restricted"} if restricted else {},
        )

    monkeypatch.setattr(ep, "fetch_org_model_catalog", fake_org_catalog)

    def scope(user_id):
        return TenantScope(org_mode=True, org_id="org-image", user_id=user_id)

    asyncio.run(ep.recommended_models(FakeRequest({"Authorization": "Bearer jwt-a"}), session, scope("u-a")))

    member_a = SettingService(session, scope("u-a")).load()
    member_b = SettingService(session, scope("u-b")).load()
    assert json.loads(member_b.minds_image_model_enabled) == {FLARE: True, SUNBURST: True}
    assert member_a.resolved_image_model == SUNBURST
    assert member_b.resolved_image_model == FLARE


# ── Handing it to anton ───────────────────────────────────────────────


def test_anton_gets_the_resolved_image_model():
    from pydantic_settings import BaseSettings

    from cowork.harnesses.anton_harness.harness import _overlay_user_settings

    class AntonWithImages(BaseSettings):
        model_config = {"env_prefix": "ANTON_", "extra": "ignore"}
        planning_model: str = "default-planning"
        image_model: str | None = None

    a = AntonWithImages()
    user = _minds(
        image_model=SUNBURST,
        minds_image_model_enabled=json.dumps({FLARE: True, SUNBURST: False}),
    )

    applied = _overlay_user_settings(a, user)

    assert "image_model" in applied
    assert a.image_model == FLARE


def test_an_anton_without_the_image_setting_is_skipped_not_crashed():
    """The pinned anton can predate `image_model` (see _overlay_user_settings)."""
    from anton.config.settings import AntonSettings

    from cowork.harnesses.anton_harness.harness import _overlay_user_settings

    a = AntonSettings(_env_file=None)
    applied = _overlay_user_settings(a, _minds())

    assert ("image_model" in applied) == hasattr(a, "image_model")


class _RuntimeWithOverlay:
    calls: list[dict] = []

    def __init__(self, name, *, coding_provider, coding_model, coding_api_key, coding_base_url,
                 workspace_path, workspace_env_overlay=None, cell_sandbox=None):
        type(self).calls.append({"workspace_env_overlay": workspace_env_overlay})


class _RuntimeWithImageModel:
    calls: list[dict] = []

    def __init__(self, name, *, coding_provider, coding_model, coding_api_key, coding_base_url,
                 workspace_path, workspace_env_overlay=None, image_model=None, cell_sandbox=None):
        type(self).calls.append({"image_model": image_model, "workspace_env_overlay": workspace_env_overlay})


@pytest.fixture
def sandbox_off(monkeypatch):
    monkeypatch.delenv("COWORK_CODE_SANDBOX", raising=False)
    us.get_app_settings.cache_clear()
    yield
    us.get_app_settings.cache_clear()


def _make(runtime_cls, monkeypatch, tmp_path, image_model):
    from cowork.services import scratchpad_runtime

    monkeypatch.setattr(runtime_cls, "calls", [])
    monkeypatch.setattr("anton.core.backends.local.LocalScratchpadRuntime", runtime_cls)
    scratchpad_runtime._make_runtime(
        "app", workspace=tmp_path, coding_provider="openai", coding_model="m",
        coding_api_key="k", coding_base_url="u", image_model=image_model,
    )
    (call,) = runtime_cls.calls
    return call


def test_a_pad_gets_the_image_model_in_both_env_names(sandbox_off, monkeypatch, tmp_path):
    call = _make(_RuntimeWithOverlay, monkeypatch, tmp_path, FLARE)

    assert call["workspace_env_overlay"] == {
        "ANTON_IMAGE_MODEL": FLARE, "ANTON_SCRATCHPAD_IMAGE_MODEL": FLARE,
    }


def test_a_pad_takes_anton_s_own_keyword_when_it_has_one(sandbox_off, monkeypatch, tmp_path):
    call = _make(_RuntimeWithImageModel, monkeypatch, tmp_path, FLARE)

    assert call == {"image_model": FLARE, "workspace_env_overlay": None}


def test_no_image_model_leaves_the_pad_call_as_it_was(sandbox_off, monkeypatch, tmp_path):
    assert _make(_RuntimeWithOverlay, monkeypatch, tmp_path, "") == {"workspace_env_overlay": None}


def test_the_pool_hands_its_pads_the_resolved_image_model(sandbox_off, monkeypatch, tmp_path):
    from cowork.services import scratchpad_runtime

    settings = _minds(
        image_model=SUNBURST,
        minds_image_model_enabled=json.dumps({FLARE: True, SUNBURST: False}),
    )
    monkeypatch.setattr("cowork.common.settings.user_settings.get_user_settings", lambda: settings)
    monkeypatch.setattr(_RuntimeWithOverlay, "calls", [])
    monkeypatch.setattr("anton.core.backends.local.LocalScratchpadRuntime", _RuntimeWithOverlay)
    monkeypatch.setattr(scratchpad_runtime, "_pads", {})

    scratchpad_runtime.get_or_create("app", workspace_path=str(tmp_path))

    (call,) = _RuntimeWithOverlay.calls
    assert call["workspace_env_overlay"]["ANTON_SCRATCHPAD_IMAGE_MODEL"] == FLARE


# ── Only where an image model can be served ──────────────────────────


def _settings(**kw) -> UserSettings:
    return UserSettings(_env_file=None, **kw)


def _gemini_only(**kw) -> UserSettings:
    return _settings(
        planning_provider=Provider.GEMINI, coding_provider=Provider.GEMINI, router_provider=Provider.GEMINI,
        gemini_api_key=SecretStr("gm-test"), **kw,
    )


def _custom_endpoint_only(**kw) -> UserSettings:
    return _settings(
        planning_provider=Provider.OPENAI_COMPATIBLE, coding_provider=Provider.OPENAI_COMPATIBLE,
        router_provider=Provider.OPENAI_COMPATIBLE, planning_model="llama3", coding_model="llama3",
        router_model="llama3", openai_base_url="http://localhost:11434/v1", **kw,
    )


@pytest.mark.parametrize(
    ("settings", "handed"),
    [
        pytest.param(lambda: _gemini_only(), None, id="gemini-key-only"),
        pytest.param(lambda: _custom_endpoint_only(), None, id="custom-endpoint-only"),
        pytest.param(
            lambda: _settings(
                planning_provider=Provider.ANTHROPIC, coding_provider=Provider.ANTHROPIC,
                router_provider=Provider.ANTHROPIC, anthropic_api_key=SecretStr("sk-ant-test"),
            ),
            None, id="anthropic-key-only",
        ),
        pytest.param(lambda: _gemini_only(image_model=SUNBURST), SUNBURST, id="gemini-with-an-explicit-pick"),
        pytest.param(
            lambda: _custom_endpoint_only(minds_api_key=SecretStr("mdb_test")), FLARE,
            id="custom-endpoint-with-a-mindshub-key",
        ),
        pytest.param(
            lambda: _settings(
                planning_provider=Provider.OPENAI, coding_provider=Provider.OPENAI,
                router_provider=Provider.OPENAI, openai_api_key=SecretStr("sk-test"),
            ),
            FLARE, id="openai-key",
        ),
        pytest.param(
            lambda: _settings(
                planning_provider=Provider.ANTHROPIC, coding_provider=Provider.OPENAI,
                router_provider=Provider.ANTHROPIC, anthropic_api_key=SecretStr("sk-ant-test"),
                openai_api_key=SecretStr("sk-test"),
            ),
            FLARE, id="one-role-on-openai",
        ),
        pytest.param(lambda: _minds(), FLARE, id="mindshub-key"),
    ],
)
def test_the_image_model_is_handed_over_only_where_it_can_be_served(settings, handed):
    s = settings()

    assert s.image_model_for_agent == handed
    # The turn's snapshot is a copy: it must not read a filled default as a pick.
    assert s.model_copy().image_model_for_agent == handed


def test_org_mode_always_hands_the_image_model_over(org_mode):
    assert UserSettings().image_model_for_agent == FLARE


def test_a_stored_pick_counts_as_a_pick(session):
    from cowork.services.settings import SettingService

    service = SettingService(session, LOCAL_SCOPE)
    for key, value in (("planning_provider", "gemini"), ("coding_provider", "gemini"),
                       ("router_provider", "gemini"), ("gemini_api_key", "gm-test")):
        service.upsert_setting(key, value)
    assert service.load().image_model_for_agent is None

    service.upsert_setting("image_model", SUNBURST)

    assert service.load().image_model_for_agent == SUNBURST


def test_anton_gets_no_image_model_where_none_can_be_served():
    from pydantic_settings import BaseSettings

    from cowork.harnesses.anton_harness.harness import _overlay_user_settings

    class AntonWithImages(BaseSettings):
        model_config = {"env_prefix": "ANTON_", "extra": "ignore"}
        image_model: str | None = None

    a = AntonWithImages()

    assert "image_model" not in _overlay_user_settings(a, _gemini_only())
    assert a.image_model is None


def test_a_pad_gets_no_image_model_where_none_can_be_served(sandbox_off, monkeypatch, tmp_path):
    from cowork.services import scratchpad_runtime

    settings = _custom_endpoint_only()
    monkeypatch.setattr("cowork.common.settings.user_settings.get_user_settings", lambda: settings)
    monkeypatch.setattr(_RuntimeWithOverlay, "calls", [])
    monkeypatch.setattr("anton.core.backends.local.LocalScratchpadRuntime", _RuntimeWithOverlay)
    monkeypatch.setattr(scratchpad_runtime, "_pads", {})

    scratchpad_runtime.get_or_create("app", workspace_path=str(tmp_path))

    assert _RuntimeWithOverlay.calls == [{"workspace_env_overlay": None}]


# ── Showing the images ────────────────────────────────────────────────


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _image(root, rel="images/cat.png"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG)
    return path


def test_a_desktop_result_names_the_project_file(tmp_path):
    from cowork.harnesses.anton_harness.agent_files import AgentFiles
    from cowork.harnesses.anton_harness.stream_formatter import generated_images

    files = AgentFiles.on_this_server(tmp_path)
    cat = _image(tmp_path)
    _image(tmp_path, "images/dog.webp")
    (tmp_path / "notes.txt").write_text("not an image")
    result = json.dumps({
        "images": [
            {"path": str(cat), "width": 1024, "height": 1024},
            {"path": "images/dog.webp"},
            {"path": str(cat)},  # named twice, shown once
            {"path": "notes.txt"},
            {"path": "images/missing.png"},
        ],
        "model": FLARE,
    })

    assert generated_images(result, files) == [{"path": "images/cat.png"}, {"path": "images/dog.webp"}]


def test_a_cloud_pods_result_names_the_file_in_the_conversations_folder(tmp_path):
    from cowork.harnesses.anton_harness.agent_files import AgentFiles
    from cowork.harnesses.anton_harness.stream_formatter import generated_images

    conversation_id = uuid.uuid4()
    _image(tmp_path / "conversations" / str(conversation_id))
    files = AgentFiles.in_cloud_pod(tmp_path, conversation_id=conversation_id, workspace_rel_path=None)

    result = json.dumps({"path": "/workspace/images/cat.png"})

    assert generated_images(result, files) == [{"path": f"conversations/{conversation_id}/images/cat.png"}]


def _artifact_image(root, slug="cat", name="cat.png"):
    return _image(root, f".anton/artifacts/{slug}/{name}")


def _anton_result(path, absolute_path=None):
    """generate_image's result: `path` in the project's terms, `absolute_path`
    where the agent's OS wrote it."""
    result = {"status": "generated", "path": path, "mime_type": "image/png", "width": 1024, "height": 1024}
    if absolute_path is not None:
        result["absolute_path"] = absolute_path
    return json.dumps(result)


def _pod(tmp_path, conversation_id, *, workspace_rel_path="org-a/project-a"):
    from cowork.harnesses.anton_harness.agent_files import AgentFiles

    return AgentFiles.in_cloud_pod(tmp_path, conversation_id=conversation_id, workspace_rel_path=workspace_rel_path)


def test_a_desktop_image_artifact_names_its_project_file(tmp_path):
    from cowork.harnesses.anton_harness.agent_files import AgentFiles
    from cowork.harnesses.anton_harness.stream_formatter import generated_images

    saved = _artifact_image(tmp_path)
    files = AgentFiles.on_this_server(tmp_path)
    expected = [{"path": ".anton/artifacts/cat/cat.png"}]

    assert generated_images(_anton_result(".anton/artifacts/cat/cat.png", str(saved)), files) == expected
    assert generated_images(_anton_result(".anton/artifacts/cat/cat.png"), files) == expected


@pytest.mark.parametrize(
    "result",
    [
        _anton_result(".anton/artifacts/cat/cat.png", "/project-artifacts/cat/cat.png"),
        _anton_result(".anton/artifacts/cat/cat.png"),
        _anton_result("/project-artifacts/cat/cat.png"),
    ],
    ids=["both", "path-only", "absolute-path-only"],
)
def test_a_cloud_image_artifact_names_the_projects_artifacts_folder(tmp_path, result):
    """The default case on a pod: a new image artifact, written through the pod's
    mount of the project's artifacts folder, not under the conversation's."""
    from cowork.harnesses.anton_harness.stream_formatter import generated_images

    _artifact_image(tmp_path)

    assert generated_images(result, _pod(tmp_path, uuid.uuid4())) == [{"path": ".anton/artifacts/cat/cat.png"}]


def test_a_cloud_workspace_image_names_the_conversations_folder(tmp_path):
    from cowork.harnesses.anton_harness.stream_formatter import generated_images

    conversation_id = uuid.uuid4()
    _image(tmp_path)  # the same name at the project root must not be the answer
    _image(tmp_path / "conversations" / str(conversation_id))
    files = _pod(tmp_path, conversation_id)
    expected = [{"path": f"conversations/{conversation_id}/images/cat.png"}]

    assert generated_images(_anton_result("images/cat.png", "/workspace/images/cat.png"), files) == expected
    assert generated_images(_anton_result("images/cat.png"), files) == expected


def test_a_pod_whose_artifacts_are_not_this_projects_names_no_artifact(tmp_path):
    """With the project's path unknown, the pod's artifacts mount is the default
    project's, and its working directory's `.anton/artifacts` is not where the
    image went either, so nothing is shown rather than a wrong picture."""
    from cowork.harnesses.anton_harness.stream_formatter import generated_images

    conversation_id = uuid.uuid4()
    _artifact_image(tmp_path)
    _artifact_image(tmp_path / "conversations" / str(conversation_id))
    files = _pod(tmp_path, conversation_id, workspace_rel_path=None)

    result = _anton_result(".anton/artifacts/cat/cat.png", "/project-artifacts/cat/cat.png")
    assert generated_images(result, files) == []


def test_an_artifact_path_cannot_climb_or_link_out_of_the_artifacts_folder(tmp_path):
    from cowork.harnesses.anton_harness.stream_formatter import generated_images

    _image(tmp_path, "secret.png")
    artifact = _artifact_image(tmp_path)
    (artifact.parent / "link.png").symlink_to(tmp_path / "secret.png")
    files = _pod(tmp_path, uuid.uuid4())

    for result in (
        _anton_result(".anton/artifacts/../../secret.png"),
        _anton_result("/project-artifacts/../secret.png"),
        _anton_result(".anton/artifacts/cat/link.png", "/project-artifacts/cat/link.png"),
    ):
        assert generated_images(result, files) == [], result


def test_a_refusal_names_no_image(tmp_path):
    from cowork.harnesses.anton_harness.agent_files import AgentFiles
    from cowork.harnesses.anton_harness.stream_formatter import generated_images

    files = AgentFiles.on_this_server(tmp_path)

    assert generated_images("Image refused: moderation_blocked", files) == []


async def _stream(*events, sink=None):
    from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream

    async def source():
        for event in events:
            yield event

    parsed = []
    async for chunk in format_responses_stream(source(), model="m", event_sink=sink):
        for line in chunk.splitlines():
            if line.startswith("data:"):
                parsed.append(json.loads(line[len("data:"):].strip()))
    return [e for e in parsed if e.get("type") == "response.in_progress" and e.get("tool_use_id")]


def _image_result(content, tool_use_id="t1"):
    return StreamToolResult(name="generate_image", content=content, id=tool_use_id)


async def test_an_image_result_is_its_own_saved_event(tmp_path):
    """The UI renders it inline from the result JSON; it is saved with the turn
    (through the event sink) so a reload shows it, and it is never read as a
    scratchpad cell."""
    from cowork.harnesses.anton_harness.agent_files import AgentFiles
    from cowork.harnesses.anton_harness.scratchpad_cell_replay import extract_scratchpad_cells
    from cowork.harnesses.anton_harness.stream_formatter import TurnContext

    cat = _image(tmp_path)
    content = json.dumps({
        "path": "images/cat.png", "width": 1024, "height": 1024, "mime_type": "image/png",
        "model": FLARE, "usage": {"input_tokens": 16, "output_tokens": 824},
    })
    context = TurnContext(question="draw a cat", files=AgentFiles.on_this_server(tmp_path))
    saved: list[dict] = []

    (event,) = await _stream(context, _image_result(content), sink=lambda _type, data: saved.append(data))

    assert event["thought_role"] == "thought.tool_call.result"
    assert event["tool_name"] == "generate_image"
    assert event["tool_use_id"] == "t1"
    assert event["content"] == content
    assert event["generated_images"] == [{"path": "images/cat.png"}]
    assert "cell_status" not in event
    assert event in saved
    assert extract_scratchpad_cells(saved) == []
    assert cat.is_file()


async def test_an_image_result_is_not_a_figure_step(tmp_path, monkeypatch):
    """Out of turn_results: the figure sources never read an image as a step."""
    from cowork.harnesses.anton_harness import stream_formatter

    seen = []

    async def capture(evidence):
        seen.append(evidence)
        return stream_formatter.FigureSources()

    monkeypatch.setattr(stream_formatter, "find_answer_sources", capture)

    await _stream(_image_result(json.dumps({"path": "images/cat.png"})))

    (evidence,) = seen
    assert evidence.steps == ()


async def test_other_tools_keep_the_scratchpad_result_role(tmp_path):
    content = json.dumps({"path": str(_image(tmp_path))})

    (event,) = await _stream(StreamToolResult(name="scratchpad", content=content, id="t2"))

    assert event["thought_role"] == "thought.scratchpad.result"
    assert "generated_images" not in event


async def test_a_turn_with_no_known_project_still_streams_the_image(tmp_path):
    content = json.dumps({"path": str(_image(tmp_path))})

    (event,) = await _stream(_image_result(content))

    assert event["thought_role"] == "thought.tool_call.result"
    assert event["content"] == content
    assert event["generated_images"] == []


async def test_an_image_that_resolves_to_no_project_file_names_none(tmp_path):
    """Always present, empty when nothing resolves, so the UI never falls back to
    the raw path: on a cloud pod that is relative to the conversation's folder,
    and read at the project root it names a different file or none."""
    from cowork.harnesses.anton_harness.agent_files import AgentFiles
    from cowork.harnesses.anton_harness.stream_formatter import TurnContext

    conversation_id = uuid.uuid4()
    _image(tmp_path)  # images/cat.png at the project root, not in the conversation's folder
    files = AgentFiles.in_cloud_pod(tmp_path, conversation_id=conversation_id, workspace_rel_path=None)
    context = TurnContext(question="draw a cat", files=files)

    (event,) = await _stream(context, _image_result(json.dumps({"path": "images/cat.png"})))

    assert event["generated_images"] == []


@pytest.fixture
def local_project(tmp_path, monkeypatch, cleanup_tmp_projects):
    from fastapi.testclient import TestClient

    from cowork.db.session import get_engine
    from cowork.models.project import Project

    _mode(monkeypatch, "local")
    try:
        name = f"images-{uuid.uuid4().hex[:8]}"
        root = tmp_path / name
        root.mkdir()
        engine = get_engine(us.get_app_settings().database.uri)
        with Session(engine) as db:
            db.add(Project(id=uuid.uuid4(), name=name, path=str(root), org_id=None))
            db.commit()

        from cowork.server import create_app

        client = TestClient(create_app())
        try:
            yield client, name, root
        finally:
            client.close()
    finally:
        us.get_app_settings.cache_clear()


@pytest.mark.parametrize(
    ("rel", "content_type"),
    [("images/cat.png", "image/png"), ("images/cat.jpeg", "image/jpeg"), ("images/cat.webp", "image/webp")],
)
def test_the_ui_can_fetch_a_generated_image_with_its_type(local_project, rel, content_type):
    """The path `generated_images` names is the one `files-raw` serves, typed so
    the UI's blob: URL renders as an image."""
    client, project, root = local_project
    _image(root, rel)

    response = client.get(f"/api/v1/projects/{project}/files-raw/{rel}")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == content_type
    assert response.content == PNG
