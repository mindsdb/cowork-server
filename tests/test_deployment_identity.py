"""The server uses one trusted pack snapshot for every self-hosted turn."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

import cowork.server as cowork_server
from cowork.common.settings.app_settings import AppSettings, get_app_settings
from cowork.common.settings import user_settings
from cowork.common.settings.user_settings import UserSettings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.deployment_identity import identity_for_turn
from tests._turns import built_turn

PRODUCT = "Example Workbench"
SUB = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
PARTITION = "44e92829-e701-599b-88f9-69f5bf1165eb"


@pytest.fixture(autouse=True)
def isolated_identity(monkeypatch, tmp_path):
    for name in ("COWORK_DEPLOYMENT_KIND", "COWORK_DEPLOYMENT_BRAND_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


def _enable(monkeypatch, path: Path | None) -> None:
    monkeypatch.setenv("COWORK_DEPLOYMENT_KIND", "self-hosted")
    if path is not None:
        monkeypatch.setenv("COWORK_DEPLOYMENT_BRAND_FILE", str(path))
    get_app_settings.cache_clear()


def _pack(tmp_path, *, name=PRODUCT) -> Path:
    path = tmp_path / "pack.json"
    path.write_text(json.dumps({"name": name, "browserTitle": "ignored"}), encoding="utf-8")
    return path


def _older_config(*missing):
    from anton.core.session import ChatSessionConfig

    fields = []
    for field in dataclasses.fields(ChatSessionConfig):
        if field.name in missing:
            continue
        if field.default is not dataclasses.MISSING:
            default = dataclasses.field(default=field.default)
        elif field.default_factory is not dataclasses.MISSING:
            default = dataclasses.field(default_factory=field.default_factory)
        else:
            default = dataclasses.field()
        fields.append((field.name, field.type, default))
    return dataclasses.make_dataclass("ChatSessionConfig", fields)


def test_kind_is_explicit_and_not_a_member_setting():
    with pytest.raises(ValidationError):
        AppSettings(_env_file=None, COWORK_DEPLOYMENT_KIND="web")
    assert "deployment_kind" not in UserSettings.model_fields
    assert "deployment_brand_file" not in UserSettings.model_fields
    assert "deployment_identity" not in UserSettings.model_fields


@pytest.mark.parametrize("org", [False, True])
def test_unset_keeps_defaults_without_importing_the_new_anton_model(monkeypatch, tmp_path, org):
    import anton.core.llm.identity as identity_module

    monkeypatch.delattr(identity_module, "DeploymentIdentity")
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org" if org else "local")
    # A path alone never opts in or reads a file.
    monkeypatch.setenv("COWORK_DEPLOYMENT_BRAND_FILE", str(tmp_path / "missing.json"))
    get_app_settings.cache_clear()
    cowork_server.create_app()
    assert identity_for_turn() is None


@pytest.mark.parametrize("name", ["Cowork", PRODUCT, "Équipe Workbench"])
def test_startup_freezes_the_validated_pack_name(monkeypatch, tmp_path, name):
    pack = _pack(tmp_path, name=name)
    _enable(monkeypatch, pack)
    cowork_server.create_app()
    identity = identity_for_turn()
    assert identity.kind == "self-hosted" and identity.product_name == name
    pack.write_text('{"name":"Replacement"}', encoding="utf-8")
    cowork_server.create_app()
    assert identity_for_turn() is identity
    assert identity_for_turn().product_name == name
    with pytest.raises(ValidationError):
        identity.product_name = "Member override"


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32"])
def test_startup_accepts_the_pack_loaders_json_encodings(monkeypatch, tmp_path, encoding):
    # cowork-self-hosted scripts/branding.py:read_pack_json passes bytes to
    # json.loads, which accepts a BOM and detects UTF-16/32 as well as UTF-8.
    pack = tmp_path / "pack.json"
    pack.write_bytes(json.dumps({"name": PRODUCT}).encode(encoding))
    _enable(monkeypatch, pack)

    cowork_server.create_app()

    assert identity_for_turn().product_name == PRODUCT


def test_startup_accepts_a_valid_pack_larger_than_64_kib(monkeypatch, tmp_path):
    # The existing operator pack contract does not limit JSON formatting size.
    pack = tmp_path / "pack.json"
    pack.write_bytes(b" " * (64 * 1024) + json.dumps({"name": PRODUCT}).encode())
    _enable(monkeypatch, pack)

    cowork_server.create_app()

    assert identity_for_turn().product_name == PRODUCT


def test_startup_refuses_duplicate_pack_names(monkeypatch, tmp_path):
    pack = tmp_path / "pack.json"
    pack.write_text('{"name":"First name","name":"Second name"}', encoding="utf-8")
    _enable(monkeypatch, pack)

    with pytest.raises(RuntimeError, match="JSON object with a valid name"):
        cowork_server.create_app()


def test_explicit_identity_cannot_run_before_startup(monkeypatch, tmp_path):
    _enable(monkeypatch, _pack(tmp_path))
    with pytest.raises(RuntimeError, match="not initialized"):
        identity_for_turn()


def test_unreadable_pack_refuses_startup(monkeypatch, tmp_path):
    pack = _pack(tmp_path)
    pack.chmod(0)
    _enable(monkeypatch, pack)
    try:
        with pytest.raises(RuntimeError, match="readable regular file"):
            cowork_server.create_app()
    finally:
        pack.chmod(0o600)


@pytest.mark.parametrize("path", [None, Path("pack.json"), Path("/opt/../pack.json")])
def test_explicit_identity_requires_an_absolute_pack(monkeypatch, path):
    _enable(monkeypatch, path)
    with pytest.raises(RuntimeError, match="absolute path"):
        cowork_server.create_app()


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink", "parent_symlink", "fifo"])
def test_startup_refuses_unsafe_pack_files(monkeypatch, tmp_path, kind):
    path = tmp_path / "pack.json"
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        target = tmp_path / "source.json"
        target.write_text(json.dumps({"name": PRODUCT}))
        path.symlink_to(target)
    elif kind == "parent_symlink":
        folder = tmp_path / "real"
        folder.mkdir()
        _pack(folder)
        link = tmp_path / "linked"
        link.symlink_to(folder, target_is_directory=True)
        path = link / "pack.json"
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFOs require POSIX")
        os.mkfifo(path)
    _enable(monkeypatch, path)
    with pytest.raises(RuntimeError, match="readable regular file"):
        cowork_server.create_app()


@pytest.mark.parametrize("name", [None, 23, "", " ", " padded ", "x" * 65, "line\nbreak", "hidden\u200bname"])
def test_startup_refuses_invalid_names_without_echoing_them(monkeypatch, tmp_path, name):
    _enable(monkeypatch, _pack(tmp_path, name=name))
    with pytest.raises(RuntimeError, match="JSON object with a valid name") as refused:
        cowork_server.create_app()
    if isinstance(name, str) and len(name) > 10:
        assert name not in str(refused.value)


@pytest.mark.parametrize("raw", [b"not json", b"[]", b"{}", b"\xff"])
def test_startup_refuses_malformed_packs(monkeypatch, tmp_path, raw):
    pack = tmp_path / "pack.json"
    pack.write_bytes(raw)
    _enable(monkeypatch, pack)
    with pytest.raises(RuntimeError, match="JSON object with a valid name"):
        cowork_server.create_app()


@pytest.mark.parametrize("remote,tenancy", [(True, "local"), (False, "org")])
def test_explicit_identity_refuses_turn_paths_that_cannot_carry_it(monkeypatch, tmp_path, remote, tenancy):
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote" if remote else "in_process")
    monkeypatch.setenv("COWORK_TENANCY_MODE", tenancy)
    _enable(monkeypatch, _pack(tmp_path))
    with pytest.raises(RuntimeError, match="local, in-process turns"):
        cowork_server.create_app()


@pytest.mark.parametrize("missing", ["deployment_identity", "learned_memory_scope"])
def test_explicit_identity_requires_both_anton_config_fields(monkeypatch, tmp_path, missing):
    monkeypatch.setattr("anton.core.session.ChatSessionConfig", _older_config(missing))
    _enable(monkeypatch, _pack(tmp_path))
    with pytest.raises(RuntimeError, match="ChatSessionConfig"):
        cowork_server.create_app()


def test_explicit_identity_requires_the_anton_model(monkeypatch, tmp_path):
    import anton.core.llm.identity as identity_module

    monkeypatch.delattr(identity_module, "DeploymentIdentity")
    _enable(monkeypatch, _pack(tmp_path))
    with pytest.raises(RuntimeError, match="needs an Anton with DeploymentIdentity"):
        cowork_server.create_app()


@pytest.mark.parametrize("missing", ["build_memory_context", "get_scratchpad_context", "memory_presence"])
def test_explicit_identity_requires_anton_memory_support(monkeypatch, tmp_path, missing):
    from anton.core.memory.cortex import Cortex
    from anton.core.memory.hippocampus import Hippocampus

    if missing == "memory_presence":
        monkeypatch.delattr(Hippocampus, missing)
    else:
        monkeypatch.setattr(Cortex, missing, lambda self: "")
    _enable(monkeypatch, _pack(tmp_path))
    with pytest.raises(RuntimeError, match="scoped memory context"):
        cowork_server.create_app()


async def test_unset_builds_a_real_harness_config_with_an_older_anton(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "anton.core.session.ChatSessionConfig", _older_config("deployment_identity", "learned_memory_scope")
    )
    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=LOCAL_SCOPE) as turn:
        assert not hasattr(turn.config, "deployment_identity")
        assert not hasattr(turn.config, "learned_memory_scope")


@pytest.mark.parametrize("name", ["Cowork", PRODUCT])
@pytest.mark.parametrize(
    "enabled,mode,has_person,shared,expected",
    [
        (True, "autopilot", True, False, "personal"),
        (True, "copilot", True, False, "personal"),
        (True, "autopilot", True, True, "shared"),
        (False, "autopilot", True, False, "off"),
        (True, "off", True, False, "personal"),
        (True, "autopilot", False, False, "unavailable"),
    ],
)
async def test_real_harness_passes_only_the_operator_identity_and_actual_memory_scope(
    monkeypatch, tmp_path, personal_deployment, name, enabled, mode, has_person, shared, expected
):
    from anton.core.session import ChatSession

    resources = ALL_PERSONAL_RESOURCES - {PersonalResource.MEMORY} if shared else ALL_PERSONAL_RESOURCES
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(resources)))
    monkeypatch.setenv("COWORK_SURFACE", "web")
    _enable(monkeypatch, _pack(tmp_path, name=name))
    cowork_server.create_app()
    identity = identity_for_turn()
    scope = TenantScope(personal=True, org_id=PARTITION, user_id=SUB, personal_resources=resources) if has_person else LOCAL_SCOPE

    def configure_member(_path):
        # built_turn has installed its real UserSettings by this point.
        saved = user_settings.get_user_settings()
        saved.memory_enabled = enabled
        saved.memory_mode = mode
        saved.__dict__["deployment_identity"] = {"kind": "self-hosted", "product_name": "Member override"}
        saved.__dict__["deployment_brand_file"] = "/member/pack.json"

    async with built_turn(
        monkeypatch=monkeypatch, tmp_path=tmp_path, scope=scope, before_build=configure_member
    ) as turn:
        assert turn.config.deployment_identity is identity
        assert turn.config.deployment_identity.product_name == name
        assert turn.config.learned_memory_scope == expected
        assert turn.config.surface == "web"
        assert "The Cowork UI displays" in turn.config.system_prompt_context.suffix
        # Exercise Anton's real prompt builder with the config the server
        # hands it, without allowing prompt assembly to call a model.
        calls = [AsyncMock(side_effect=AssertionError("Prompt capture must not call a model")) for _ in range(2)]
        for method, call in zip(("plan", "code"), calls):
            monkeypatch.setattr(turn.config.llm_client, method, call)
        session = ChatSession(turn.config)
        try:
            prompt = await session._build_system_prompt("Who are you and what do you remember about me?")
        finally:
            await session.close()
        for call in calls:
            call.assert_not_awaited()
        assert f'Deployment product name: "{name}"' in prompt
        assert "made by MindsHub" in prompt
        assert "self-hosted deployment accessed in the browser" in prompt
        assert "The operator installs, updates and removes this deployment" in prompt
        assert "This conversation is running in the Cowork desktop app" not in prompt
        assert "desktop UI" not in prompt
        assert "desktop chat" not in prompt
        (publish,) = [tool for tool in turn.config.tools if tool.name == "publish_or_preview"]
        assert "desktop app" not in publish.description
        assert "Member override" not in prompt
        notices = {
            "personal": "No personal memories are saved",
            "shared": "Learned memory is shared across people",
            "off": "Learned memory is off",
            "unavailable": "Personal learned memory is unavailable",
        }
        assert notices[expected] in prompt
        assert "not remembered user preferences" in prompt


async def test_self_hosted_publish_tool_names_neither_the_desktop_app_nor_mindshub(monkeypatch, tmp_path):
    from cowork.harnesses.anton_harness import tools

    _enable(monkeypatch, _pack(tmp_path))
    cowork_server.create_app()
    report = tmp_path / "report.html"
    report.write_text("<h1>report</h1>", encoding="utf-8")
    monkeypatch.setattr(tools, "_published_state", lambda raw_path: {})
    publish = tools.build_cowork_publish_tool()

    answer = await publish.handler(SimpleNamespace(), {"file_path": str(report), "action": "ask"})

    assert "NOT been published" in answer
    assert "publishing works directly from chat, no slash command needed" in answer
    for text in (publish.description, publish.prompt, answer):
        assert "desktop" not in text
        assert "MindsHub" not in text
