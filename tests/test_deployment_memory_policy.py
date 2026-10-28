"""Deployment explanations follow access policy, not the encoding switch."""
from __future__ import annotations

import json

import pytest

from anton.core.memory.hippocampus import Hippocampus
from anton.core.session import ChatSession
from cowork.common.settings import user_settings
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.harnesses.memory.store import GlobalMemoryStore
from cowork.server import create_app
from tests._turns import built_turn


@pytest.mark.parametrize("trusted,shared,enabled,mode", [
    (True, False, True, "off"),
    (True, True, True, "off"),
    (False, False, False, "autopilot"),
])
async def test_encoding_off_keeps_existing_saved_memory_readable(
    monkeypatch, tmp_path, personal_deployment, trusted, shared, enabled, mode,
):
    resources = ALL_PERSONAL_RESOURCES - {PersonalResource.MEMORY} if shared else ALL_PERSONAL_RESOURCES
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", str(trusted).lower())
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(resources)))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_DEPLOYMENT_KIND", "self-hosted")
    pack = tmp_path / "pack.json"
    pack.write_text(json.dumps({"name": "Acme Cowork"}))
    monkeypatch.setenv("COWORK_DEPLOYMENT_BRAND_FILE", str(pack))
    get_app_settings.cache_clear()
    create_app()
    scope = TenantScope(
        personal=True, org_id="44e92829-e701-599b-88f9-69f5bf1165eb",
        user_id="9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d", personal_resources=resources,
    ) if trusted else LOCAL_SCOPE
    saved = Hippocampus(GlobalMemoryStore(scope=scope).root)
    saved.rewrite_identity(["Prefers calendar quarters"])
    saved.encode_lesson("Use scratchpad for tables", topic="scratchpad")
    before = {path: path.read_bytes() for path in saved._dir.iterdir() if path.is_file()}

    def configure(_path):
        settings = user_settings.get_user_settings()
        settings.memory_enabled = enabled
        settings.memory_mode = mode

    async with built_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, scope=scope, before_build=configure) as turn:
        session = ChatSession(turn.config)
        try:
            prompt = await session._build_system_prompt("What do you remember?")
            assert "Prefers calendar quarters" in prompt
            assert "Stored memories are not read" not in prompt
            assert "No personal memories are saved" not in prompt
            assert turn.config.learned_memory_scope == ("shared" if shared else "personal")
            assert "Use scratchpad for tables" in turn.config.cortex.get_scratchpad_context(
                learned_memory_scope=turn.config.learned_memory_scope,
            )
            assert {path: path.read_bytes() for path in before} == before
        finally:
            await session.close()
