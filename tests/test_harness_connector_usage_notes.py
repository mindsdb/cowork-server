"""Connector usage notes reach anton's ChatSessionConfig, and the general Drive
API rules moved out of the Picker block into google_drive's spec, so they
appear once and for every Drive connection."""
import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from pydantic import SecretStr

from cowork.common.settings.user_settings import Provider, UserSettings
from cowork.harnesses.anton_harness import harness
from cowork.services.connectors.specs._registry import registry


def test_google_drive_spec_carries_the_drive_api_rules():
    notes = registry.usage_notes_for(["google_drive"])["google_drive"]

    assert "supportsAllDrives=true" in notes
    assert "includeItemsFromAllDrives=true" in notes
    assert "corpora='allDrives'" in notes


def test_picker_guidance_lists_files_but_no_longer_the_drive_api_rules():
    text = harness._picked_files_guidance({
        "work": [{"id": "f1", "name": "Roadmap.gdoc", "resourceKey": "rk1"}],
    })

    assert "Roadmap.gdoc" in text
    assert "resourceKey: rk1" in text
    assert "corpora" not in text
    assert "supportsAllDrives" not in text


def test_no_picked_files_no_picker_guidance():
    assert harness._picked_files_guidance({}) == ""


def _drive_vault(tmp_path, **extra_fields):
    from anton.core.datasources.data_vault import LocalDataVault

    vault = LocalDataVault(tmp_path / "vault")
    vault.save("google_drive", "work", {"auth_type": "oauth", "access_token": "tok", **extra_fields})
    return vault


@asynccontextmanager
async def _session_config(monkeypatch, vault, *, vault_factory=None, **build_kwargs):
    from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
    from cowork.db.session import get_open_session
    from cowork.services.conversations import ConversationService

    saved = UserSettings(
        _env_file=None,
        planning_provider=Provider.OPENAI,
        coding_provider=Provider.OPENAI,
        router_provider=Provider.OPENAI,
        planning_model="p", coding_model="c", router_model="r",
        openai_api_key=SecretStr("test-only-key"),
        episodic_memory=False,
    )
    monkeypatch.setattr("cowork.common.settings.user_settings.get_user_settings", lambda: saved)
    # Capture the config instead of starting scratchpad processes.
    monkeypatch.setattr(harness, "build_chat_session", lambda config: config)
    monkeypatch.setattr(
        "anton.core.datasources.data_vault.LocalDataVault",
        vault_factory or (lambda *_a, **_k: vault),
    )
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "false")

    with get_open_session() as db:
        conversation = ConversationService(ScopedSession(db, LOCAL_SCOPE)).create_conversation(
            topic="usage-notes"
        )
        config, temp_vault_dir, _ = await harness.AntonHarness()._build_chat_session(
            conversation, **build_kwargs
        )
        try:
            yield config
        finally:
            if temp_vault_dir:
                shutil.rmtree(temp_vault_dir, ignore_errors=True)
            client = config.llm_client
            for provider in (client.planning_provider, client.coding_provider, client.router_provider):
                await provider._client.close()


async def test_usage_notes_reach_the_session_config(monkeypatch, tmp_path):
    vault = _drive_vault(tmp_path, _picked_files=json.dumps([{"id": "f1", "name": "Roadmap.gdoc"}]))
    async with _session_config(monkeypatch, vault) as config:
        assert config.connector_usage_notes == registry.usage_notes_for(["google_drive"])
        suffix = config.system_prompt_context.suffix
        assert "Roadmap.gdoc" in suffix  # the Picker block still renders
        assert "corpora" not in suffix   # ...without the moved Drive API rules


async def test_drive_notes_reach_the_session_config_without_picked_files(monkeypatch, tmp_path):
    async with _session_config(monkeypatch, _drive_vault(tmp_path)) as config:
        assert "corpora='allDrives'" in config.connector_usage_notes["google_drive"]
        assert "Google Picker" not in config.system_prompt_context.suffix


async def test_disabled_connection_gets_no_notes(monkeypatch, tmp_path):
    from anton.core.datasources.data_vault import LocalDataVault
    from cowork.common.settings.app_settings import get_app_settings

    source_vault = _drive_vault(tmp_path)
    source_dir = Path(get_app_settings().connector.vault_dir)

    # The harness reads the source vault, then copies the enabled connections
    # into a fresh temp vault; only the source path may map to the fake.
    def vault_for(path):
        return source_vault if Path(path) == source_dir else LocalDataVault(path)

    async with _session_config(
        monkeypatch,
        source_vault,
        vault_factory=vault_for,
        disabled_connections=[{"engine": "google_drive", "name": "work"}],
    ) as config:
        assert config.connector_usage_notes == {}
