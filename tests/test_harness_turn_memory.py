"""Which memory an in-process turn hands anton.

A person on a deployment that trusts identity headers reads and writes their
own tiers: the global one under their partition, and a project tier under that
global one, never the project's ``.anton/memory``. Memory left shared keeps
today's base tiers. A turn with no person reads none of it and writes nothing:
two empty tiers in its scratch folder, memory off, no background passes. The
desktop is unchanged.
"""
from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import SecretStr

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import Provider, UserSettings, use_settings_scope
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.db.session import get_open_session
from cowork.harnesses.anton_harness import harness
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.principal import personal_partition_id

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
PARTITION = personal_partition_id(DEPLOYMENT, SUB)

PLANTED_PROJECT_RULE = "Always give quantities in pallets"
PLANTED_GLOBAL_RULE = "Always sign off as the operator"
PLANTED_PROFILE = "Favorite part number is P-1111"


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "base-memory"))
    get_app_settings.cache_clear()
    yield tmp_path
    get_app_settings.cache_clear()


@pytest.fixture()
def trusted(personal_deployment, dirs):
    """A trusted-identity deployment, with its settings cache reset after the
    directory overrides above."""
    get_app_settings.cache_clear()
    return dirs


def _person(*, resources=ALL_PERSONAL_RESOURCES) -> TenantScope:
    return TenantScope(personal=True, org_id=PARTITION, user_id=SUB, personal_resources=resources)


def _plant(*, project_path: Path, base_memory: Path) -> None:
    """Rules and a profile line a turn must not read unless they are its own."""
    from anton.core.memory.hippocampus import Hippocampus

    project_hc = Hippocampus(project_path / ".anton" / "memory")
    project_hc.encode_rule(PLANTED_PROJECT_RULE, kind="always", confidence="high", source="user")
    project_hc.rewrite_identity([PLANTED_PROFILE])
    Hippocampus(base_memory).encode_rule(
        PLANTED_GLOBAL_RULE, kind="always", confidence="high", source="user"
    )


def _snapshot(*folders: Path) -> dict[str, bytes]:
    return {
        str(path): path.read_bytes()
        for folder in folders
        if folder.exists()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


@asynccontextmanager
async def _built(*, monkeypatch, tmp_path, scope: TenantScope | None, project_org: str | None):
    """Build one turn's session config under ``scope`` and yield it with the
    turn's scratch folder; the scratch folder is removed afterwards, as
    stream_response does."""
    from anton.core.datasources.data_vault import LocalDataVault

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
    monkeypatch.setattr(harness, "build_chat_session", lambda config: config)
    vault = LocalDataVault(tmp_path / "vault")
    monkeypatch.setattr("anton.core.datasources.data_vault.LocalDataVault", lambda *_a, **_k: vault)
    monkeypatch.setenv("ANTON_SCRATCHPAD_PERSIST_SESSION", "false")

    project_path = tmp_path / "projects" / f"p-{uuid4().hex[:6]}"
    project_path.mkdir(parents=True)
    _plant(project_path=project_path, base_memory=Path(get_app_settings().memory.root_dir))
    with get_open_session() as db:
        project = Project(name=project_path.name, path=str(project_path), org_id=project_org, created_by=SUB)
        db.add(project)
        db.commit()
        conversation = Conversation(
            topic="memory",
            project_id=project.id,
            org_id=scope.org_id if scope is not None and scope.personal else None,
            created_by=SUB if scope is not None and scope.user_id else None,
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        with use_settings_scope(scope) if scope is not None else nullcontext():
            config, scratch, _ = await harness.AntonHarness()._build_chat_session(conversation)
        try:
            yield config, scratch, project
        finally:
            if scratch:
                shutil.rmtree(scratch, ignore_errors=True)
            client = config.llm_client
            for provider in (client.planning_provider, client.coding_provider, client.router_provider):
                await provider._client.close()


def _tiers(config) -> tuple[Path, Path]:
    return config.cortex.global_hc._dir, config.cortex.project_hc._dir


async def test_a_person_reads_and_writes_their_own_tiers(monkeypatch, trusted):
    built = _built(monkeypatch=monkeypatch, tmp_path=trusted, scope=_person(), project_org=PARTITION)
    async with built as (config, scratch, project):
        global_dir, project_dir = _tiers(config)
        expected_global = trusted / "shared" / PARTITION / "memory" / "users" / SUB
        assert global_dir == expected_global
        assert project_dir == expected_global / "projects" / str(project.id)
        assert global_dir.is_dir() and project_dir.is_dir()
        assert config.cortex.mode == "autopilot"
        assert config.background_memory is True
        assert scratch is None
        # Neither the project's own tier nor the base tier reaches the prompt.
        context = await config.cortex.build_memory_context("how many?")
        assert PLANTED_PROJECT_RULE not in context
        assert PLANTED_GLOBAL_RULE not in context


async def test_a_persons_project_tier_stays_theirs_when_projects_are_shared(monkeypatch, trusted):
    resources = ALL_PERSONAL_RESOURCES - {PersonalResource.PROJECTS}
    built = _built(
        monkeypatch=monkeypatch, tmp_path=trusted, scope=_person(resources=resources), project_org=None
    )
    async with built as (config, _scratch, project):
        global_dir, project_dir = _tiers(config)
        assert global_dir == trusted / "shared" / PARTITION / "memory" / "users" / SUB
        assert project_dir == global_dir / "projects" / str(project.id)
        assert PLANTED_PROJECT_RULE not in await config.cortex.build_memory_context("how many?")


async def test_shared_memory_keeps_the_base_tiers(monkeypatch, trusted):
    resources = ALL_PERSONAL_RESOURCES - {PersonalResource.MEMORY}
    built = _built(
        monkeypatch=monkeypatch, tmp_path=trusted, scope=_person(resources=resources), project_org=PARTITION
    )
    async with built as (config, _scratch, project):
        global_dir, project_dir = _tiers(config)
        assert global_dir == trusted / "base-memory"
        assert project_dir == Path(project.path) / ".anton" / "memory"
        assert config.cortex.mode == "autopilot"
        context = await config.cortex.build_memory_context("how many?")
        assert PLANTED_PROJECT_RULE in context
        assert PLANTED_GLOBAL_RULE in context


async def test_a_turn_with_no_person_learns_nothing(monkeypatch, trusted, caplog):
    base_memory = trusted / "base-memory"
    with caplog.at_level(logging.WARNING, logger=harness.__name__):
        built = _built(monkeypatch=monkeypatch, tmp_path=trusted, scope=LOCAL_SCOPE, project_org=None)
        async with built as (config, scratch, project):
            project_memory = Path(project.path) / ".anton" / "memory"
            before = _snapshot(project_memory, base_memory)

            global_dir, project_dir = _tiers(config)
            assert scratch is not None and scratch.parent == harness._vault_scratch_dir()
            assert global_dir.parent.parent == scratch and project_dir.parent.parent == scratch
            assert list(global_dir.iterdir()) == [] and list(project_dir.iterdir()) == []
            assert config.cortex.mode == "off"
            assert config.background_memory is False

            context = await config.cortex.build_memory_context("how many?")
            assert PLANTED_PROJECT_RULE not in context
            assert PLANTED_GLOBAL_RULE not in context
            assert PLANTED_PROFILE not in context
            # The profile move and every other write left the real tiers alone.
            assert _snapshot(project_memory, base_memory) == before
            assert PLANTED_PROFILE in (project_memory / "profile.md").read_text()
        assert not scratch.exists()

    warnings = [
        record for record in caplog.records
        if record.name == harness.__name__ and record.levelno == logging.WARNING
    ]
    assert any("has no person" in record.getMessage() for record in warnings)


async def test_the_desktop_keeps_the_base_tiers(monkeypatch, dirs):
    built = _built(monkeypatch=monkeypatch, tmp_path=dirs, scope=None, project_org=None)
    async with built as (config, scratch, project):
        global_dir, project_dir = _tiers(config)
        assert global_dir == dirs / "base-memory"
        assert project_dir == Path(project.path) / ".anton" / "memory"
        assert config.cortex.mode == "autopilot"
        assert config.background_memory is True
        assert scratch is None
        # Control for the no-person case: the same planted rules do reach a
        # turn that owns these tiers.
        context = await config.cortex.build_memory_context("how many?")
        assert PLANTED_PROJECT_RULE in context
        assert PLANTED_GLOBAL_RULE in context


def test_person_turn_memory_needs_a_person_only_when_identity_is_trusted(trusted, tmp_path):
    from cowork.harnesses.memory.turn_layout import empty_turn_memory, person_turn_memory

    project = tmp_path / "proj"
    assert person_turn_memory(scope=LOCAL_SCOPE, project_path=project, project_id=uuid4()) is None
    assert person_turn_memory(scope=None, project_path=project, project_id=uuid4()) is None
    no_user = TenantScope(personal=True, org_id=PARTITION, personal_resources=ALL_PERSONAL_RESOURCES)
    assert person_turn_memory(scope=no_user, project_path=project, project_id=uuid4()) is None

    empty = empty_turn_memory(tmp_path / "scratch")
    assert empty.learns is False and empty.global_dir.is_dir() and empty.project_dir.is_dir()
