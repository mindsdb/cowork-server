"""The managed folder's memory in every in-process turn.

Every turn hands anton's Cortex the managed ``memory/`` as its read-only layer
(``managed_hc``): a person's turn, a turn with no person and a desktop turn.
The turn's learned tiers are never that folder, so nothing the turn learns
lands there. The managed_dir fixture writes a real Hippocampus folder, and
each turn here builds the installed anton's Cortex, so these tests check the
layer that Cortex is handed and the rule and lesson it reads from it. How
anton renders the layer into the prompt is anton's, and its own tests cover
it. With no managed folder, no layer is passed.
"""
from __future__ import annotations

import stat
from pathlib import Path

import pytest

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.principal import personal_partition_id
from tests._fakes import MANAGED_LESSON, MANAGED_RULE
from tests._turns import built_turn

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
PARTITION = personal_partition_id(DEPLOYMENT, SUB)
PERSON = TenantScope(personal=True, org_id=PARTITION, user_id=SUB, personal_resources=ALL_PERSONAL_RESOURCES)


@pytest.fixture
def stores(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "base-memory"))
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "base-vault"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "base-skills"))
    get_app_settings.cache_clear()
    yield tmp_path
    get_app_settings.cache_clear()


@pytest.fixture
def read_only_managed_memory(managed_dir):
    """The managed memory as the image ships it: root-owned files at 0444 in a
    folder nobody else writes. Here: no write bit anywhere, so a write raises."""
    memory = managed_dir.memory_dir
    for path in memory.iterdir():
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    memory.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    yield managed_dir
    memory.chmod(stat.S_IRWXU)
    for path in memory.iterdir():
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _snapshot(folder: Path) -> dict[str, bytes]:
    return {str(path): path.read_bytes() for path in sorted(folder.rglob("*")) if path.is_file()}


def _assert_managed_layer(cortex, managed_dir) -> None:
    layer = cortex.managed_hc
    assert layer is not None
    assert layer._dir == managed_dir.memory_dir
    assert MANAGED_RULE in layer.recall_rules()
    assert MANAGED_LESSON in layer.recall_lessons()
    assert managed_dir.memory_dir not in (cortex.global_hc._dir, cortex.project_hc._dir)


async def test_a_persons_turn_reads_the_managed_layer_and_never_writes_it(
    monkeypatch, stores, personal_deployment, read_only_managed_memory
):
    from anton.core.memory.base import Engram

    managed_dir = read_only_managed_memory
    get_app_settings.cache_clear()
    before = _snapshot(managed_dir.memory_dir)

    async with built_turn(monkeypatch=monkeypatch, tmp_path=stores, scope=PERSON) as turn:
        cortex = turn.config.cortex
        _assert_managed_layer(cortex, managed_dir)
        assert cortex.global_hc._dir == stores / "shared" / PARTITION / "memory" / "users" / SUB
        assert cortex.mode == "autopilot"

        # What a turn learns lands in the person's tiers, all three kinds.
        await cortex.encode(
            [
                Engram(text="Favorite part number is P-1111", kind="profile", scope="global"),
                Engram(text="Answer in pallets", kind="always", scope="global", confidence="high", source="user"),
                Engram(text="The plant ships on Fridays", kind="lesson", scope="project", source="llm"),
            ]
        )
        assert "P-1111" in cortex.global_hc.recall_identities()
        assert "Answer in pallets" in cortex.global_hc.recall_rules()
        assert "ships on Fridays" in cortex.project_hc.recall_lessons()

    assert _snapshot(managed_dir.memory_dir) == before


async def test_a_turn_with_no_person_still_gets_the_managed_rules(
    monkeypatch, stores, personal_deployment, read_only_managed_memory
):
    managed_dir = read_only_managed_memory
    get_app_settings.cache_clear()
    before = _snapshot(managed_dir.memory_dir)

    async with built_turn(monkeypatch=monkeypatch, tmp_path=stores, scope=LOCAL_SCOPE) as turn:
        cortex = turn.config.cortex
        _assert_managed_layer(cortex, managed_dir)
        # Its learned tiers are the empty scratch ones, and it learns nothing.
        assert cortex.global_hc._dir.parent.parent == turn.scratch
        assert cortex.mode == "off"
        assert turn.config.background_memory is False

    assert _snapshot(managed_dir.memory_dir) == before


async def test_a_desktop_turn_gets_the_managed_layer_too(monkeypatch, stores, managed_dir):
    async with built_turn(monkeypatch=monkeypatch, tmp_path=stores, scope=None) as turn:
        cortex = turn.config.cortex
        _assert_managed_layer(cortex, managed_dir)
        assert cortex.global_hc._dir == stores / "base-memory"


async def test_no_managed_folder_passes_no_layer(monkeypatch, stores, managed_dir):
    monkeypatch.delenv("COWORK_MANAGED_DIR")
    get_app_settings.cache_clear()

    async with built_turn(monkeypatch=monkeypatch, tmp_path=stores, scope=None) as turn:
        assert turn.config.cortex.managed_hc is None
