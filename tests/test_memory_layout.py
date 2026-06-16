from pathlib import Path

import pytest

from cowork.harnesses.memory.adapter import BaseMemoryAdapter
from cowork.harnesses.memory.layout import MemoryLayout
from cowork.harnesses.memory.registry import MemorySlot, SLOT_REGISTRY
from cowork.harnesses.memory.store import SharedMemoryStore


class _TestAdapter(BaseMemoryAdapter):
    harness_id = "test"


@pytest.fixture
def memory_root(tmp_path):
    return tmp_path / "memory"


@pytest.fixture
def layout(memory_root):
    return MemoryLayout(memory_root=memory_root)


def test_ensure_canonical_files_creates_slot_files(layout, memory_root):
    layout.ensure_canonical_files()

    for slot in MemorySlot:
        assert (memory_root / SLOT_REGISTRY[slot].filename).is_file()


def test_ensure_canonical_files_is_idempotent(layout, memory_root):
    layout.ensure_canonical_files()
    rules = memory_root / "rules.md"
    rules.write_text("keep me\n", encoding="utf-8")

    layout.ensure_canonical_files()

    assert rules.read_text(encoding="utf-8") == "keep me\n"


def test_ensure_layout_creates_symlinks(layout, memory_root, tmp_path):
    link_dir = tmp_path / "hermes" / "memories"
    adapter = _TestAdapter()
    adapter.RUNTIME_SYMLINKS = {
        link_dir / "USER.md": MemorySlot.PROFILE,
        link_dir / "MEMORY.md": MemorySlot.LESSONS,
    }

    layout.ensure_layout([adapter])

    user_link = link_dir / "USER.md"
    memory_link = link_dir / "MEMORY.md"

    assert user_link.is_symlink()
    assert memory_link.is_symlink()
    assert user_link.resolve() == (memory_root / "profile.md").resolve()
    assert memory_link.resolve() == (memory_root / "lessons.md").resolve()


def test_ensure_layout_skips_when_real_file_blocks_symlink(layout, tmp_path):
    link_dir = tmp_path / "hermes" / "memories"
    link_dir.mkdir(parents=True)
    user_file = link_dir / "USER.md"
    user_file.write_text("real file\n", encoding="utf-8")

    adapter = _TestAdapter()
    adapter.RUNTIME_SYMLINKS = {user_file: MemorySlot.PROFILE}

    layout.ensure_layout([adapter])

    assert user_file.is_file()
    assert not user_file.is_symlink()
    assert user_file.read_text(encoding="utf-8") == "real file\n"


def test_symlink_round_trip_via_store(layout, memory_root, tmp_path):
    link_path = tmp_path / "hermes" / "memories" / "MEMORY.md"
    adapter = _TestAdapter()
    adapter.RUNTIME_SYMLINKS = {link_path: MemorySlot.LESSONS}

    layout.ensure_layout([adapter])

    store = SharedMemoryStore(root=memory_root)
    store.write(MemorySlot.LESSONS, "shared lesson")

    assert link_path.read_text(encoding="utf-8").strip() == "shared lesson"
