from pathlib import Path

from cowork.harnesses.hermes_harness.memory_adapter import HermesMemoryAdapter
from cowork.harnesses.memory.adapter import get_memory_adapter
from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.runtime import ensure_all_layouts


def test_get_memory_adapter_returns_registered_hermes():
    adapter = get_memory_adapter("hermes")
    assert adapter is not None
    assert isinstance(adapter, HermesMemoryAdapter)


def test_build_prompt_context_includes_rules(tmp_path, monkeypatch):
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    (memory_root / "rules.md").write_text("Always be concise\n", encoding="utf-8")

    monkeypatch.setattr(
        "cowork.harnesses.memory.store.get_app_settings",
        lambda: type(
            "S",
            (),
            {"memory": type("M", (), {"root_dir": str(memory_root)})()},
        )(),
    )

    context = HermesMemoryAdapter().build_prompt_context(Path("/tmp/test_project"))
    assert "Always be concise" in context


def test_ensure_all_layouts_creates_symlinks(tmp_path, monkeypatch):
    memory_root = tmp_path / "memory"
    hermes_memories = tmp_path / "hermes" / "memories"
    hermes_memories.mkdir(parents=True)

    monkeypatch.setattr(
        "cowork.harnesses.memory.store.get_app_settings",
        lambda: type(
            "S",
            (),
            {"memory": type("M", (), {"root_dir": str(memory_root)})()},
        )(),
    )
    monkeypatch.setattr(
        "cowork.harnesses.memory.layout.get_app_settings",
        lambda: type(
            "S",
            (),
            {"memory": type("M", (), {"root_dir": str(memory_root)})()},
        )(),
    )
    monkeypatch.setattr(
        "cowork.harnesses.hermes_harness.memory_adapter.memory_dir",
        hermes_memories,
    )
    monkeypatch.setattr(
        HermesMemoryAdapter,
        "RUNTIME_SYMLINKS",
        {
            hermes_memories / "USER.md": MemorySlot.PROFILE,
            hermes_memories / "MEMORY.md": MemorySlot.LESSONS,
        },
    )

    ensure_all_layouts()
    assert (hermes_memories / "USER.md").is_symlink()
    assert (hermes_memories / "MEMORY.md").is_symlink()
