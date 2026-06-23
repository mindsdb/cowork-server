from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel, create_engine

import cowork.harnesses.hermes_harness.memory_adapter  # noqa: F401
from cowork.common.settings.app_settings import AppSettings, MemorySettings
from cowork.harnesses.memory.migration import migrate_harness_memory_to_shared
from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import SharedMemoryStore
from cowork.models.setting import Setting


@pytest.fixture
def memory_root(tmp_path):
    return tmp_path / "memory"


@pytest.fixture
def db_session(tmp_path, memory_root, monkeypatch):
    def _settings() -> AppSettings:
        return AppSettings(memory=MemorySettings(root_dir=str(memory_root)))

    monkeypatch.setattr("cowork.harnesses.memory.store.get_app_settings", _settings)

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_migration_copies_legacy_files(db_session, memory_root, tmp_path, monkeypatch):
    anton_dir = tmp_path / "anton" / "memory"
    anton_dir.mkdir(parents=True)
    (anton_dir / "rules.md").write_text("Always use TypeScript\n", encoding="utf-8")

    hermes_dir = tmp_path / "hermes" / "memories"
    hermes_dir.mkdir(parents=True)
    (hermes_dir / "USER.md").write_text("User prefers dark mode\n", encoding="utf-8")
    (hermes_dir / "MEMORY.md").write_text("Lesson one\n", encoding="utf-8")

    monkeypatch.setattr(
        "cowork.harnesses.memory.migration._MIGRATION_SOURCES",
        [
            (anton_dir / "rules.md", MemorySlot.RULES),
            (hermes_dir / "USER.md", MemorySlot.PROFILE),
            (hermes_dir / "MEMORY.md", MemorySlot.LESSONS),
        ],
    )
    monkeypatch.setattr(
        "cowork.harnesses.hermes_harness.memory_adapter.HermesMemoryAdapter.RUNTIME_SYMLINKS",
        {
            hermes_dir / "USER.md": MemorySlot.PROFILE,
            hermes_dir / "MEMORY.md": MemorySlot.LESSONS,
        },
    )

    store = SharedMemoryStore(root=memory_root)
    assert migrate_harness_memory_to_shared(db_session) is True

    assert store.read(MemorySlot.RULES).strip() == "Always use TypeScript"
    assert store.read(MemorySlot.PROFILE).strip() == "User prefers dark mode"
    assert store.read(MemorySlot.LESSONS).strip() == "Lesson one"
    assert not (hermes_dir / "USER.md").exists()
    assert not (hermes_dir / "MEMORY.md").exists()
    assert migrate_harness_memory_to_shared(db_session) is False


def test_migration_skips_when_canonical_slot_already_has_content(
    db_session, memory_root, tmp_path, monkeypatch
):
    store = SharedMemoryStore(root=memory_root)
    store.write(MemorySlot.RULES, "existing rules")

    anton_dir = tmp_path / "anton" / "memory"
    anton_dir.mkdir(parents=True)
    (anton_dir / "rules.md").write_text("legacy rules\n", encoding="utf-8")

    monkeypatch.setattr(
        "cowork.harnesses.memory.migration._MIGRATION_SOURCES",
        [(anton_dir / "rules.md", MemorySlot.RULES)],
    )
    monkeypatch.setattr(
        "cowork.harnesses.hermes_harness.memory_adapter.HermesMemoryAdapter.RUNTIME_SYMLINKS",
        {},
    )

    assert migrate_harness_memory_to_shared(db_session) is True
    assert store.read(MemorySlot.RULES).strip() == "existing rules"


def test_migration_combines_multiple_sources_for_same_slot(
    db_session, memory_root, tmp_path, monkeypatch
):
    anton_dir = tmp_path / "anton" / "memory"
    anton_dir.mkdir(parents=True)
    (anton_dir / "profile.md").write_text("Anton profile note\n", encoding="utf-8")
    (anton_dir / "lessons.md").write_text("Anton lesson\n", encoding="utf-8")

    hermes_dir = tmp_path / "hermes" / "memories"
    hermes_dir.mkdir(parents=True)
    (hermes_dir / "USER.md").write_text("Hermes user prefs\n", encoding="utf-8")
    (hermes_dir / "MEMORY.md").write_text("Hermes lesson\n", encoding="utf-8")

    monkeypatch.setattr(
        "cowork.harnesses.memory.migration._MIGRATION_SOURCES",
        [
            (anton_dir / "profile.md", MemorySlot.PROFILE),
            (hermes_dir / "USER.md", MemorySlot.PROFILE),
            (anton_dir / "lessons.md", MemorySlot.LESSONS),
            (hermes_dir / "MEMORY.md", MemorySlot.LESSONS),
        ],
    )
    monkeypatch.setattr(
        "cowork.harnesses.hermes_harness.memory_adapter.HermesMemoryAdapter.RUNTIME_SYMLINKS",
        {
            hermes_dir / "USER.md": MemorySlot.PROFILE,
            hermes_dir / "MEMORY.md": MemorySlot.LESSONS,
        },
    )

    store = SharedMemoryStore(root=memory_root)
    assert migrate_harness_memory_to_shared(db_session) is True

    assert store.read(MemorySlot.PROFILE).strip() == "Anton profile note\n\nHermes user prefs"
    assert store.read(MemorySlot.LESSONS).strip() == "Anton lesson\n\nHermes lesson"
