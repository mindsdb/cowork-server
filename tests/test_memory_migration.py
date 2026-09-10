from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel, create_engine

from cowork.common.settings.app_settings import AppSettings, MemorySettings
from cowork.harnesses.memory.migration import migrate_harness_memory_to_shared, retire_hermes_memory
from cowork.harnesses.memory.registry import MemorySlot
from cowork.harnesses.memory.store import GlobalMemoryStore
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

    store = GlobalMemoryStore(root=memory_root)
    assert migrate_harness_memory_to_shared(db_session) is True

    assert store.read(MemorySlot.RULES).strip() == "Always use TypeScript"
    assert store.read(MemorySlot.PROFILE).strip() == "User prefers dark mode"
    assert store.read(MemorySlot.LESSONS).strip() == "Lesson one"
    # Sources are never deleted.
    assert (hermes_dir / "USER.md").is_file()
    assert (hermes_dir / "MEMORY.md").is_file()
    assert migrate_harness_memory_to_shared(db_session) is False


def test_migration_skips_when_canonical_slot_already_has_content(
    db_session, memory_root, tmp_path, monkeypatch
):
    store = GlobalMemoryStore(root=memory_root)
    store.write(MemorySlot.RULES, "existing rules")

    anton_dir = tmp_path / "anton" / "memory"
    anton_dir.mkdir(parents=True)
    (anton_dir / "rules.md").write_text("legacy rules\n", encoding="utf-8")

    monkeypatch.setattr(
        "cowork.harnesses.memory.migration._MIGRATION_SOURCES",
        [(anton_dir / "rules.md", MemorySlot.RULES)],
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

    store = GlobalMemoryStore(root=memory_root)
    assert migrate_harness_memory_to_shared(db_session) is True

    assert store.read(MemorySlot.PROFILE).strip() == "Anton profile note\n\nHermes user prefs"
    assert store.read(MemorySlot.LESSONS).strip() == "Anton lesson\n\nHermes lesson"


def _hermes_files(monkeypatch, hermes_dir):
    monkeypatch.setattr(
        "cowork.harnesses.memory.migration._HERMES_MEMORY_FILES",
        [
            (hermes_dir / "USER.md", MemorySlot.PROFILE),
            (hermes_dir / "MEMORY.md", MemorySlot.LESSONS),
        ],
    )


def test_retire_merges_divergent_real_copies_without_touching_them(
    db_session, memory_root, tmp_path, monkeypatch
):
    # Windows without symlink permission: the layout step copied the canonical
    # file, Hermes then appended to the copy. Only the new paragraphs come over.
    store = GlobalMemoryStore(root=memory_root)
    store.write(MemorySlot.PROFILE, "canonical profile")
    hermes_dir = tmp_path / "hermes" / "memories"
    hermes_dir.mkdir(parents=True)
    user_text = "canonical profile\n\nHermes-only note\n"
    (hermes_dir / "USER.md").write_text(user_text, encoding="utf-8")
    (hermes_dir / "MEMORY.md").write_text("Hermes lesson\n", encoding="utf-8")
    _hermes_files(monkeypatch, hermes_dir)

    assert retire_hermes_memory(db_session) is True

    assert store.read(MemorySlot.PROFILE).strip() == "canonical profile\n\nHermes-only note"
    assert store.read(MemorySlot.LESSONS).strip() == "Hermes lesson"
    assert (hermes_dir / "USER.md").read_text(encoding="utf-8") == user_text
    assert retire_hermes_memory(db_session) is False


def test_retire_skips_symlinks_and_identical_copies(db_session, memory_root, tmp_path, monkeypatch):
    store = GlobalMemoryStore(root=memory_root)
    store.write(MemorySlot.PROFILE, "profile")
    store.write(MemorySlot.LESSONS, "same lesson")
    hermes_dir = tmp_path / "hermes" / "memories"
    hermes_dir.mkdir(parents=True)
    (hermes_dir / "USER.md").symlink_to(memory_root / "profile.md")
    (hermes_dir / "MEMORY.md").write_text("same lesson\n", encoding="utf-8")
    _hermes_files(monkeypatch, hermes_dir)

    assert retire_hermes_memory(db_session) is True

    assert store.read(MemorySlot.PROFILE).strip() == "profile"
    assert store.read(MemorySlot.LESSONS).strip() == "same lesson"
    assert (hermes_dir / "USER.md").is_symlink()
