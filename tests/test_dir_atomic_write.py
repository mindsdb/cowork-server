"""One helper replaces a file in a pinned folder, durably and without
following a link.

`dir_atomic_write` writes the new bytes beside the name, syncs them, renames
them over the name and syncs the folder. The artifact journal, artifact
metadata and the source's folder sync all go through it or its `dir_fsync`.
"""
from __future__ import annotations

import json
import os
import stat
import uuid
from pathlib import Path

import pytest

from cowork.common import paths
from cowork.common.paths import dir_atomic_write, pinned_dir
from cowork.services import artifact_identity, artifact_journal, artifact_revisions
from cowork.services.artifact_identity import ensure_full_id
from cowork.services.artifact_revisions import current_source, save_source

POSIX = pytest.mark.skipif(os.name == "nt", reason="Windows has no dir_fd and keeps no permission bits")


def _leftovers(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir() if path.name.endswith(".tmp"))


def test_a_write_lands_private_and_leaves_no_temporary_file(tmp_path):
    with pinned_dir(tmp_path) as folder:
        dir_atomic_write(folder, "record.json", content=b"{}\n")

    assert (tmp_path / "record.json").read_bytes() == b"{}\n"
    assert _leftovers(tmp_path) == []


@POSIX
def test_a_write_takes_the_mode_it_is_given_and_is_private_without_one(tmp_path):
    with pinned_dir(tmp_path) as folder:
        dir_atomic_write(folder, "shared.json", content=b"{}", mode=0o644)
        dir_atomic_write(folder, "private.json", content=b"{}")

    assert stat.S_IMODE((tmp_path / "shared.json").stat().st_mode) == 0o644
    assert stat.S_IMODE((tmp_path / "private.json").stat().st_mode) == 0o600


@POSIX
def test_a_link_at_the_name_is_replaced_and_its_target_is_untouched(tmp_path):
    target = tmp_path / "elsewhere.txt"
    target.write_text("someone else's", encoding="utf-8")
    (tmp_path / "record.json").symlink_to(target)

    with pinned_dir(tmp_path) as folder:
        dir_atomic_write(folder, "record.json", content=b"ours")

    assert not (tmp_path / "record.json").is_symlink()
    assert (tmp_path / "record.json").read_bytes() == b"ours"
    assert target.read_text(encoding="utf-8") == "someone else's"


@POSIX
def test_a_second_name_at_the_name_is_replaced_and_the_first_keeps_its_bytes(tmp_path):
    first = tmp_path / "first.txt"
    first.write_text("before", encoding="utf-8")
    os.link(first, tmp_path / "record.json")

    with pinned_dir(tmp_path) as folder:
        dir_atomic_write(folder, "record.json", content=b"after")

    assert (tmp_path / "record.json").read_bytes() == b"after"
    assert first.read_text(encoding="utf-8") == "before"
    assert first.stat().st_nlink == 1


@POSIX
@pytest.mark.parametrize("plant", [os.symlink, os.link], ids=["symbolic-link", "hard-link"])
def test_a_link_planted_at_the_temporary_name_is_never_written_through(tmp_path, monkeypatch, plant):
    target = tmp_path / "outside" / "secret.txt"
    target.parent.mkdir()
    target.write_text("someone else's", encoding="utf-8")
    folder_path = tmp_path / "folder"
    folder_path.mkdir()
    (folder_path / "record.json").write_bytes(b"old")
    fixed = uuid.UUID(int=0)
    monkeypatch.setattr(paths.uuid, "uuid4", lambda: fixed)
    plant(target, folder_path / f".record.json.{fixed.hex}.tmp")

    with pinned_dir(folder_path) as folder, pytest.raises(OSError):
        dir_atomic_write(folder, "record.json", content=b"new")

    assert target.read_text(encoding="utf-8") == "someone else's"
    assert (folder_path / "record.json").read_bytes() == b"old"


@POSIX
def test_the_file_and_then_its_folder_are_synced(tmp_path, monkeypatch):
    synced: list[str] = []
    real_fsync = os.fsync

    def fsync(fd):
        synced.append("folder" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        real_fsync(fd)

    monkeypatch.setattr(paths.os, "fsync", fsync)
    with pinned_dir(tmp_path) as folder:
        dir_atomic_write(folder, "record.json", content=b"{}")

    assert synced == ["file", "folder"]


def test_a_write_that_fails_leaves_the_old_file_and_no_temporary_file(tmp_path, monkeypatch):
    (tmp_path / "record.json").write_bytes(b"old")

    def fail(_fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(paths.os, "fsync", fail)
    with pinned_dir(tmp_path) as folder, pytest.raises(OSError, match="No space left"):
        dir_atomic_write(folder, "record.json", content=b"new")

    assert (tmp_path / "record.json").read_bytes() == b"old"
    assert _leftovers(tmp_path) == []


def test_a_folder_that_cannot_be_synced_still_takes_the_write(tmp_path, monkeypatch):
    def fail(_folder):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(paths, "dir_fsync", fail)
    with pinned_dir(tmp_path) as folder:
        dir_atomic_write(folder, "record.json", content=b"new")

    assert (tmp_path / "record.json").read_bytes() == b"new"


@pytest.mark.parametrize("name", ["", ".", "..", "../escaped.json", "a/b.json", "a\\b.json", "a\0b"])
def test_a_name_with_more_than_one_part_is_refused_before_any_open(tmp_path, monkeypatch, name):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("opened a file")

    monkeypatch.setattr(paths, "dir_open", unexpected)
    with pinned_dir(tmp_path) as folder, pytest.raises(ValueError):
        dir_atomic_write(folder, name, content=b"{}")


# ── its callers ──────────────────────────────────────────────────────────────


@pytest.fixture
def artifact(tmp_path) -> tuple[Path, dict, str]:
    folder = tmp_path / "alice" / ".anton" / "artifacts" / "report"
    folder.mkdir(parents=True)
    metadata = {"id": "a1b2c3d4", "slug": "report", "name": "Report", "type": "document", "primary": "brief.md"}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (folder / "brief.md").write_text("# First\n", encoding="utf-8")
    return folder, metadata, "brief.md"


def _spy(monkeypatch, module, name: str) -> list[str]:
    calls: list[str] = []
    real = getattr(module, name)

    def spy(folder, child, *args, **kwargs):
        calls.append(child)
        return real(folder, child, *args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return calls


def test_the_journal_and_the_metadata_write_through_the_helper(artifact, monkeypatch):
    folder, metadata, _ = artifact
    journal_writes = _spy(monkeypatch, artifact_journal, "dir_atomic_write")
    metadata_writes = _spy(monkeypatch, artifact_identity, "dir_atomic_write")

    artifact_id, metadata = ensure_full_id(folder, metadata)
    current_source(folder, metadata, artifact_id)

    assert metadata_writes == ["metadata.json"]
    assert "manifest.json" in journal_writes


def test_a_source_save_syncs_the_sources_folder_through_dir_fsync(artifact, monkeypatch):
    folder, metadata, _ = artifact
    artifact_id, metadata = ensure_full_id(folder, metadata)
    head = current_source(folder, metadata, artifact_id)["revision"]
    synced: list[Path] = []
    real = artifact_revisions.dir_fsync

    def spy(pinned):
        synced.append(pinned.path)
        real(pinned)

    monkeypatch.setattr(artifact_revisions, "dir_fsync", spy)
    save_source(folder, metadata, artifact_id, content="# Second\n", expected_revision_id=head["id"])

    assert synced == [folder]
