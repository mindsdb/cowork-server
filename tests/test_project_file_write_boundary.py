"""Filesystem-boundary checks for project-file writes."""

from __future__ import annotations

import os
from contextlib import contextmanager
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints import project_files
from cowork.db.scoped import LOCAL_SCOPE


def _write(base, relative: str, content: str, monkeypatch):
    @contextmanager
    def opened_inventory(_scoped):
        with project_files.pinned_dir(base, nofollow_base=True) as root:
            yield (("project", root),)

    monkeypatch.setattr(
        project_files, "_opened_project_directory_inventory", opened_inventory
    )
    return project_files.write_project_file(
        "project",
        project_files._validated_project_path(relative),
        project_files._FileWriteRequest(content=content),
        SimpleNamespace(scope=LOCAL_SCOPE),
    )


def test_write_sanitizes_each_component_at_the_filesystem_boundary(
    tmp_path, monkeypatch
):
    base = tmp_path / "project"
    base.mkdir()
    validated = project_files._validated_project_path("nested/notes.txt")
    real_basename = project_files.os.path.basename
    real_open = project_files._opened_pinned_descendant
    sanitized = []
    opened = []

    def record_basename(value):
        sanitized.append(value)
        return real_basename(value)

    @contextmanager
    def opened_inventory(_scoped):
        with project_files.pinned_dir(base, nofollow_base=True) as root:
            yield (("project", root),)

    @contextmanager
    def record_open(directory, parts, **kwargs):
        opened.append((directory.path, parts, kwargs))
        with real_open(directory, parts, **kwargs) as pinned:
            yield pinned

    monkeypatch.setattr(project_files.os.path, "basename", record_basename)
    monkeypatch.setattr(
        project_files, "_opened_project_directory_inventory", opened_inventory
    )
    monkeypatch.setattr(project_files, "_opened_pinned_descendant", record_open)

    result = project_files.write_project_file(
        "project",
        validated,
        project_files._FileWriteRequest(content="hello"),
        SimpleNamespace(scope=LOCAL_SCOPE),
    )

    assert sanitized == ["project", "nested", "notes.txt", "nested"]
    assert opened == [(base, ("nested",), {"create": True})]
    assert (base / "nested" / "notes.txt").read_text(encoding="utf-8") == "hello"
    assert result["path"] == "nested/notes.txt"


def test_write_refuses_a_directory_as_the_final_entry(tmp_path, monkeypatch):
    base = tmp_path / "project"
    (base / "nested").mkdir(parents=True)

    with pytest.raises(HTTPException) as error:
        _write(base, "nested", "no", monkeypatch)

    assert error.value.status_code == 400
    assert error.value.detail == "Path is a directory"


@pytest.mark.skipif(os.name == "nt", reason="Windows local mode has no O_NOFOLLOW")
def test_write_refuses_a_symlinked_parent(tmp_path, monkeypatch):
    base = tmp_path / "project"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    victim = outside / "notes.txt"
    victim.write_text("keep", encoding="utf-8")
    (base / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(HTTPException) as error:
        _write(base, "nested/notes.txt", "replace", monkeypatch)

    assert error.value.status_code == 404
    assert victim.read_text(encoding="utf-8") == "keep"


@pytest.mark.skipif(os.name == "nt", reason="Windows local mode has no O_NOFOLLOW")
def test_write_refuses_a_symlinked_project_root(tmp_path, monkeypatch):
    real_root = tmp_path / "real-project"
    real_root.mkdir()
    victim = real_root / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    linked_root = tmp_path / "linked-project"
    linked_root.symlink_to(real_root, target_is_directory=True)

    class Catalog:
        def __init__(self, _scoped):
            pass

        def list_projects(self):
            return [SimpleNamespace(name="project", path=str(linked_root))]

        def ensure_dir_exists(self, _project):
            pass

    monkeypatch.setattr(project_files, "ProjectService", Catalog)

    with pytest.raises(HTTPException) as error:
        project_files.write_project_file(
            "project",
            project_files._validated_project_path("victim.txt"),
            project_files._FileWriteRequest(content="replace"),
            SimpleNamespace(scope=LOCAL_SCOPE),
        )

    assert error.value.status_code == 404
    assert victim.read_text(encoding="utf-8") == "keep"


async def test_upload_writes_through_the_same_pinned_boundary(tmp_path, monkeypatch):
    base = tmp_path / "project"
    base.mkdir()

    @contextmanager
    def opened_inventory(_scoped):
        with project_files.pinned_dir(base, nofollow_base=True) as root:
            yield (("project", root),)

    monkeypatch.setattr(
        project_files, "_opened_project_directory_inventory", opened_inventory
    )
    upload = project_files.UploadFile(
        file=BytesIO(b"uploaded"), filename="notes.txt"
    )

    result = await project_files.upload_project_files(
        "project", SimpleNamespace(scope=LOCAL_SCOPE), [upload]
    )

    assert result == {
        "results": [{"name": "notes.txt", "ok": True, "size": 8}]
    }
    assert (base / "notes.txt").read_bytes() == b"uploaded"


@pytest.mark.skipif(os.name == "nt", reason="Windows local mode has no O_NOFOLLOW")
async def test_upload_refuses_a_symlinked_final_file(tmp_path, monkeypatch):
    base = tmp_path / "project"
    base.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    (base / "notes.txt").symlink_to(victim)

    @contextmanager
    def opened_inventory(_scoped):
        with project_files.pinned_dir(base, nofollow_base=True) as root:
            yield (("project", root),)

    monkeypatch.setattr(
        project_files, "_opened_project_directory_inventory", opened_inventory
    )
    upload = project_files.UploadFile(
        file=BytesIO(b"replace"), filename="notes.txt"
    )

    result = await project_files.upload_project_files(
        "project", SimpleNamespace(scope=LOCAL_SCOPE), [upload]
    )

    assert result == {
        "results": [
            {"name": "notes.txt", "ok": False, "error": "File write failed"}
        ]
    }
    assert victim.read_text(encoding="utf-8") == "keep"
