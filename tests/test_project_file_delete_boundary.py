"""Filesystem-boundary checks for project-file deletion."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints import project_files
from cowork.db.scoped import LOCAL_SCOPE


def _delete(base, relative: str, monkeypatch):
    class Catalog:
        def __init__(self, _scoped):
            pass

        def list_projects(self):
            return [SimpleNamespace(name="project", path=str(base))]

        def ensure_dir_exists(self, _project):
            pass

    monkeypatch.setattr(project_files, "ProjectService", Catalog)
    scoped = SimpleNamespace(scope=LOCAL_SCOPE)
    return project_files.delete_project_file(
        "project", project_files._validated_project_path(relative), scoped
    )


def test_delete_project_file_preserves_nested_paths(tmp_path, monkeypatch):
    base = tmp_path / "project"
    target = base / "nested" / ".anton" / "notes.md"
    target.parent.mkdir(parents=True)
    target.write_text("private")

    result = _delete(base, "nested/.anton/notes.md", monkeypatch)

    assert result == {"status": "deleted", "path": "nested/.anton/notes.md"}
    assert not target.exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows local mode has no O_NOFOLLOW")
def test_delete_project_file_refuses_a_symlinked_parent(tmp_path, monkeypatch):
    base = tmp_path / "project"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    victim = outside / "victim.txt"
    victim.write_text("keep")
    (base / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(HTTPException) as error:
        _delete(base, "nested/victim.txt", monkeypatch)

    assert error.value.status_code == 404
    assert victim.read_text() == "keep"


@pytest.mark.skipif(os.name == "nt", reason="Windows local mode has no O_NOFOLLOW")
def test_delete_project_file_refuses_a_symlinked_file(tmp_path, monkeypatch):
    base = tmp_path / "project"
    outside = tmp_path / "victim.txt"
    base.mkdir()
    outside.write_text("keep")
    linked = base / "linked.txt"
    linked.symlink_to(outside)

    with pytest.raises(HTTPException) as error:
        _delete(base, "linked.txt", monkeypatch)

    assert error.value.status_code == 404
    assert linked.is_symlink()
    assert outside.read_text() == "keep"
