"""Filesystem-boundary checks for project-file writes."""

from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints import project_files
from cowork.db.scoped import LOCAL_SCOPE


def _write(base, relative: str, content: str, monkeypatch):
    monkeypatch.setattr(project_files, "_project_dir", lambda *_args: base)
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
    real_open = project_files.opened_subdir_nofollow
    sanitized = []
    opened = []

    def record_basename(value):
        sanitized.append(value)
        return real_basename(value)

    @contextmanager
    def record_open(directory, *parts, **kwargs):
        opened.append((directory, parts, kwargs))
        with real_open(directory, *parts, **kwargs) as pinned:
            yield pinned

    monkeypatch.setattr(project_files, "_project_dir", lambda *_args: base)
    monkeypatch.setattr(project_files.os.path, "basename", record_basename)
    monkeypatch.setattr(project_files, "opened_subdir_nofollow", record_open)

    result = project_files.write_project_file(
        "project",
        validated,
        project_files._FileWriteRequest(content="hello"),
        SimpleNamespace(scope=LOCAL_SCOPE),
    )

    assert sanitized == ["project", "nested", "notes.txt"]
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
