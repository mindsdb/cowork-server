"""Request project names select only server-opened project roots."""

from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from cowork.api.v1.endpoints import project_files
from cowork.db.scoped import LOCAL_SCOPE


def test_project_dir_queries_with_the_basename_sanitizer_result(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "reports"
    project_root.mkdir()
    seen = []

    class Catalog:
        def __init__(self, _scoped):
            pass

        def get_project_by_name(self, selected_name):
            seen.append(selected_name)
            return SimpleNamespace(path=str(project_root))

        def ensure_dir_exists(self, _project):
            pass

    monkeypatch.setattr(project_files, "ProjectService", Catalog)
    request_name = ("reports" + "x")[:-1]

    assert project_files._project_dir(request_name, object()) == project_root
    assert seen == ["reports"]


@pytest.mark.parametrize("request_name", ("../reports", r"..\reports", ".", ""))
def test_project_dir_rejects_non_basename_names_before_catalog_access(
    request_name, monkeypatch
):
    class CatalogReached(Exception):
        pass

    monkeypatch.setattr(
        project_files,
        "ProjectService",
        lambda _scoped: (_ for _ in ()).throw(CatalogReached),
    )

    with pytest.raises(HTTPException) as error:
        project_files._project_dir(request_name, object())

    assert error.value.status_code == 404


def test_delete_pins_complete_inventory_before_project_name_selection(
    tmp_path, monkeypatch
):
    reports_root = tmp_path / "reports"
    other_root = tmp_path / "other"
    reports_root.mkdir()
    other_root.mkdir()
    target = reports_root / "notes.txt"
    target.write_text("delete", encoding="utf-8")
    validated_path = project_files._validated_project_path("notes.txt")
    projects = [
        SimpleNamespace(name="reports", path=str(reports_root)),
        SimpleNamespace(name="other", path=str(other_root)),
    ]
    events = []
    opened = []
    closed = []

    class Catalog:
        def __init__(self, _scoped):
            events.append("catalog")

        def list_projects(self):
            events.append("list")
            return projects

        def ensure_dir_exists(self, project):
            events.append(("ensure", project.name))

    @contextmanager
    def record_pin(base, **_kwargs):
        events.append(("pin", base))
        opened.append(base)
        try:
            yield project_files.PinnedDir(None, base)
        finally:
            closed.append(base)

    original_basename = project_files.os.path.basename

    def inspect_request(value):
        assert events == [
            "catalog",
            "list",
            ("ensure", "reports"),
            ("pin", reports_root),
            ("ensure", "other"),
            ("pin", other_root),
        ]
        events.append("request")
        return original_basename(value)

    monkeypatch.setattr(project_files, "ProjectService", Catalog)
    monkeypatch.setattr(project_files, "pinned_dir", record_pin)
    monkeypatch.setattr(project_files.os.path, "basename", inspect_request)

    result = project_files.delete_project_file(
        "reports",
        validated_path,
        SimpleNamespace(scope=LOCAL_SCOPE),
    )

    assert result == {"status": "deleted", "path": "notes.txt"}
    assert opened == [reports_root, other_root]
    assert closed == [other_root, reports_root]
    assert not target.exists()


def test_delete_uses_the_server_opened_handle_selected_by_name(monkeypatch):
    server_path = project_files.Path("/server/catalog/reports")
    server_handle = project_files.PinnedDir(None, server_path)
    seen = []

    @contextmanager
    def opened_inventory(_scoped):
        yield (("reports", server_handle),)

    @contextmanager
    def inspect_entry(directory, parts):
        seen.append((directory, parts))
        yield directory, "notes.txt"

    class ReachedFilesystemCheck(Exception):
        pass

    def stop_at_filesystem_check(_directory, _name):
        raise ReachedFilesystemCheck

    monkeypatch.setattr(
        project_files,
        "_opened_project_directory_inventory",
        opened_inventory,
    )
    monkeypatch.setattr(
        project_files,
        "_opened_existing_project_entry",
        inspect_entry,
    )
    monkeypatch.setattr(project_files, "dir_lstat", stop_at_filesystem_check)

    with pytest.raises(ReachedFilesystemCheck):
        project_files.delete_project_file(
            ("reports" + "x")[:-1],
            project_files._validated_project_path("notes.txt"),
            SimpleNamespace(scope=LOCAL_SCOPE),
        )

    assert seen == [(server_handle, ("notes.txt",))]
    assert seen[0][0] is server_handle


@pytest.mark.skipif(os.name == "nt", reason="Windows local mode has no O_NOFOLLOW")
def test_delete_refuses_a_symlinked_project_root(tmp_path, monkeypatch):
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
        project_files.delete_project_file(
            "project",
            project_files._validated_project_path("victim.txt"),
            SimpleNamespace(scope=LOCAL_SCOPE),
        )

    assert error.value.status_code == 404
    assert victim.read_text(encoding="utf-8") == "keep"
