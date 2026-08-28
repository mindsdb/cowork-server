"""Request project names are sanitized before scoped catalog lookup."""

from __future__ import annotations

import os
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


@pytest.mark.skipif(os.name == "nt", reason="Windows local mode has no O_NOFOLLOW")
def test_delete_refuses_a_symlinked_project_root(tmp_path, monkeypatch):
    real_root = tmp_path / "real-project"
    real_root.mkdir()
    victim = real_root / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    linked_root = tmp_path / "linked-project"
    linked_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr(project_files, "_project_dir", lambda *_args: linked_root)

    with pytest.raises(HTTPException) as error:
        project_files.delete_project_file(
            "project",
            project_files._validated_project_path("victim.txt"),
            SimpleNamespace(scope=LOCAL_SCOPE),
        )

    assert error.value.status_code == 404
    assert victim.read_text(encoding="utf-8") == "keep"
