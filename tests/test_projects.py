from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from cowork.models.project import Project
from cowork.services.projects import artifact_root_for_project, safe_project_path


def _settings(root_dir):
    return SimpleNamespace(project=SimpleNamespace(root_dir=str(root_dir)))


def test_safe_project_path_allows_registered_root_child(tmp_path):
    project_dir = tmp_path / "project-a"
    project_dir.mkdir()
    project = Project(name="project-a", path=str(project_dir), is_active=False)

    with patch("cowork.services.projects.get_app_settings", return_value=_settings(tmp_path)):
        assert safe_project_path(project) == project_dir.resolve()
        assert artifact_root_for_project(project) == project_dir.resolve() / ".anton" / "artifacts"


def test_safe_project_path_rejects_paths_outside_project_root(tmp_path):
    project = Project(name="outside", path=str(tmp_path.parent / "outside"), is_active=False)

    with patch("cowork.services.projects.get_app_settings", return_value=_settings(tmp_path)):
        assert safe_project_path(project) is None
        assert artifact_root_for_project(project) is None


def test_safe_project_path_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside-project"
    outside.mkdir(exist_ok=True)
    project_link = tmp_path / "linked-project"
    project_link.symlink_to(outside, target_is_directory=True)
    project = Project(name="linked-project", path=str(project_link), is_active=False)

    with patch("cowork.services.projects.get_app_settings", return_value=_settings(tmp_path)):
        assert safe_project_path(project) is None
