"""Regression test for artifact serve URL prefix.

The serve_url_for() helper must produce URLs that match the router
mount point. This has regressed twice already (the prefix was /v1/
when the router uses /api/v1/), so we pin it with a test.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cowork.api.v1.router import api_router
from cowork.db.scoped import LOCAL_SCOPE
from cowork.services.artifacts import _project_artifacts_base, serve_url_for


def _get_artifacts_prefix() -> str:
    """Derive the full URL prefix for artifact serving from the router.

    FastAPI flattens sub-router paths when they're included, so
    route.path already contains the full prefix (e.g.
    "/api/v1/artifacts/serve/{project_name}/{file_path:path}").
    """
    for route in api_router.routes:
        if hasattr(route, "path") and "/serve/" in route.path:
            serve_idx = route.path.index("/serve/")
            return route.path[: serve_idx + len("/serve/")]
    raise AssertionError("No /serve/ route found on api_router")


def test_serve_url_prefix_matches_router():
    """serve_url_for() URLs must start with the router's artifact serve prefix."""
    expected_prefix = _get_artifacts_prefix()

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir) / "my-project"
        artifacts = project_dir / ".anton" / "artifacts"
        artifacts.mkdir(parents=True)
        test_file = artifacts / "index.html"
        test_file.write_text("<h1>hi</h1>")

        with patch(
            "cowork.services.artifacts._registered_project_dirs",
            return_value=[project_dir],
        ):
            url = serve_url_for(str(test_file), scope=LOCAL_SCOPE)

        assert url, "serve_url_for returned empty string"
        assert url.startswith(expected_prefix), (
            f"URL prefix mismatch: got {url!r}, expected to start with {expected_prefix!r}"
        )


def test_project_base_accepts_registered_path_through_symlink(tmp_path):
    real_root = tmp_path / "real-projects"
    artifacts = real_root / "proj" / ".anton" / "artifacts"
    artifacts.mkdir(parents=True)
    alias_root = tmp_path / "projects-alias"
    alias_root.symlink_to(real_root, target_is_directory=True)

    # A desktop session; no project here is an adopted folder.
    session = SimpleNamespace(scope=LOCAL_SCOPE)
    with patch(
        "cowork.services.artifacts._registered_project_dirs",
        return_value=[alias_root / "proj"],
    ), patch(
        "cowork.services.artifacts._projects_root",
        return_value=alias_root,
    ), patch(
        "cowork.services.artifacts._external_project_dir",
        return_value=None,
    ):
        assert _project_artifacts_base("proj", session) == artifacts.resolve()


def test_serve_url_is_empty_for_a_linked_artifacts_folder(tmp_path):
    """The route refuses a project whose `.anton/artifacts` is a link, so a
    card URL for a file there would get only a 404."""
    project_dir = tmp_path / "my-project"
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "dash").mkdir(parents=True)
    (elsewhere / "dash" / "index.html").write_text("<h1>hi</h1>")
    (project_dir / ".anton").mkdir(parents=True)
    (project_dir / ".anton" / "artifacts").symlink_to(elsewhere, target_is_directory=True)

    with patch(
        "cowork.services.artifacts._registered_project_dirs",
        return_value=[project_dir],
    ):
        url = serve_url_for(str(project_dir / ".anton" / "artifacts" / "dash" / "index.html"), scope=LOCAL_SCOPE)

    assert url == ""
