"""Regression test for artifact serve URL prefix.

The serve_url_for() helper must produce URLs that match the router
mount point. This has regressed twice already (the prefix was /v1/
when the router uses /api/v1/), so we pin it with a test.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from cowork.api.v1.router import api_router
from cowork.services.artifacts import serve_url_for


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
            url = serve_url_for(str(test_file))

        assert url, "serve_url_for returned empty string"
        assert url.startswith(expected_prefix), (
            f"URL prefix mismatch: got {url!r}, expected to start with {expected_prefix!r}"
        )
