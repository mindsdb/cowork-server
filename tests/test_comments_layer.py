"""On-artifact comment marker layer injection.

The layer is injected into the top-level HTML document served for preview ONLY
when the renderer opts in via the activation query flag. Asset requests and
flag-less requests must stream untouched.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from cowork.server import app
from cowork.services.comments_layer import ACTIVATION_PARAM, LAYER_JS, inject_layer

client = TestClient(app)

_HTML = "<html><head></head><body><h1>Report</h1></body></html>"


def test_inject_layer_before_body_close():
    out = inject_layer(_HTML)
    assert "anton-comments" in out
    assert out.index("<script>") < out.index("</body>")


def test_inject_layer_appends_when_no_body():
    out = inject_layer("<div>x</div>")
    assert out.startswith("<div>x</div>")
    assert out.rstrip().endswith("</script>")


def test_layer_js_has_no_script_terminator():
    # A literal </script> in the payload would break out of the injected tag.
    assert "</script>" not in LAYER_JS


def _make_project(tmp: str):
    project_dir = Path(tmp) / "proj"
    artifacts = project_dir / ".anton" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "index.html").write_text(_HTML, encoding="utf-8")
    (artifacts / "styles.css").write_text("body{color:red}", encoding="utf-8")
    return project_dir


def test_serve_injects_only_with_flag():
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = _make_project(tmp)
        with patch(
            "cowork.services.artifacts._registered_project_dirs",
            return_value=[project_dir],
        ), patch(
            "cowork.services.artifacts._projects_root",
            return_value=project_dir.parent,
        ):
            # Entry document with the flag → layer injected.
            r = client.get(f"/api/v1/artifacts/serve/proj/index.html?{ACTIVATION_PARAM}=1")
            assert r.status_code == 200
            assert "anton-comments" in r.text

            # Same document without the flag → untouched.
            r2 = client.get("/api/v1/artifacts/serve/proj/index.html")
            assert r2.status_code == 200
            assert "anton-comments" not in r2.text

            # Non-HTML asset with the flag → untouched (only text/html is wrapped).
            r3 = client.get(f"/api/v1/artifacts/serve/proj/styles.css?{ACTIVATION_PARAM}=1")
            assert r3.status_code == 200
            assert "anton-comments" not in r3.text
