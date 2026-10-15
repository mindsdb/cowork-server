"""`html_artifacts()` — the list behind `GET /publish/` — offers artifact
content only, never README, generation inputs or revision copies."""

import json
from pathlib import Path
from unittest.mock import patch

from cowork.services.artifacts import html_artifacts


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _listed(container: Path) -> set[str]:
    with patch("cowork.services.artifacts._scan_artifact_dirs", lambda: [container]):
        return {str(Path(e["path"]).relative_to(container)) for e in html_artifacts()}


def test_non_content_files_are_not_listed(tmp_path: Path):
    art = tmp_path / "report"
    _write(art / "metadata.json", json.dumps({"id": "r", "type": "html-app", "primary": "index.html"}))
    _write(art / "index.html", "<h1>hi</h1>")
    _write(art / "README.md")
    _write(art / "prd.md")
    _write(art / "spec.md")
    _write(art / ".revisions" / "x.html")
    _write(art / ".revisions" / "r1" / "notes.md")

    assert _listed(tmp_path) == {"report/index.html"}


def test_non_content_names_match_first_component_only(tmp_path: Path):
    art = tmp_path / "site"
    _write(art / "metadata.json", json.dumps({"id": "s", "type": "html-app", "primary": "index.html"}))
    _write(art / "index.html")
    _write(art / "docs" / "README.md")
    _write(art / "docs" / "prd.md")
    _write(art / "docs" / ".revisions" / "old.html")

    assert _listed(tmp_path) == {"site/index.html", "site/docs/README.md", "site/docs/prd.md"}


def test_fullstack_artifact_stays_one_entry(tmp_path: Path):
    art = tmp_path / "app"
    _write(art / "metadata.json", json.dumps({
        "id": "a", "type": "fullstack-app", "name": "My App", "primary": "static/index.html",
    }))
    _write(art / "static" / "index.html")
    _write(art / "static" / "about.html")
    _write(art / "README.md")
    _write(art / "spec.md")
    _write(art / ".revisions" / "static" / "index.html")

    with patch("cowork.services.artifacts._fullstack_types", lambda: frozenset({"fullstack-app"})), \
            patch("cowork.services.artifacts._scan_artifact_dirs", lambda: [tmp_path]):
        entries = html_artifacts()

    assert [(e["title"], Path(e["path"]).relative_to(tmp_path).as_posix()) for e in entries] == [
        ("My App", "app/static/index.html"),
    ]
