"""`html_artifacts()` — the list behind `GET /publish/` — offers artifact
content only, never README, generation inputs or revision copies; and
`publish_artifact` refuses those files when asked for them directly."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cowork.db.scoped import LOCAL_SCOPE
from cowork.services.artifacts import html_artifacts
from cowork.services.publish import publish_artifact

_FULLSTACK = "fullstack-stateless-app"


def _write(path: Path, text: str = "x", mtime: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _meta(folder: Path, **fields) -> None:
    _write(folder / "metadata.json", json.dumps({"id": folder.name, "type": "html-app", **fields}))


def _entries(*containers: Path) -> list[dict]:
    with patch("cowork.services.artifacts._scan_artifact_dirs", lambda *, scope: list(containers)), \
            patch("cowork.services.artifacts._fullstack_types", lambda: frozenset({_FULLSTACK})):
        return html_artifacts(scope=LOCAL_SCOPE)


def _listed(container: Path) -> set[str]:
    return {Path(e["path"]).relative_to(container).as_posix() for e in _entries(container)}


def test_non_content_files_are_not_listed(tmp_path: Path):
    art = tmp_path / "report"
    _meta(art, primary="index.html")
    _write(art / "index.html", "<h1>hi</h1>")
    _write(art / "README.md")
    _write(art / "prd.md")
    _write(art / "spec.md")
    _write(art / ".revisions" / "x.html")
    _write(art / ".revisions" / "r1" / "notes.md")

    assert _listed(tmp_path) == {"report/index.html"}


def test_non_content_names_match_first_component_only(tmp_path: Path):
    art = tmp_path / "site"
    _meta(art, primary="index.html")
    _write(art / "index.html")
    _write(art / "docs" / "README.md")
    _write(art / "docs" / "prd.md")
    _write(art / "docs" / ".revisions" / "old.html")

    assert _listed(tmp_path) == {"site/index.html", "site/docs/README.md", "site/docs/prd.md"}


def test_symlinks_are_not_listed(tmp_path: Path):
    art = tmp_path / "site"
    _meta(art, primary="index.html")
    _write(art / "index.html")
    _write(art / ".revisions" / "r1" / "index.html")
    (art / "latest.html").symlink_to(art / ".revisions" / "r1" / "index.html")

    assert _listed(tmp_path) == {"site/index.html"}


def test_fullstack_artifact_stays_one_entry(tmp_path: Path):
    art = tmp_path / "app"
    _meta(art, type=_FULLSTACK, name="My App", primary="static/index.html")
    _write(art / "static" / "index.html")
    _write(art / "static" / "about.html")
    _write(art / "README.md")
    _write(art / "spec.md")
    _write(art / ".revisions" / "static" / "index.html")

    assert _listed(tmp_path) == {"app/static/index.html"}
    assert [e["title"] for e in _entries(tmp_path)] == ["My App"]


def test_fullstack_without_primary_falls_back_to_a_content_page(tmp_path: Path):
    art = tmp_path / "app"
    _meta(art, type=_FULLSTACK, name="My App")
    _write(art / "static" / "index.html", mtime=1000)
    _write(art / "spec.md", mtime=3000)
    _write(art / ".revisions" / "static" / "index.html", mtime=4000)

    assert _listed(tmp_path) == {"app/static/index.html"}


def test_fullstack_with_non_page_primary_is_listed(tmp_path: Path):
    art = tmp_path / "api"
    _meta(art, type=_FULLSTACK, name="API", primary="backend.py")
    _write(art / "backend.py")
    _write(art / "README.md")
    _write(art / "prd.md")

    assert _listed(tmp_path) == {"api/backend.py"}


def test_non_content_files_do_not_take_slots_in_the_cap(tmp_path: Path):
    for i in range(40):
        art = tmp_path / f"a{i:02d}"
        _meta(art, primary="index.html")
        _write(art / "index.html", mtime=1000 + i)
        _write(art / "README.md", mtime=5000 + i)
        _write(art / "prd.md", mtime=5000 + i)

    listed = _listed(tmp_path)
    assert len(listed) == 40
    assert all(p.endswith("/index.html") for p in listed)


def test_cap_keeps_the_newest_across_projects(tmp_path: Path):
    old_project = tmp_path / "p1"
    for i in range(40):
        _write(old_project / f"a{i:02d}" / "index.html", mtime=1000 + i)
    new_project = tmp_path / "p2"
    _write(new_project / "fresh" / "index.html", mtime=9000)

    paths = [Path(e["path"]) for e in _entries(old_project, new_project)]
    assert len(paths) == 40
    assert paths[0] == new_project / "fresh" / "index.html"


@pytest.mark.parametrize("rel", ["prd.md", "README.md", ".revisions/r1/index.html"])
def test_publish_refuses_non_content_files(tmp_path: Path, rel: str):
    art = tmp_path / "report"
    _meta(art, primary="index.html")
    _write(art / "index.html")
    _write(art / rel)

    with pytest.raises(ValueError, match="artifact content can be published"):
        publish_artifact(
            art / rel, artifacts_base=tmp_path, api_key="k", publish_url="https://example.invalid",
        )


def test_autopublish_skips_a_non_content_primary(tmp_path: Path):
    from cowork.services.artifact_autopublish import REASON_NOT_PUBLISHABLE, needs_publish

    art = tmp_path / "notes"
    _meta(art, primary="README.md")
    _write(art / "README.md")
    _write(art / "data.csv")

    assert needs_publish(art, tmp_path).reason == REASON_NOT_PUBLISHABLE
