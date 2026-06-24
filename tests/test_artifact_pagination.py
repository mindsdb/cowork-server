"""Artifact listing pagination.

`list_artifacts` used to hard-return the newest 80 artifacts and silently drop
the rest — a real data-loss bug for any workspace with a large library. The
listing is now paginated (`list_artifacts_page`) and reports the true total, so
the client can render "showing N of M" + load-more and never silently lose an
artifact.

Fixtures build artifact folders under a temp project dir and patch
`_registered_project_dirs` to register it — the same isolation pattern as
test_artifacts_url.py, so the real ~/.cowork is never touched.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cowork.services import artifacts as artifacts_service
from cowork.services.artifacts import list_artifacts, list_artifacts_page


def _make_artifact(base: Path, slug: str, *, mtime: float | None = None) -> Path:
    """A minimal valid artifact folder (metadata.json + one file)."""
    folder = base / slug
    folder.mkdir(parents=True)
    (folder / "doc.md").write_text(f"# {slug}\n")
    (folder / "metadata.json").write_text(
        json.dumps({"slug": slug, "name": slug.title(), "type": "document"})
    )
    if mtime is not None:
        import os

        os.utime(folder / "metadata.json", (mtime, mtime))
    return folder


@pytest.fixture
def project_with_artifacts(tmp_path):
    """A registered project containing `count` artifacts, newest-slug-last by mtime.

    Yields ``(register_cm_factory, make)`` where calling the returned
    contextmanager patches `_registered_project_dirs` to the temp project.
    """
    project_dir = tmp_path / "proj"
    artifacts_dir = project_dir / ".anton" / "artifacts"
    artifacts_dir.mkdir(parents=True)

    def register():
        return patch.object(
            artifacts_service,
            "_registered_project_dirs",
            return_value=[project_dir],
        )

    return artifacts_dir, register


def test_page_reports_true_total_and_does_not_silently_cap(project_with_artifacts):
    """The whole point: more than the old 80-cap, total is honest, nothing dropped."""
    artifacts_dir, register = project_with_artifacts
    # 95 artifacts — past the old silent `[:80]` cutoff.
    for i in range(95):
        _make_artifact(artifacts_dir, f"a{i:03d}", mtime=1000 + i)

    with register():
        page = list_artifacts_page(limit=40, offset=0)

    assert page["total"] == 95, "total must reflect every artifact, not a cap"
    assert len(page["artifacts"]) == 40
    assert page["hasMore"] is True
    assert page["offset"] == 0
    assert page["limit"] == 40


def test_pagination_window_is_newest_first_and_contiguous(project_with_artifacts):
    artifacts_dir, register = project_with_artifacts
    for i in range(10):
        _make_artifact(artifacts_dir, f"a{i:02d}", mtime=1000 + i)

    with register():
        page1 = list_artifacts_page(limit=4, offset=0)
        page2 = list_artifacts_page(limit=4, offset=4)
        page3 = list_artifacts_page(limit=4, offset=8)

    # Newest mtime first → a09, a08, ...
    slugs1 = [c["slug"] for c in page1["artifacts"]]
    slugs2 = [c["slug"] for c in page2["artifacts"]]
    slugs3 = [c["slug"] for c in page3["artifacts"]]
    assert slugs1 == ["a09", "a08", "a07", "a06"]
    assert slugs2 == ["a05", "a04", "a03", "a02"]
    assert slugs3 == ["a01", "a00"]  # last page is short

    # No overlap, full coverage.
    all_slugs = slugs1 + slugs2 + slugs3
    assert len(all_slugs) == len(set(all_slugs)) == 10
    assert page3["hasMore"] is False


def test_offset_past_end_returns_empty_page(project_with_artifacts):
    artifacts_dir, register = project_with_artifacts
    for i in range(5):
        _make_artifact(artifacts_dir, f"a{i}", mtime=1000 + i)

    with register():
        page = list_artifacts_page(limit=10, offset=100)

    assert page["artifacts"] == []
    assert page["total"] == 5
    assert page["hasMore"] is False


def test_unreadable_metadata_excluded_from_total_and_page(project_with_artifacts):
    artifacts_dir, register = project_with_artifacts
    _make_artifact(artifacts_dir, "good", mtime=1000)
    # A folder with broken metadata must not count toward the total or appear.
    broken = artifacts_dir / "broken"
    broken.mkdir()
    (broken / "metadata.json").write_text("{ not json")

    with register():
        page = list_artifacts_page(limit=10, offset=0)

    assert page["total"] == 1
    assert [c["slug"] for c in page["artifacts"]] == ["good"]


def test_folder_filter_drops_artifacts_before_pagination(project_with_artifacts):
    """The endpoint passes a visibility filter; total must reflect only what
    passes it (so 'showing N of M' counts only visible artifacts)."""
    artifacts_dir, register = project_with_artifacts
    for i in range(6):
        _make_artifact(artifacts_dir, f"a{i}", mtime=1000 + i)

    # Hide the odd-indexed slugs.
    def only_even(folder: Path) -> bool:
        return int(folder.name[1:]) % 2 == 0

    with register():
        page = list_artifacts_page(limit=10, offset=0, folder_filter=only_even)

    slugs = sorted(c["slug"] for c in page["artifacts"])
    assert slugs == ["a0", "a2", "a4"]
    assert page["total"] == 3, "total counts only artifacts passing the filter"


def test_list_artifacts_full_list_has_no_cap(project_with_artifacts):
    """The back-compat full-list helper must return every artifact (the silent
    80-cap is gone), so in-process callers (inline chat cards) don't lose data."""
    artifacts_dir, register = project_with_artifacts
    for i in range(110):
        _make_artifact(artifacts_dir, f"a{i:03d}", mtime=1000 + i)

    with register():
        cards = list_artifacts()

    assert len(cards) == 110


def test_limit_is_clamped_to_a_sane_maximum(project_with_artifacts):
    artifacts_dir, register = project_with_artifacts
    _make_artifact(artifacts_dir, "a0", mtime=1000)

    with register():
        # A wildly large limit is clamped, never honored verbatim.
        page = list_artifacts_page(limit=10_000, offset=0)

    assert page["limit"] <= 500
