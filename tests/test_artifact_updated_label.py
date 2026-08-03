"""Tests for the artifacts-page `updated` label sharing card_for_folder's
content_mtime clock, instead of metadata.json's mtime (ENG-1123 Bug 2)."""

import json
import os
import time
from pathlib import Path

from cowork.services.artifacts import _human_mtime, card_for_folder


def _touch(path: Path, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


def test_human_mtime_zero_is_empty():
    assert _human_mtime(0) == ""


def test_human_mtime_negative_is_empty():
    assert _human_mtime(-1) == ""


def test_human_mtime_positive_formats_an_age():
    assert _human_mtime(time.time() - 90) == "updated 1m ago"


def _make_document(tmp_path: Path, name: str = "report.csv") -> Path:
    root = tmp_path / "doc-art"
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps({"id": "doc-art", "type": "document", "primary": name}),
        encoding="utf-8",
    )
    (root / name).write_text("a,b\n1,2\n", encoding="utf-8")
    return root


def test_updated_changes_after_content_rewrite_without_touching_metadata(tmp_path: Path):
    root = _make_document(tmp_path)
    content = root / "report.csv"
    meta = root / "metadata.json"
    # Both start ~1 day old — far enough apart from "now" that the two
    # assertions below can't land in the same age bucket by coincidence.
    old = time.time() - 90_000
    _touch(meta, old)
    _touch(content, old)
    before = card_for_folder(root)["updated"]

    # Rewrite content only, bumping it to "now"; metadata.json's mtime is
    # untouched — this is exactly the Bug 2 scenario ("updated" used to be
    # stuck on metadata's clock and never moved after a content-only
    # rewrite).
    _touch(content, time.time())
    after = card_for_folder(root)["updated"]

    assert before != after
    assert after == "updated just now"


def test_updated_is_empty_for_freshly_created_empty_artifact(tmp_path: Path):
    # ENG-372 shape: metadata.json exists, no user content files yet.
    root = tmp_path / "empty-art"
    root.mkdir()
    (root / "metadata.json").write_text(
        json.dumps({"id": "empty-art", "type": "document"}), encoding="utf-8"
    )
    assert card_for_folder(root)["updated"] == ""
