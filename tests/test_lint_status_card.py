"""`cards_for_slugs` overlays the harness's in-memory lint verdict onto the
card it builds from disk — never read from metadata.json, since
`ChatSession.artifact_lint_status` is per-turn and anton no longer persists
any lint result to disk.
"""
from __future__ import annotations

import json

import pytest

from cowork.services.task_objects import cards_for_slugs


def _make_artifact(base, slug, *, meta: dict):
    folder = base / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.html").write_text("<html></html>")
    (folder / "metadata.json").write_text(json.dumps(meta))
    return folder


@pytest.fixture
def base(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    return root


def _meta(slug: str) -> dict:
    return {"slug": slug, "name": slug, "type": "html-app"}


def test_card_carries_lint_status_when_slug_matches(base):
    _make_artifact(base, "dash-1", meta=_meta("dash-1"))

    cards = cards_for_slugs(base, ["dash-1"], lint_status_by_slug={"dash-1": "has_errors"})

    assert cards[0]["lintStatus"] == "has_errors"


def test_card_has_no_lint_status_key_when_mapping_omits_the_slug(base):
    _make_artifact(base, "dash-1", meta=_meta("dash-1"))

    cards = cards_for_slugs(base, ["dash-1"], lint_status_by_slug={"other-slug": "has_errors"})

    assert "lintStatus" not in cards[0]


def test_card_has_no_lint_status_key_when_mapping_is_none(base):
    """Default behaviour, and what every existing caller gets unchanged."""
    _make_artifact(base, "dash-1", meta=_meta("dash-1"))

    cards = cards_for_slugs(base, ["dash-1"])

    assert "lintStatus" not in cards[0]


def test_card_has_no_lint_status_key_for_a_falsy_status(base):
    """Defensive: an empty-string status must not add a meaningless key."""
    _make_artifact(base, "dash-1", meta=_meta("dash-1"))

    cards = cards_for_slugs(base, ["dash-1"], lint_status_by_slug={"dash-1": ""})

    assert "lintStatus" not in cards[0]
