"""Regression tests for per-project skill link reconciliation.

Focus on the two properties that broke Windows boot (see skill_links.py):
  1. link create/remove is idempotent and round-trips, and
  2. a single unlinkable skill never aborts a full reconcile (boot safety).
"""
from pathlib import Path

import cowork.services.skill_links as sl


def test_ensure_and_remove_link_roundtrip(tmp_path: Path):
    target = tmp_path / "canon"
    target.mkdir()
    (target / "SKILL.md").write_text("hi", encoding="utf-8")
    link = tmp_path / "project" / "skills" / "canon"

    sl._ensure_symlink(link, target)
    assert sl._is_dir_link(link)
    assert (link / "SKILL.md").read_text(encoding="utf-8") == "hi"

    # Idempotent: a second call to the same target is a no-op, not an error.
    sl._ensure_symlink(link, target)
    assert sl._is_dir_link(link)

    sl._remove_link(link)
    assert not link.exists()
    # Removing an absent link is a no-op.
    sl._remove_link(link)


def test_ensure_link_retargets(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    link = tmp_path / "skills" / "s"

    sl._ensure_symlink(link, a)
    sl._ensure_symlink(link, b)  # retarget
    assert link.resolve() == b.resolve()


def test_reconcile_all_skips_unlinkable_skill(monkeypatch):
    """A skill whose links can't be reconciled is logged and skipped, so the
    rest still reconcile and boot proceeds — the WinError 1314 regression."""
    seen: list[str] = []

    def fake_reconcile(skill):
        seen.append(skill.name)
        if skill.name == "boom":
            raise OSError("A required privilege is not held by the client")

    monkeypatch.setattr(sl, "reconcile_skill_links", fake_reconcile)

    class _S:
        def __init__(self, name):
            self.name = name

    # Must not raise, and must attempt every skill despite the failure.
    sl.reconcile_all([_S("ok1"), _S("boom"), _S("ok2")])
    assert seen == ["ok1", "boom", "ok2"]
