import pytest

from cowork.common.paths import safe_join, safe_join_lexical


def test_rejects_symlink_pointing_outside_base(tmp_path):
    """The exact cross-org case: a link inside org A resolving into org B."""
    org_a = tmp_path / "org-a"
    org_b = tmp_path / "org-b"
    (org_a / "projects").mkdir(parents=True)
    org_b.mkdir()
    (org_b / "secret.txt").write_text("org b data")
    (org_a / "projects" / "escape").symlink_to(org_b)

    with pytest.raises(ValueError):
        safe_join(org_a, "projects", "escape", "secret.txt")


def test_rejects_symlink_to_absolute_path(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "passwd").symlink_to("/etc/passwd")

    with pytest.raises(ValueError):
        safe_join(base, "passwd")


def test_rejects_symlinked_intermediate_directory(tmp_path):
    """The escape is a parent component, not the leaf."""
    base = tmp_path / "base"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    (outside / "f.txt").write_text("x")
    (base / "hop").symlink_to(outside)

    with pytest.raises(ValueError):
        safe_join(base, "hop", "f.txt")


def test_allows_symlink_that_stays_inside_base(tmp_path):
    base = tmp_path / "base"
    (base / "real").mkdir(parents=True)
    (base / "real" / "f.txt").write_text("x")
    (base / "link").symlink_to(base / "real")

    assert safe_join(base, "link", "f.txt").read_text() == "x"


def test_allows_path_that_does_not_exist_yet(tmp_path):
    """Callers safe_join paths before creating them; that must still work."""
    base = tmp_path / "base"
    base.mkdir()
    assert safe_join(base, "new", "nested", "f.txt") == base / "new" / "nested" / "f.txt"


def test_still_rejects_dotdot_without_any_symlink(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(ValueError):
        safe_join(base, "..", "elsewhere")


def test_base_and_sibling_prefix_are_unrelated(tmp_path):
    """Regression guard for the existing commonpath behaviour."""
    base = tmp_path / "base"
    base.mkdir()
    (tmp_path / "base-other").mkdir()
    with pytest.raises(ValueError):
        safe_join(base, "..", "base-other", "f.txt")


def test_safe_join_lexical_still_rejects_dotdot(tmp_path):
    """The weaker join keeps the lexical guard; it is not a free-for-all."""
    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(ValueError):
        safe_join_lexical(base, "..", "elsewhere")


def test_safe_join_lexical_does_not_resolve_symlinks(tmp_path):
    """The one property that distinguishes it from safe_join: a symlink whose
    target legitimately lives outside base (skill_links.py's per-project
    skill link) is not rejected, because nothing is read through it here."""
    base = tmp_path / "base"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    (base / "link").symlink_to(outside)

    assert safe_join_lexical(base, "link") == base / "link"


def test_returns_the_resolved_path_not_the_lexical_one(tmp_path):
    """The caller must act on the path that was actually checked.

    Returning the lexical path left every component a live symlink at use time,
    so containment was proved about a string nobody went on to use. It does not
    make the join atomic, but it removes the check-one-path-use-another gap.
    """
    base = tmp_path / "base"
    (base / "real").mkdir(parents=True)
    (base / "real" / "f.txt").write_text("x")
    (base / "link").symlink_to(base / "real")

    joined = safe_join(base, "link", "f.txt")

    assert joined == (base / "real" / "f.txt").resolve()
    assert "link" not in joined.parts
