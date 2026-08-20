"""`remote_skill_draft_payload` — turning a pod's wire entry into a draft card.

The pod's payload is untrusted, so most of these are about what must NOT get
through; the rest pin that a remote card matches the desktop one.
"""

from pathlib import Path

from cowork.services.task_objects import _skill_draft_payload, remote_skill_draft_payload

SKILL_MD = "---\nname: my-skill\ndescription: A test skill\nmetadata:\n  display_name: My Skill\n---\nsteps here"


def _entry(slug="my-skill", **files):
    return {"slug": slug, "files": {"SKILL.md": SKILL_MD, **files}}


def test_builds_a_card_from_the_wire_entry():
    card = remote_skill_draft_payload(_entry())
    assert card["slug"] == "my-skill"
    assert card["description"] == "A test skill"
    assert card["instructions"] == "steps here"
    assert card["skill_md"] == SKILL_MD


def test_siblings_reach_the_card():
    card = remote_skill_draft_payload(_entry(**{"recipe.md": "detail"}))
    assert card["files"] == [{"name": "recipe.md", "text": "detail"}]


def test_a_remote_card_matches_a_desktop_one(tmp_path):
    """Same folder contents must give the same card whichever path built it —
    the renderer and the Save call cannot tell them apart."""
    folder = tmp_path / "my-skill"
    folder.mkdir()
    (folder / "SKILL.md").write_text(SKILL_MD)
    (folder / "recipe.md").write_text("detail")

    desktop = _skill_draft_payload(folder)
    remote = remote_skill_draft_payload(_entry(**{"recipe.md": "detail"}))
    assert remote == desktop


def test_an_escaping_filename_is_dropped(tmp_path):
    """`a/../../evil` would land outside the temp dir — anywhere the server can
    write. The draft still comes through, minus that file."""
    card = remote_skill_draft_payload(_entry(**{"../../evil.txt": "pwned"}))
    assert card is not None
    assert card["files"] == []
    assert not (tmp_path.parent / "evil.txt").exists()


def test_a_nested_filename_is_dropped():
    card = remote_skill_draft_payload(_entry(**{"sub/inner.md": "x"}))
    assert [f["name"] for f in card["files"]] == []


def test_an_invalid_slug_is_rejected():
    for slug in ["../escape", "UPPER", "has space", "", "a" * 100]:
        assert remote_skill_draft_payload(_entry(slug=slug)) is None


def test_an_absolute_slug_cannot_name_a_folder():
    assert remote_skill_draft_payload(_entry(slug="/etc/passwd")) is None


def test_an_entry_without_skill_md_is_rejected():
    assert remote_skill_draft_payload({"slug": "my-skill", "files": {"recipe.md": "x"}}) is None


def test_a_malformed_entry_is_rejected():
    for bad in [None, {}, {"slug": "my-skill"}, {"files": {}}, {"slug": 1, "files": {}},
                {"slug": "my-skill", "files": "not-a-dict"}]:
        assert remote_skill_draft_payload(bad) is None


def test_non_string_file_values_are_skipped_not_fatal():
    card = remote_skill_draft_payload({
        "slug": "my-skill",
        "files": {"SKILL.md": SKILL_MD, "bad.md": 42, 7: "x"},
    })
    assert card is not None
    assert card["files"] == []


def test_nothing_is_left_on_disk():
    """The staging folder is a throwaway: a remote draft has no server-side home,
    unlike the desktop one that lives under the project."""
    card = remote_skill_draft_payload(_entry())
    assert not Path(card.get("path", "/nonexistent")).exists()
