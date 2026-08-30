"""Tests for SkillService.import_skill (upload of a skill file)."""

import io
import zipfile
from pathlib import Path

import pytest
from cowork.services.skills import SkillService

VALID = b"""---
name: My Test Skill
description: does a thing
---
Step 1. do the thing
"""


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


@pytest.fixture
def svc(tmp_path: Path):
    s = SkillService()
    s.root = tmp_path
    yield s


def test_import_md(svc: SkillService):
    skill = svc.import_skill(VALID, filename="thing.md")
    assert skill.name == "my-test-skill"  # normalized slug
    assert skill.description == "does a thing"
    assert skill.created_at is not None
    assert (svc.root / "my-test-skill" / "SKILL.md").exists()


def test_import_skill_extension(svc: SkillService):
    skill = svc.import_skill(VALID, filename="thing.skill")
    assert skill.name == "my-test-skill"


def test_import_unparseable(svc: SkillService):
    with pytest.raises(ValueError):
        svc.import_skill(b"no frontmatter here", filename="x.md")


def test_import_duplicate(svc: SkillService):
    svc.import_skill(VALID, filename="x.md")
    with pytest.raises(FileExistsError):
        svc.import_skill(VALID, filename="x.md")


def test_skill_dir_rejects_a_symlink_escape(svc: SkillService):
    container = svc.root
    svc.root = container / "skills"
    outside = container / "outside"
    svc.root.mkdir()
    outside.mkdir()
    (svc.root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="Invalid skill name"):
        svc._skill_dir("escape")


@pytest.mark.parametrize("slug", ["nested/escape", "nested\\escape"])
def test_skill_entry_requires_a_leaf_if_the_upstream_validator_changes(
    svc: SkillService,
    monkeypatch,
    slug,
):
    monkeypatch.setattr("cowork.services.skills.validate_name", lambda value: value)

    with pytest.raises(ValueError, match="Invalid skill name"):
        svc._skill_entry(slug)


def test_discard_incomplete_skill_removes_external_symlink_only(svc: SkillService):
    container = svc.root
    svc.root = container / "skills"
    outside = container / "outside"
    svc.root.mkdir()
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    link = svc.root / "escape"
    link.symlink_to(outside, target_is_directory=True)

    assert svc.discard_incomplete_skill("escape") is True

    assert not link.exists()
    assert (outside / "keep.txt").read_text() == "keep"


def test_discard_incomplete_skill_preserves_same_root_target(svc: SkillService):
    svc.create_skill(
        "real-skill",
        "Canonical instructions",
        description="Canonical skill",
    )
    link = svc.root / "alias"
    link.symlink_to(svc.root / "real-skill", target_is_directory=True)

    assert svc.discard_incomplete_skill("alias") is True

    assert not link.exists()
    assert svc.get_skill("real-skill").instructions == "Canonical instructions"


def test_rename_failure_restores_source_and_preserves_racing_destination(
    svc: SkillService,
    monkeypatch,
):
    svc.create_skill(
        "old-skill",
        "Original instructions",
        description="Original description",
    )
    source_file = svc.root / "old-skill" / "SKILL.md"
    original_bytes = source_file.read_bytes()
    destination_bytes: dict[str, bytes] = {}

    def fail_after_destination_appears(old_slug: str, new_slug: str) -> None:
        assert (old_slug, new_slug) == ("old-skill", "new-skill")
        svc.create_skill(
            "new-skill",
            "Concurrent winner",
            description="Destination must survive",
        )
        destination_bytes["content"] = (
            svc.root / "new-skill" / "SKILL.md"
        ).read_bytes()
        raise OSError("destination appeared before rename")

    monkeypatch.setattr(svc, "_rename_dir", fail_after_destination_appears)

    with pytest.raises(OSError, match="destination appeared"):
        svc.update_skill(
            "old-skill",
            label="new-skill",
            instructions="Uncommitted replacement",
        )

    assert source_file.read_bytes() == original_bytes
    restored = svc.get_skill("old-skill")
    assert restored.name == "old-skill"
    assert restored.instructions == "Original instructions"
    destination_file = svc.root / "new-skill" / "SKILL.md"
    assert destination_file.read_bytes() == destination_bytes["content"]
    assert svc.get_skill("new-skill").instructions == "Concurrent winner"


def test_import_zip_keeps_sibling_files(svc: SkillService):
    data = _zip({"SKILL.md": VALID, "assets/helper.py": b"print(1)\n"})
    skill = svc.import_skill(data, filename="pack.zip")
    assert skill.name == "my-test-skill"
    dest = svc.root / "my-test-skill"
    assert (dest / "SKILL.md").exists()
    assert (dest / "assets" / "helper.py").read_bytes() == b"print(1)\n"


def test_import_zip_wrapped_folder(svc: SkillService):
    # zip packed with its containing folder: myskill/SKILL.md + myskill/assets/...
    data = _zip({"myskill/SKILL.md": VALID, "myskill/assets/a.py": b"x\n"})
    skill = svc.import_skill(data, filename="pack.zip")
    assert skill.name == "my-test-skill"
    dest = svc.root / "my-test-skill"
    assert (dest / "SKILL.md").exists()
    assert (dest / "assets" / "a.py").read_bytes() == b"x\n"


def test_import_zip_single_md_renamed(svc: SkillService):
    data = _zip({"whatever.md": VALID})
    skill = svc.import_skill(data, filename="pack.zip")
    assert skill.name == "my-test-skill"
    assert (svc.root / "my-test-skill" / "SKILL.md").exists()


def test_import_zip_path_traversal_rejected(svc: SkillService):
    data = _zip({"SKILL.md": VALID, "../escape.txt": b"evil"})
    with pytest.raises(ValueError):
        svc.import_skill(data, filename="pack.zip")
