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

