"""Tests for the file-backed skill store and the skill API's org-mode gates."""

import io
import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from cowork.api.v1.endpoints import skills as skills_api
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession, TenantScope
from cowork.models.shared_resource import (
    SharedResourceAttribution,
    SharedResourceMutation,
)
from cowork.principal import Principal
from cowork.schemas.skills import SkillCreateRequest, SkillUpdateRequest
from cowork.services.shared_resources import SKILL, SharedResourceAccess
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


def test_finalize_staged_delete_rejects_a_foreign_path(svc: SkillService):
    outside = svc.root.parent / "outside" / "keep"
    outside.mkdir(parents=True)

    with pytest.raises(ValueError, match="Invalid staged skill deletion path"):
        svc.finalize_staged_delete(outside)

    assert outside.is_dir()


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


def test_rename_refuses_a_destination_link_instead_of_following_it(
    svc: SkillService,
):
    """A link planted at the destination between check and rename is refused.

    Resolving the destination to a real path moves the directory onto the
    link's target instead, so the empty directory of a skill somebody else is
    still writing disappears without a trace.
    """
    svc.create_skill("old-skill", "Original instructions", description="Original")
    decoy = svc.root / "decoy-skill"
    decoy.mkdir()
    (svc.root / "new-skill").symlink_to(decoy, target_is_directory=True)

    with pytest.raises(OSError):
        svc._rename_dir("old-skill", "new-skill")

    assert list(decoy.iterdir()) == []
    assert (svc.root / "new-skill").is_symlink()
    assert svc.get_skill("old-skill").instructions == "Original instructions"


def test_restore_staged_delete_refuses_a_link_planted_at_the_slug(
    svc: SkillService,
):
    svc.create_skill("staged-skill", "Original instructions", description="Original")
    decoy = svc.root / "decoy-skill"
    decoy.mkdir()
    staged = svc.stage_delete("staged-skill")
    (svc.root / "staged-skill").symlink_to(decoy, target_is_directory=True)

    with pytest.raises(OSError):
        svc.restore_staged_delete("staged-skill", staged)

    assert list(decoy.iterdir()) == []
    # The bytes stay staged, so a later restore can still put them back.
    assert (staged / "SKILL.md").is_file()


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


ORG = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ALICE = "11111111-1111-4111-8111-111111111111"
BOB = "22222222-2222-4222-8222-222222222222"


def _principal(user_id: str) -> Principal:
    return Principal(
        user_id=user_id,
        org_id=ORG,
        email=f"{user_id[0]}@example.com",
        roles=frozenset(),
    )


def _scoped(engine, user_id: str) -> ScopedSession:
    return ScopedSession(
        Session(engine),
        TenantScope(org_mode=True, org_id=ORG, user_id=user_id),
    )


@pytest.fixture()
def org_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    get_app_settings.cache_clear()
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    yield engine
    get_app_settings.cache_clear()


def test_rename_into_a_destination_claimed_after_the_check_conflicts(
    org_engine,
    monkeypatch,
):
    """A destination taken between the pre-flight check and the audit commit.

    The unique attribution key would fail the commit with a server error, and
    the caller would never learn which name was refused.
    """
    alice = _scoped(org_engine, ALICE)
    skills_api.create_skill(
        SkillCreateRequest(label="alpha skill", instructions="original"),
        alice,
        _principal(ALICE),
    )
    service = SkillService(alice.scope)
    original_bytes = (service._skill_dir("alpha-skill") / "SKILL.md").read_bytes()
    bob_access = SharedResourceAccess(_scoped(org_engine, BOB), _principal(BOB))
    store_update = SkillService.update_skill

    def claim_the_destination_after_the_rename(self, skill_id, **changes):
        renamed = store_update(self, skill_id, **changes)
        bob_access.claim(SKILL, renamed.name)
        return renamed

    monkeypatch.setattr(
        SkillService,
        "update_skill",
        claim_the_destination_after_the_rename,
    )

    with pytest.raises(HTTPException) as conflict:
        skills_api.update_skill(
            "alpha-skill",
            SkillUpdateRequest(label="beta skill"),
            alice,
            _principal(ALICE),
        )

    assert conflict.value.status_code == 409
    assert "beta-skill" in conflict.value.detail
    assert not service._skill_entry("beta-skill").is_dir()
    canonical = service._skill_dir("alpha-skill") / "SKILL.md"
    assert canonical.read_bytes() == original_bytes


def test_rename_reports_a_lost_attribution_key_race_as_a_conflict(
    org_engine,
    monkeypatch,
):
    alice = _scoped(org_engine, ALICE)
    skills_api.create_skill(
        SkillCreateRequest(label="racing skill", instructions="original"),
        alice,
        _principal(ALICE),
    )
    service = SkillService(alice.scope)
    original_bytes = (service._skill_dir("racing-skill") / "SKILL.md").read_bytes()

    def lose_the_unique_key(*_args, **_kwargs):
        raise IntegrityError(
            "UPDATE shared_resource_attributions",
            {},
            Exception("uq_shared_resource_attribution_key"),
        )

    monkeypatch.setattr(SharedResourceAccess, "record_updates", lose_the_unique_key)

    with pytest.raises(HTTPException) as conflict:
        skills_api.update_skill(
            "racing-skill",
            SkillUpdateRequest(label="won by somebody else"),
            alice,
            _principal(ALICE),
        )

    assert conflict.value.status_code == 409
    assert "won-by-somebody-else" in conflict.value.detail
    assert not service._skill_entry("won-by-somebody-else").is_dir()
    canonical = service._skill_dir("racing-skill") / "SKILL.md"
    assert canonical.read_bytes() == original_bytes


def test_update_through_an_alias_audits_the_canonical_slug(org_engine):
    """Attribution follows the directory the mutation lands on.

    Keying it on the request path segment instead files the audit under a name
    nothing owns, and leaves authorization reading a different row than the one
    the edit wrote.
    """
    alice = _scoped(org_engine, ALICE)
    skills_api.create_skill(
        SkillCreateRequest(label="canonical skill", instructions="original"),
        alice,
        _principal(ALICE),
    )
    service = SkillService(alice.scope)
    (service.root / "alias-skill").symlink_to(
        service.root / "canonical-skill",
        target_is_directory=True,
    )

    edited = skills_api.update_skill(
        "alias-skill",
        SkillUpdateRequest(instructions="edited through the alias"),
        alice,
        _principal(ALICE),
    )

    assert edited["id"] == "canonical-skill"
    rows = alice.exec(alice.select(SharedResourceAttribution)).all()
    assert [row.resource_key for row in rows] == ["canonical-skill"]
    events = alice.exec(alice.select(SharedResourceMutation)).all()
    assert {event.resource_key for event in events} == {"canonical-skill"}


def test_delete_through_an_alias_audits_the_canonical_slug(org_engine):
    alice = _scoped(org_engine, ALICE)
    skills_api.create_skill(
        SkillCreateRequest(label="deletable skill", instructions="original"),
        alice,
        _principal(ALICE),
    )
    service = SkillService(alice.scope)
    (service.root / "alias-skill").symlink_to(
        service.root / "deletable-skill",
        target_is_directory=True,
    )

    skills_api.delete_skill("alias-skill", alice, _principal(ALICE))

    assert alice.exec(alice.select(SharedResourceAttribution)).all() == []
    events = alice.exec(alice.select(SharedResourceMutation)).all()
    assert {event.resource_key for event in events} == {"deletable-skill"}
    assert not service._skill_entry("deletable-skill").exists()


def test_create_reserves_no_claim_for_an_occupied_slug(org_engine, monkeypatch):
    """Bytes with no attribution row: a store seeded before ownership existed.

    A claim reserved over them survives a crash, and recovery then hands the
    skill to whoever reserved it rather than to whoever wrote it.
    """
    alice = _scoped(org_engine, ALICE)
    SkillService(alice.scope).create_skill(
        label="occupied skill",
        instructions="existing bytes",
    )
    reserved: list[tuple[str, str]] = []
    reserve_claim = SharedResourceAccess.reserve_claim

    def record_reservation(self, kind, key, **options):
        reserved.append((kind, key))
        return reserve_claim(self, kind, key, **options)

    monkeypatch.setattr(SharedResourceAccess, "reserve_claim", record_reservation)

    with pytest.raises(HTTPException) as conflict:
        skills_api.create_skill(
            SkillCreateRequest(label="occupied skill", instructions="second writer"),
            alice,
            _principal(ALICE),
        )

    assert conflict.value.status_code == 409
    assert reserved == []
    assert alice.exec(alice.select(SharedResourceAttribution)).all() == []
    assert (
        SkillService(alice.scope).get_skill("occupied-skill").instructions
        == "existing bytes"
    )
