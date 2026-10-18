"""The managed folder's skills.

Every person lists them as built-in, with no capability, and a managed skill
wins over a same-named skill in the person's own store. Their names are
reserved: create, upload, rename into, edit, disable and delete of a managed
slug each answer 403 with the shared sentence, whether or not the person's
store holds that name too, and in every scope. A turn reads them from anton's
read-only roots after the harness's own skills, and never links a person's
same-named skill over them; a person's turn reads the skills others share
with them after the managed ones. With no managed folder, the listing and the
turn's roots are unchanged.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import delete

from cowork.api.v1.endpoints import skills as skills_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_open_session
from cowork.harnesses.anton_harness.harness import HOST_SKILLS_DIR
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.principal import personal_partition_id
from cowork.schemas.skills import SkillCreateRequest, SkillUpdateRequest
from cowork.services.skills import (
    SHARED_SKILL_REFUSAL,
    CodeSkillService,
    SkillImmutableError,
    SkillService,
    link_turn_skills,
)
from tests._turns import built_turn

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
PARTITION_A = personal_partition_id(DEPLOYMENT, SUB_A)
ALICE = TenantScope(personal=True, org_id=PARTITION_A, user_id=SUB_A, personal_resources=ALL_PERSONAL_RESOURCES)

MANAGED = "team-report"
MANAGED_BODY = "Write the report the operator's way."


def _write_skill(folder, slug: str, body: str) -> None:
    (folder / slug).mkdir(parents=True)
    (folder / slug / "SKILL.md").write_text(
        f"---\nname: {slug}\ndescription: {slug} skill\n---\n{body}\n", encoding="utf-8"
    )


@pytest.fixture
def stores(monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "home" / "skills"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "home" / "projects"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "home" / "memory"))
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "home" / "data-vault"))
    get_app_settings.cache_clear()
    yield tmp_path
    with get_open_session() as db:
        db.exec(delete(Conversation).where(Conversation.org_id == PARTITION_A))
        db.exec(delete(Project).where(Project.org_id == PARTITION_A))
        db.commit()
    get_app_settings.cache_clear()


@pytest.fixture
def managed_skill(managed_dir, personal_deployment, stores):
    get_app_settings.cache_clear()
    _write_skill(managed_dir.skills_dir, MANAGED, MANAGED_BODY)
    return managed_dir


@pytest.fixture
def scoped(stores):
    raw = get_open_session()
    try:
        yield lambda scope: ScopedSession(raw, scope)
    finally:
        raw.close()


def _listed(scoped, scope) -> dict[str, dict]:
    return {skill["id"]: skill for skill in skills_ep.list_skills(scoped(scope), None)["skills"]}


def _refusal(call) -> HTTPException:
    with pytest.raises(HTTPException) as refused:
        call()
    assert refused.value.status_code == 403
    assert refused.value.detail == SHARED_SKILL_REFUSAL
    return refused.value


# ── the listing ─────────────────────────────────────────────────────────────


def test_a_managed_skill_is_listed_built_in_with_no_capability(managed_skill, scoped):
    SkillService(ALICE).create_skill("my-notes", "Mine.")

    listed = _listed(scoped, ALICE)

    managed = listed[MANAGED]
    assert managed["isBuiltin"] is True
    assert managed["capabilities"] == {"canEdit": False, "canDelete": False, "canDisable": False, "canShare": False}
    assert managed["declarative"].strip() == MANAGED_BODY
    # The person's own skill keeps its capabilities.
    assert listed["my-notes"]["isBuiltin"] is False
    assert listed["my-notes"]["capabilities"]["canEdit"] is True


def test_a_managed_skill_wins_over_a_same_named_own_skill(managed_skill, scoped, stores):
    own_store = SkillService(ALICE).root
    _write_skill(own_store, MANAGED, "My own version.")

    listed = _listed(scoped, ALICE)
    assert listed[MANAGED]["declarative"].strip() == MANAGED_BODY
    assert listed[MANAGED]["isBuiltin"] is True

    read = skills_ep.get_skill(MANAGED, scoped(ALICE), None)
    assert read["declarative"].strip() == MANAGED_BODY
    assert read["capabilities"]["canEdit"] is False


def test_without_a_managed_folder_the_listing_is_unchanged(personal_deployment, stores, scoped):
    SkillService(ALICE).create_skill("my-notes", "Mine.")
    listed = _listed(scoped, ALICE)
    assert MANAGED not in listed
    assert listed["my-notes"]["capabilities"]["canEdit"] is True
    assert not any(skill["id"] == MANAGED for skill in listed.values())


# ── every change to a managed slug ──────────────────────────────────────────


@pytest.mark.parametrize("own_copy", [False, True], ids=["no-own-copy", "hidden-own-copy"])
def test_every_change_to_a_managed_slug_is_refused(managed_skill, scoped, own_copy):
    service = SkillService(ALICE)
    service.create_skill("my-notes", "Mine.")
    if own_copy:
        _write_skill(service.root, MANAGED, "My own version.")
    scope = scoped(ALICE)

    _refusal(lambda: skills_ep.create_skill(SkillCreateRequest(label=MANAGED, instructions="x"), scope, None))
    upload = f"---\nname: {MANAGED}\ndescription: mine\n---\nMine.\n".encode()
    _refusal(lambda: skills_ep._upload_skill_bytes(upload, "SKILL.md", scope, None))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as packed:
        packed.writestr(f"{MANAGED}/SKILL.md", upload.decode())
    _refusal(lambda: skills_ep._upload_skill_bytes(archive.getvalue(), "skill.zip", scope, None))
    _refusal(lambda: skills_ep.update_skill("my-notes", SkillUpdateRequest(label=MANAGED), scope, None))
    _refusal(lambda: skills_ep.update_skill(MANAGED, SkillUpdateRequest(instructions="x"), scope, None))
    _refusal(lambda: skills_ep.update_skill(MANAGED, SkillUpdateRequest(enabled=False), scope, None))
    _refusal(lambda: skills_ep.delete_skill(MANAGED, scope, None))

    # Nothing changed: the managed skill, the person's own skills.
    assert (managed_skill.skills_dir / MANAGED / "SKILL.md").read_text().strip().endswith(MANAGED_BODY)
    assert service.get_skill("my-notes").instructions.strip() == "Mine."
    assert (service.root / MANAGED).exists() is own_copy


LOCKED = "locked-report"


def _unreadable_managed_skill(managed_dir, monkeypatch) -> Path:
    """A managed skill whose SKILL.md the server's user can't stat, as a
    folder left 0000 after the start would be. Driven by a patch, so it holds
    whether or not the tests run as root."""
    _write_skill(managed_dir.skills_dir, LOCKED, "Locked.")
    locked_file = managed_dir.skills_dir / LOCKED / "SKILL.md"
    real_is_file = Path.is_file

    def is_file(self: Path) -> bool:
        if self == locked_file:
            raise PermissionError(13, "Permission denied")
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", is_file)
    return locked_file


def test_an_unreadable_managed_skill_is_unlisted_but_keeps_its_name(managed_skill, scoped, monkeypatch):
    """The listing answers without a managed skill it can't read, and its name
    stays the operator's: a create under it is refused, not written."""
    _unreadable_managed_skill(managed_skill, monkeypatch)

    listed = _listed(scoped, ALICE)

    assert MANAGED in listed
    assert LOCKED not in listed
    _refusal(lambda: skills_ep.create_skill(SkillCreateRequest(label=LOCKED, instructions="x"), scoped(ALICE), None))
    assert not (SkillService(ALICE).root / LOCKED).exists()


def test_a_turn_never_links_an_own_skill_over_an_unreadable_managed_one(managed_skill, stores, monkeypatch):
    """While the managed copy can't be read, a person's same-named skill still
    stays out of the turn: the reserved name, not the readable folders, is the
    rule every link path uses."""
    service = SkillService(ALICE)
    service.create_skill("my-notes", "Mine.")
    _write_skill(service.root, LOCKED, "My own version.")
    _unreadable_managed_skill(managed_skill, monkeypatch)
    turn_skills = stores / "turn-skills"

    link_turn_skills(scope=ALICE, project_name="general", skills_dir=turn_skills)

    linked = {entry.name for entry in turn_skills.iterdir()}
    assert "my-notes" in linked
    assert LOCKED not in linked


def test_the_service_refuses_too_in_every_scope(managed_dir, stores):
    get_app_settings.cache_clear()
    _write_skill(managed_dir.skills_dir, MANAGED, MANAGED_BODY)
    desktop = SkillService(LOCAL_SCOPE)
    with pytest.raises(SkillImmutableError, match="shared with you"):
        desktop.create_skill(MANAGED, "x")
    _write_skill(desktop.root, MANAGED, "Mine.")
    with pytest.raises(SkillImmutableError, match="shared with you"):
        desktop.delete_skill(MANAGED)
    with pytest.raises(SkillImmutableError, match="shared with you"):
        desktop.update_skill(MANAGED, instructions="x")
    # The Code store never reserves the general catalogue's managed names.
    assert CodeSkillService(LOCAL_SCOPE).is_managed_skill(MANAGED) is False


def test_the_persons_own_skills_stay_editable(managed_skill, scoped):
    SkillService(ALICE).create_skill("my-notes", "Mine.")
    edited = skills_ep.update_skill("my-notes", SkillUpdateRequest(instructions="Changed."), scoped(ALICE), None)
    assert edited["declarative"] == "Changed."
    skills_ep.delete_skill("my-notes", scoped(ALICE), None)
    assert "my-notes" not in _listed(scoped, ALICE)


# ── a turn ──────────────────────────────────────────────────────────────────


def test_a_turn_never_links_a_same_named_own_skill(managed_skill, stores):
    service = SkillService(ALICE)
    service.create_skill("my-notes", "Mine.")
    _write_skill(service.root, MANAGED, "My own version.")
    turn_skills = stores / "turn-skills"

    link_turn_skills(scope=ALICE, project_name="general", skills_dir=turn_skills)

    linked = {entry.name for entry in turn_skills.iterdir()}
    assert "my-notes" in linked
    assert MANAGED not in linked


def _links(folder) -> dict[str, object]:
    return {entry.name: entry.resolve() for entry in folder.iterdir() if entry.is_symlink()}


async def test_a_turn_reads_managed_skills_after_the_host_skills(monkeypatch, managed_skill, stores):
    """The turn links each managed skill into its scratch folder, never handing
    anton the managed folder itself, and reads that folder after the host
    skills and before the skills others share."""
    async with built_turn(monkeypatch=monkeypatch, tmp_path=stores, scope=ALICE) as turn:
        assert turn.config.settings.skills_extra_roots == [
            HOST_SKILLS_DIR,
            turn.scratch / "managed-skills",
            turn.scratch / "shared-skills",
        ]
        assert _links(turn.scratch / "managed-skills") == {
            MANAGED: (managed_skill.skills_dir / MANAGED).resolve()
        }


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="POSIX modes, enforced for non-root")
async def test_an_unreadable_managed_skill_leaves_the_turns_other_skills(monkeypatch, managed_skill, stores):
    """A managed skill folder the server can't read, added after the start (no
    restart), leaves every other skill in the turn: anton drops a whole root's
    skills when it can't stat one folder in it, so the turn lists the readable
    managed skill and the person's own beside it, never none."""
    from anton.core.memory.skills import SkillStore

    SkillService(ALICE).create_skill("my-notes", "Mine.")
    locked = managed_skill.skills_dir / "locked-report"
    _write_skill(managed_skill.skills_dir, locked.name, "Locked.")
    locked.chmod(0o000)
    try:
        async with built_turn(monkeypatch=monkeypatch, tmp_path=stores, scope=ALICE) as turn:
            settings = turn.config.settings
            listed = {
                summary["label"]
                for summary in SkillStore(
                    root=settings.skills_root, extra_roots=settings.skills_extra_roots
                ).list_summaries()
            }
    finally:
        locked.chmod(0o755)

    assert {MANAGED, "my-notes"} <= listed
    assert locked.name not in listed


async def test_without_a_managed_folder_a_turn_reads_the_host_skills_only(monkeypatch, stores):
    async with built_turn(monkeypatch=monkeypatch, tmp_path=stores, scope=None) as turn:
        assert turn.config.settings.skills_extra_roots == [HOST_SKILLS_DIR]
