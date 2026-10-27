"""Skills that record their creator, and sharing one with everyone.

On a deployment that trusts identity headers, every skill created through the
server records its creator's sub in its metadata: a create or upload stamps
the person, never a value the request or file carries, an edit, rename or
project-rename rewrite keeps it, and the operator view stamps nobody.

With skills personal, a person shares one of their own skills, or hides it
again, with PUT /api/v1/skills/{skill_id}/sharing, or shares it as they
create or upload it. Either lists it in ``<COWORK_SHARED_DIR>/shared/skills.json``
(cowork.shared_skills). Everyone else lists it after the built-ins, the
managed skills and their own, reads it live from its creator's store with
``shared`` true and no capability, and every change they try, disabling
included, answers 403 with the shared sentence. Their turns link it
read-only, after the managed skills, with the selection rule applied to the
skill's own metadata. Its creator sees it ``shared`` with every capability and
edits, disables and deletes it as before, and everyone sees each change; its
rename is refused until it is hidden, and deleting it removes its entry.

With skills shared, everyone reads the one base store. A person changes only
a skill they created, and the operator view changes any; nobody shares one
skill at a time. The desktop and hosted org tenancy keep every skills route as
before, and share nothing.
"""
from __future__ import annotations

import errno
import io
import json
import logging
import os
import threading
import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

import cowork.server as cowork_server
from cowork import managed, shared_skills
from cowork.api.v1.endpoints import skills as skills_ep
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_open_session
from cowork.harnesses.anton_harness.harness import HOST_SKILLS_DIR
from cowork.managed import readonly_skill_dirs
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.skill import META_CREATED_BY
from cowork.principal import Principal, personal_partition_id
from cowork.schemas.skills import SkillCreateRequest, SkillSharingRequest, SkillUpdateRequest
from cowork.services.skills import (
    BUILTIN_SKILL_SLUGS,
    SHARED_SKILL_REFUSAL,
    SHARED_SKILL_RENAME_REFUSAL,
    SKILL_SHARING_BUSY,
    SKILL_SHARING_UNAVAILABLE,
    SharedSkillsBusyError,
    SkillAlreadyExistsError,
    SkillImmutableError,
    SkillService,
    SkillSharingError,
    agent_skill_names,
    link_turn_skills,
)
from tests._fakes import shared_skill_list_lock_held
from tests._turns import built_turn

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
SUB_B = "22222222-2222-4222-8222-222222222222"
PARTITION_A = personal_partition_id(DEPLOYMENT, SUB_A)
PARTITION_B = personal_partition_id(DEPLOYMENT, SUB_B)
SHARED_RESOURCES = frozenset(ALL_PERSONAL_RESOURCES - {PersonalResource.SKILLS})
NO_CAPABILITY = {"canEdit": False, "canDelete": False, "canDisable": False, "canShare": False}
# The caller's own skill where skills are personal: theirs to change and share.
EVERY_CAPABILITY = {"canEdit": True, "canDelete": True, "canDisable": True, "canShare": True}
# A skill the caller changes where nobody shares one at a time: skills left
# shared, the desktop and hosted org tenancy.
CHANGE_CAPABILITY = {"canEdit": True, "canDelete": True, "canDisable": True, "canShare": False}
TOKEN = "shared-skills-test-token"

SHARED = "team-report"
SHARED_BODY = "Write the report the team's way."


def _person(*, partition: str, sub: str, resources=ALL_PERSONAL_RESOURCES) -> TenantScope:
    return TenantScope(personal=True, org_id=partition, user_id=sub, personal_resources=resources)


ALICE = _person(partition=PARTITION_A, sub=SUB_A)
BOB = _person(partition=PARTITION_B, sub=SUB_B)
# The principals TrustedHeaderMiddleware builds for each: the partition is
# their org id.
ALICE_PRINCIPAL = Principal(user_id=SUB_A, org_id=PARTITION_A)
BOB_PRINCIPAL = Principal(user_id=SUB_B, org_id=PARTITION_B)


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
        db.exec(delete(Conversation).where(Conversation.org_id.in_([PARTITION_A, PARTITION_B])))
        db.exec(delete(Project).where(Project.org_id.in_([PARTITION_A, PARTITION_B])))
        db.commit()
    get_app_settings.cache_clear()


@pytest.fixture
def scoped(stores):
    """One request session per scope, as production opens one per request."""
    sessions: dict[TenantScope, ScopedSession] = {}

    def for_scope(scope: TenantScope) -> ScopedSession:
        if scope not in sessions:
            sessions[scope] = ScopedSession(get_open_session(), scope)
        return sessions[scope]

    try:
        yield for_scope
    finally:
        for session in sessions.values():
            session.close()


@pytest.fixture
def trusted(personal_deployment, stores):
    """Every resource personal, no managed folder."""
    get_app_settings.cache_clear()
    return stores


@pytest.fixture
def sharing(personal_deployment, managed_dir, stores):
    """Every resource personal, with a managed folder, and a shared-skill
    list that lists Alice's skill `team-report`, which she created herself."""
    get_app_settings.cache_clear()
    SkillService(ALICE).create_skill(SHARED, SHARED_BODY)
    _write_list([_entry(sub=SUB_A, partition=PARTITION_A, skill=SHARED)])
    return managed_dir


@pytest.fixture
def all_shared(personal_deployment, stores, monkeypatch):
    """Skills left off COWORK_PERSONAL_RESOURCES: one base store."""
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(SHARED_RESOURCES)))
    get_app_settings.cache_clear()
    return stores


def _entry(*, sub: str, partition: str, skill: str) -> dict[str, str]:
    return {"user_id": sub, "partition_id": partition, "skill": skill}


def _write_list(entries: list[dict[str, str]]) -> None:
    path = shared_skills.shared_skills_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))


def _list_entries() -> list[dict[str, str]]:
    path = shared_skills.shared_skills_file()
    return json.loads(path.read_text()) if path.exists() else []


def _skill_md(slug: str, body: str, *, created_by: str | None = None) -> str:
    metadata = f"metadata:\n  created_by: {created_by}\n" if created_by else ""
    return f"---\nname: {slug}\ndescription: {slug} skill\n{metadata}---\n{body}\n"


def _zip(slug: str, body: str, *, created_by: str | None = None) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as packed:
        packed.writestr(f"{slug}/SKILL.md", _skill_md(slug, body, created_by=created_by))
    return archive.getvalue()


def _stored(scope: TenantScope, slug: str) -> dict[str, str]:
    return SkillService(scope).get_skill(slug).metadata


def _listed(scoped, scope: TenantScope) -> dict[str, dict]:
    return {skill["id"]: skill for skill in skills_ep.list_skills(scoped(scope), None)["skills"]}


def _set_sharing(scoped, scope: TenantScope, principal: Principal, slug: str, *, shared: bool) -> dict:
    return skills_ep.set_skill_sharing(slug, SkillSharingRequest(shared=shared), scoped(scope), principal)


def _answers(call, status: int, detail: str) -> None:
    with pytest.raises(HTTPException) as refused:
        call()
    assert (refused.value.status_code, refused.value.detail) == (status, detail)


def _links(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    return {entry.name: entry.resolve() for entry in folder.iterdir() if entry.is_symlink()}


def _refused(call, detail: str = SHARED_SKILL_REFUSAL) -> None:
    with pytest.raises(HTTPException) as refused:
        call()
    assert (refused.value.status_code, refused.value.detail) == (403, detail)


def _name_taken_by_shared(slug: str) -> str:
    return f"Someone else shares a skill named '{slug}'. Choose another name."


def _create_refused(call, slug: str) -> None:
    """A create, upload or rename into a name someone else shares answers 409
    with a message that fits a new skill, not the shared-skill refusal."""
    with pytest.raises(HTTPException) as refused:
        call()
    assert (refused.value.status_code, refused.value.detail) == (409, _name_taken_by_shared(slug))


def _every_change_is_refused(scoped, scope: TenantScope, slug: str) -> None:
    """Every mutating skills route for ``scope`` on ``slug``, which someone
    else shares: an edit, disable, rename-away and delete answer 403 with the
    shared sentence, while a create, upload and rename into the name answer
    409 with a message that fits a new skill. The service refuses each too."""
    session = scoped(scope)
    service = SkillService(scope)
    service.create_skill("own-notes", "Mine.")
    _refused(lambda: skills_ep.update_skill(slug, SkillUpdateRequest(instructions="x"), session, None))
    _refused(lambda: skills_ep.update_skill(slug, SkillUpdateRequest(enabled=False), session, None))
    _refused(lambda: skills_ep.update_skill(slug, SkillUpdateRequest(label="taken-over"), session, None))
    _create_refused(lambda: skills_ep.update_skill("own-notes", SkillUpdateRequest(label=slug), session, None), slug)
    _create_refused(lambda: skills_ep.create_skill(SkillCreateRequest(label=slug, instructions="x"), session, None), slug)
    _create_refused(lambda: skills_ep._upload_skill_bytes(_skill_md(slug, "x").encode(), "SKILL.md", session, None), slug)
    _create_refused(lambda: skills_ep._upload_skill_bytes(_zip(slug, "x"), "skill.zip", session, None), slug)
    _refused(lambda: skills_ep.delete_skill(slug, session, None))
    with pytest.raises(SkillImmutableError, match="shared with you"):
        service.update_skill(slug, instructions="x")
    with pytest.raises(SkillImmutableError, match="shared with you"):
        service.delete_skill(slug)
    with pytest.raises(SkillAlreadyExistsError, match="Choose another name"):
        service.create_skill(slug, "x")
    service.delete_skill("own-notes")


# ── the creator stamp ───────────────────────────────────────────────────────


def test_a_create_stamps_the_person(trusted, scoped):
    created = skills_ep.create_skill(SkillCreateRequest(label="my-notes", instructions="Mine."), scoped(ALICE), None)

    assert created["id"] == "my-notes"
    assert _stored(ALICE, "my-notes")[META_CREATED_BY] == SUB_A


@pytest.mark.parametrize("filename", ["SKILL.md", "skill.zip"])
def test_an_upload_stamps_the_person_and_never_the_files_value(trusted, scoped, filename):
    raw = (
        _zip("uploaded", "Mine.", created_by=SUB_B)
        if filename.endswith(".zip")
        else _skill_md("uploaded", "Mine.", created_by=SUB_B).encode()
    )

    skills_ep._upload_skill_bytes(raw, filename, scoped(ALICE), None)

    assert _stored(ALICE, "uploaded")[META_CREATED_BY] == SUB_A


def test_every_change_keeps_the_stored_creator(trusted, scoped):
    session = scoped(ALICE)
    skills_ep.create_skill(
        SkillCreateRequest(label="my-notes", instructions="Mine.", projects=["general"]), session, None
    )

    skills_ep.update_skill("my-notes", SkillUpdateRequest(instructions="Changed."), session, None)
    skills_ep.update_skill("my-notes", SkillUpdateRequest(enabled=False), session, None)
    skills_ep.update_skill("my-notes", SkillUpdateRequest(label="renamed-notes"), session, None)
    SkillService(ALICE).replace_project_reference("general", "q4")

    metadata = _stored(ALICE, "renamed-notes")
    assert (metadata[META_CREATED_BY], metadata["projects"], metadata["enabled"]) == (SUB_A, "q4", "false")


def test_the_operator_view_never_stamps_a_creator(trusted, scoped):
    skills_ep.create_skill(SkillCreateRequest(label="ops-notes", instructions="Ours."), scoped(LOCAL_SCOPE), None)
    skills_ep._upload_skill_bytes(
        _skill_md("ops-upload", "Ours.", created_by=SUB_A).encode(), "SKILL.md", scoped(LOCAL_SCOPE), None
    )

    assert META_CREATED_BY not in _stored(LOCAL_SCOPE, "ops-notes")
    assert META_CREATED_BY not in _stored(LOCAL_SCOPE, "ops-upload")


def test_built_in_copies_record_no_creator(trusted):
    service = SkillService(ALICE)
    service.ensure_builtin_skills()
    assert all(service.get_skill(slug).created_by is None for slug in BUILTIN_SKILL_SLUGS)


# ── one skill shared ────────────────────────────────────────────────────────


def test_bob_lists_and_reads_alices_shared_skill_read_only(sharing, scoped):
    bobs = _listed(scoped, BOB)[SHARED]
    assert (bobs["isBuiltin"], bobs["shared"], bobs["capabilities"]) == (False, True, NO_CAPABILITY)
    assert bobs["declarative"].strip() == SHARED_BODY
    assert bobs["attribution"]["createdBy"] is None
    read = skills_ep.get_skill(SHARED, scoped(BOB), None)
    assert (read["shared"], read["capabilities"], read["declarative"].strip()) == (True, NO_CAPABILITY, SHARED_BODY)

    alices = _listed(scoped, ALICE)[SHARED]
    assert (alices["isBuiltin"], alices["shared"], alices["capabilities"]) == (False, True, EVERY_CAPABILITY)
    assert skills_ep.get_skill(SHARED, scoped(ALICE), None)["shared"] is True
    # Nothing was copied into Bob's own store.
    assert not (SkillService(BOB).root / SHARED).exists()


def test_a_turn_learns_the_folders_its_links_point_at(sharing, stores):
    """The turn's code sandbox grants exactly these folders read-only: Bob's
    own skills and the one Alice shares, in her own store."""
    skills_dir = stores / "bob-turn" / "skills"
    shared_dir = stores / "bob-turn" / "shared-skills"

    targets = link_turn_skills(
        scope=BOB, project_name="general", skills_dir=skills_dir, shared_skills_dir=shared_dir
    )

    linked = [*_links(skills_dir).values(), *_links(shared_dir).values()]
    assert sorted(target.resolve() for target in targets) == sorted(linked)
    assert (SkillService(ALICE).root / SHARED).resolve() in linked


def test_every_change_bob_tries_is_refused(sharing, scoped):
    before = (SkillService(ALICE).root / SHARED / "SKILL.md").read_bytes()

    _every_change_is_refused(scoped, BOB, SHARED)

    assert (SkillService(ALICE).root / SHARED / "SKILL.md").read_bytes() == before
    assert not (SkillService(BOB).root / SHARED).exists()
    assert not (SkillService(BOB).root / "taken-over").exists()


def test_a_dead_own_entry_does_not_hide_anothers_live_share(trusted, scoped):
    """A share must not slip past the live-share check when a dead own entry
    is listed ahead of someone else's live share: everyone reads the first
    entry, so a duplicate would silently shadow the live one."""
    SkillService(BOB).create_skill("x", "Bob's x.")
    alice = SkillService(ALICE)
    alice.create_skill("x", "Alice's x.")
    _write_list([
        _entry(sub=SUB_A, partition=PARTITION_A, skill="x"),
        _entry(sub=SUB_B, partition=PARTITION_B, skill="x"),
    ])

    with pytest.raises(SkillSharingError, match="Someone else already shares"):
        SkillService(ALICE)._share_new_skill("x")

    # Alice's folder is removed and Bob's live share stays first in the list.
    assert not (alice.root / "x").exists()
    assert [(e["partition_id"], e["skill"]) for e in _list_entries()] == [
        (PARTITION_A, "x"),
        (PARTITION_B, "x"),
    ]


def test_a_read_error_on_the_other_creators_folder_answers_busy(trusted, scoped, monkeypatch):
    """A transient read error on the other creator's folder must answer busy,
    not count the live skill as gone and let a duplicate share through."""
    SkillService(BOB).create_skill("x", "Bob's x.")
    alice = SkillService(ALICE)
    alice.create_skill("x", "Alice's x.")
    _write_list([_entry(sub=SUB_B, partition=PARTITION_B, skill="x")])

    real = SkillService.has_complete_skill

    def flaky(self, slug, *, strict=False):
        if slug == "x" and self._scope.org_id == PARTITION_B and strict:
            raise OSError(errno.EIO, "I/O error")
        return real(self, slug, strict=strict)

    monkeypatch.setattr(SkillService, "has_complete_skill", flaky)

    with pytest.raises(SharedSkillsBusyError):
        SkillService(ALICE)._share_new_skill("x")


def _refuse_the_list(patch: pytest.MonkeyPatch) -> None:
    """Make every read of the shared-skill list raise, as the filesystem
    would for a list it can't read, for as long as ``patch`` holds."""
    real_open_fd = managed.open_fd
    list_path = shared_skills.shared_skills_file()

    def refuse(fd_path, flags, mode=0o777, *, dir_fd=None):
        if Path(fd_path) == list_path:
            raise OSError(errno.EIO, "I/O error")
        return real_open_fd(fd_path, flags, mode, dir_fd=dir_fd)

    patch.setattr(managed, "open_fd", refuse)


def test_a_stale_entry_cant_resurrect_while_the_list_is_unreadable(trusted, scoped, monkeypatch, owned_logger):
    """A delete that strands an entry, then a recreate while the list can't be
    read, must answer busy and write nothing: the stale entry can never
    silently re-share the new skill."""
    skills_ep.create_skill(SkillCreateRequest(label="x", instructions="first", shared=True), scoped(ALICE), None)
    assert [e["skill"] for e in _list_entries()] == ["x"]

    # The delete's unshare write fails, so the delete succeeds best-effort and
    # the dead entry stays.
    real_change = shared_skills.change_shared_skills
    failing = {"on": True}

    def maybe_fail(*, change):
        if failing["on"]:
            raise OSError(errno.EIO, "I/O error")
        return real_change(change=change)

    monkeypatch.setattr(shared_skills, "change_shared_skills", maybe_fail)
    owned_logger("cowork.services.skills", level=logging.WARNING)
    skills_ep.delete_skill("x", scoped(ALICE), None)
    failing["on"] = False
    assert not (SkillService(ALICE).root / "x").exists()
    assert [e["skill"] for e in _list_entries()] == ["x"]

    # The list now can't be read. Recreating X must answer busy, not write it.
    with monkeypatch.context() as patch:
        _refuse_the_list(patch)
        with pytest.raises(HTTPException) as busy:
            skills_ep.create_skill(SkillCreateRequest(label="x", instructions="second"), scoped(ALICE), None)
    assert busy.value.status_code == 503
    assert not (SkillService(ALICE).root / "x").exists()

    # Reads recover. The stale entry still names a gone skill, so nobody reads it.
    assert [e["skill"] for e in _list_entries()] == ["x"]
    assert "x" not in _listed(scoped, BOB)
    assert not (SkillService(ALICE).root / "x").exists()


def test_a_delete_while_the_list_is_unreadable_logs_the_entry_it_may_leave(
    trusted, scoped, monkeypatch, caplog, owned_logger
):
    """Deleting a shared skill while the list can't be read still deletes it,
    and attempts the unshare whatever the forgiving cached read said: the
    failure is logged naming the file, never skipped silently."""
    skills_ep.create_skill(SkillCreateRequest(label="x", instructions="first", shared=True), scoped(ALICE), None)
    owned_logger("cowork.services.skills", level=logging.WARNING)

    with monkeypatch.context() as patch:
        _refuse_the_list(patch)
        skills_ep.delete_skill("x", scoped(ALICE), None)

    assert not (SkillService(ALICE).root / "x").exists()
    list_file = str(shared_skills.shared_skills_file())
    assert any(
        "couldn't confirm deleted skill x is no longer listed as shared" in record.getMessage()
        and list_file in record.getMessage()
        for record in caplog.records
        if record.name == "cowork.services.skills" and record.levelno == logging.WARNING
    )


def test_stop_sharing_sees_another_writers_change_to_the_list(trusted):
    """An operator's unshare between a request's first read of the list and
    its own stop_sharing leaves nothing to remove there; the skill then reads
    as not shared, never as the request's earlier read had it."""
    SkillService(ALICE).create_skill("x", "Alice's x.")
    SkillService(ALICE).share("x")
    service = SkillService(ALICE)
    assert "x" in service.shared_by_owner()

    _write_list([])

    assert service.stop_sharing("x") is False
    assert "x" not in service.shared_by_owner()


def test_renaming_a_shared_skill_while_the_list_is_unreadable_answers_busy(trusted, scoped, monkeypatch, owned_logger):
    skills_ep.create_skill(SkillCreateRequest(label="x", instructions="first", shared=True), scoped(ALICE), None)
    _refuse_the_list(monkeypatch)
    owned_logger("cowork.shared_skills", level=logging.WARNING)

    with pytest.raises(HTTPException) as busy:
        skills_ep.update_skill("x", SkillUpdateRequest(label="x-v2"), scoped(ALICE), None)

    assert busy.value.status_code == 503
    # Nothing is half-renamed: the old name stays, the new one is never written.
    assert (SkillService(ALICE).root / "x").exists()
    assert not (SkillService(ALICE).root / "x-v2").exists()


def test_alice_changes_her_shared_skill_and_bob_sees_each_change(sharing, scoped, stores):
    alice = scoped(ALICE)
    shared_dir = stores / "bob-turn" / "shared-skills"

    edited = skills_ep.update_skill(SHARED, SkillUpdateRequest(instructions="The new way."), alice, None)
    assert (edited["shared"], edited["capabilities"]) == (True, EVERY_CAPABILITY)
    assert _listed(scoped, BOB)[SHARED]["declarative"] == "The new way."
    link_turn_skills(scope=BOB, project_name="general", skills_dir=stores / "bob-turn" / "skills", shared_skills_dir=shared_dir)
    assert (shared_dir / SHARED / "SKILL.md").read_text().endswith("The new way.")

    # Disabling lives in the skill's own SKILL.md, so it reaches everyone.
    skills_ep.update_skill(SHARED, SkillUpdateRequest(enabled=False), alice, None)
    assert _listed(scoped, BOB)[SHARED]["enabled"] is False
    link_turn_skills(scope=BOB, project_name="general", skills_dir=stores / "bob-turn" / "skills", shared_skills_dir=shared_dir)
    assert SHARED not in _links(shared_dir)

    skills_ep.delete_skill(SHARED, alice, None)
    assert SHARED not in _listed(scoped, BOB)
    with pytest.raises(HTTPException) as gone:
        skills_ep.get_skill(SHARED, scoped(BOB), None)
    assert gone.value.status_code == 404
    # The name is free for Bob again: nothing shares it any more.
    created = skills_ep.create_skill(SkillCreateRequest(label=SHARED, instructions="Bob's."), scoped(BOB), None)
    assert (created["shared"], created["capabilities"]) == (False, EVERY_CAPABILITY)


def test_unsharing_removes_it_from_bob_and_alice_keeps_it(sharing, scoped):
    _write_list([])

    assert SHARED not in _listed(scoped, BOB)
    alices = _listed(scoped, ALICE)[SHARED]
    assert (alices["shared"], alices["capabilities"]) == (False, EVERY_CAPABILITY)


def test_alice_renames_her_shared_skill_only_after_unsharing_it(sharing, scoped):
    """The list names a skill by slug, so a rename while shared would
    silently stop sharing it: it is refused until she hides it herself."""
    alice = scoped(ALICE)

    _refused(
        lambda: skills_ep.update_skill(SHARED, SkillUpdateRequest(label="team-report-v2"), alice, None),
        SHARED_SKILL_RENAME_REFUSAL,
    )
    with pytest.raises(SkillImmutableError, match="Stop sharing"):
        SkillService(ALICE).update_skill(SHARED, label="team-report-v2")
    assert SHARED in _listed(scoped, BOB)
    # A label that keeps the slug is no rename.
    kept = skills_ep.update_skill(SHARED, SkillUpdateRequest(label=SHARED, name="Team Report"), alice, None)
    assert (kept["id"], kept["name"]) == (SHARED, "Team Report")

    _set_sharing(scoped, ALICE, ALICE_PRINCIPAL, SHARED, shared=False)
    renamed = skills_ep.update_skill(SHARED, SkillUpdateRequest(label="team-report-v2"), alice, None)
    assert renamed["id"] == "team-report-v2"


@pytest.mark.parametrize("way", ["create", "upload", "upload-zip", "rename"])
def test_a_new_skill_under_a_gone_shared_name_clears_the_leftover_entry(sharing, scoped, way):
    """Alice's shared skill is gone, but the list still has her entry for its
    name: her next skill there, made with Only me, removes the entry first,
    so it is never shared, and Bob never lists it."""
    SkillService._rmtree_direct_child(SkillService(ALICE).root, SHARED)
    SkillService(ALICE).create_skill("private-notes", "Alice's private notes, never shared.")
    alice = scoped(ALICE)
    make = {
        "create": lambda: skills_ep.create_skill(SkillCreateRequest(label=SHARED, instructions="x"), alice, None),
        "upload": lambda: skills_ep._upload_skill_bytes(_skill_md(SHARED, "x").encode(), "SKILL.md", alice, None),
        "upload-zip": lambda: skills_ep._upload_skill_bytes(_zip(SHARED, "x"), "skill.zip", alice, None),
        "rename": lambda: skills_ep.update_skill("private-notes", SkillUpdateRequest(label=SHARED), alice, None),
    }[way]

    made = make()

    assert (made["id"], made["shared"]) == (SHARED, False)
    assert _list_entries() == []
    assert SHARED not in _listed(scoped, BOB)
    assert _listed(scoped, ALICE)[SHARED]["shared"] is False


def test_bobs_turn_links_alices_shared_skill_and_not_her_unshared_one(sharing, stores):
    SkillService(ALICE).create_skill("alices-private", "Alice's alone.")
    skills_dir = stores / "bob-turn" / "skills"
    shared_dir = stores / "bob-turn" / "shared-skills"

    link_turn_skills(scope=BOB, project_name="general", skills_dir=skills_dir, shared_skills_dir=shared_dir)

    alice_store = SkillService(ALICE).root.resolve()
    assert _links(shared_dir) == {SHARED: alice_store / SHARED}
    own = _links(skills_dir)
    assert SHARED not in own and "alices-private" not in own
    assert {target.parent for target in own.values()} == {SkillService(BOB).root.resolve()}


def test_a_shared_skill_follows_its_own_project_scope(sharing, stores):
    SkillService(ALICE).update_skill(SHARED, projects=["reports"])
    shared_dir = stores / "bob-turn" / "shared-skills"

    for project, linked in (("general", False), ("reports", True)):
        link_turn_skills(
            scope=BOB, project_name=project, skills_dir=stores / "bob-turn" / "skills", shared_skills_dir=shared_dir
        )
        assert (SHARED in _links(shared_dir)) is linked


async def test_bobs_turn_reads_shared_skills_from_its_last_read_only_root(sharing, monkeypatch, stores):
    SkillService(ALICE).create_skill("alices-private", "Alice's alone.")

    async with built_turn(monkeypatch=monkeypatch, tmp_path=stores, scope=BOB) as turn:
        roots = turn.config.settings.skills_extra_roots
        assert roots == [HOST_SKILLS_DIR, turn.scratch / "managed-skills", turn.scratch / "shared-skills"]
        assert _links(roots[-1]) == {SHARED: SkillService(ALICE).root.resolve() / SHARED}
        own = _links(turn.config.settings.skills_root)
        assert SHARED not in own and "alices-private" not in own

    async with built_turn(monkeypatch=monkeypatch, tmp_path=stores, scope=ALICE) as turn:
        # Alice's own skill reaches her turn from her store, as before.
        assert SHARED in _links(turn.config.settings.skills_root)
        assert _links(turn.config.settings.skills_extra_roots[-1]) == {}


def test_an_earlier_layer_hides_a_later_same_named_one(sharing, scoped):
    bob = SkillService(BOB)
    # Made before the share: a create after it is refused, the name being
    # Alice's shared skill's.
    (bob.root / SHARED).mkdir(parents=True)
    (bob.root / SHARED / "SKILL.md").write_text(_skill_md(SHARED, "Bob's own version."))
    builtin = sorted(BUILTIN_SKILL_SLUGS)[0]
    SkillService(ALICE).create_skill("operator-name", "Alice's.")
    (sharing.skills_dir / "operator-name").mkdir()
    (sharing.skills_dir / "operator-name" / "SKILL.md").write_text(_skill_md("operator-name", "The operator's."))
    _write_list(
        [
            _entry(sub=SUB_A, partition=PARTITION_A, skill=SHARED),
            _entry(sub=SUB_A, partition=PARTITION_A, skill="operator-name"),
            _entry(sub=SUB_A, partition=PARTITION_A, skill=builtin),
        ],
    )

    listed = _listed(scoped, BOB)
    # Bob's own hides Alice's, and stays his to change.
    assert listed[SHARED]["declarative"].strip() == "Bob's own version."
    assert (listed[SHARED]["shared"], listed[SHARED]["capabilities"]) == (False, EVERY_CAPABILITY)
    # The operator's managed skill hides Alice's.
    assert listed["operator-name"]["declarative"].strip() == "The operator's."
    assert (listed["operator-name"]["isBuiltin"], listed["operator-name"]["shared"]) == (True, False)
    assert listed["operator-name"]["capabilities"] == NO_CAPABILITY
    # A packaged built-in's name is never someone else's.
    assert listed[builtin]["shared"] is False
    assert bob.shared_with_owner() == {}


@pytest.mark.parametrize("root", ["anton", "host"])
def test_a_skill_the_agent_already_has_is_never_shared(sharing, scoped, stores, root):
    """anton reads its packaged skills after the shared ones and the
    harness's own before them, so a shared skill of either name would
    replace anton's in everyone's turns or reach nobody. A registry entry for
    one shares nothing, and Bob still makes his own of that name."""
    from anton.core.memory.skills import SkillStore

    slug = sorted(readonly_skill_dirs(SkillStore().builtin_root if root == "anton" else HOST_SKILLS_DIR).readable)[0]
    assert slug in agent_skill_names()
    SkillService(ALICE).create_skill(slug, "Alice's version, for everyone.")
    _write_list([_entry(sub=SUB_A, partition=PARTITION_A, skill=slug)])
    shared_dir = stores / "bob-turn" / "shared-skills"

    assert slug not in SkillService(BOB).shared_with_owner()
    assert slug not in _listed(scoped, BOB)
    link_turn_skills(
        scope=BOB, project_name="general", skills_dir=stores / "bob-turn" / "skills", shared_skills_dir=shared_dir
    )
    assert slug not in _links(shared_dir)
    created = skills_ep.create_skill(SkillCreateRequest(label=slug, instructions="Bob's."), scoped(BOB), None)
    assert (created["id"], created["shared"]) == (slug, False)


def test_only_list_entries_of_existing_skills_are_read(sharing, scoped):
    SkillService(ALICE).create_skill("alices-private", "Alice's alone.")
    _write_list(
        [
            _entry(sub=SUB_A, partition=PARTITION_A, skill=SHARED),
            _entry(sub=SUB_A, partition=PARTITION_A, skill="never-made"),
        ],
    )

    assert set(SkillService(BOB).shared_with_owner()) == {SHARED}
    listed = _listed(scoped, BOB)
    assert "never-made" not in listed and "alices-private" not in listed
    created = skills_ep.create_skill(SkillCreateRequest(label="never-made", instructions="Bob's."), scoped(BOB), None)
    assert created["id"] == "never-made"


def test_one_rename_reads_the_list_once(sharing, scoped, monkeypatch):
    """One request reads ``shared/skills.json`` once, however many of its
    checks ask what is shared (SkillService._sharing)."""
    SkillService(ALICE).create_skill("notes", "Alice's notes.")
    reads: list[Path] = []
    read = shared_skills.read_shared_skills

    def counted():
        reads.append(shared_skills.shared_skills_file())
        return read()

    monkeypatch.setattr(shared_skills, "read_shared_skills", counted)

    renamed = skills_ep.update_skill("notes", SkillUpdateRequest(label="notes-v2"), scoped(ALICE), None)

    assert renamed["id"] == "notes-v2"
    assert reads == [shared_skills.shared_skills_file()]


def test_a_malformed_list_shares_nothing_and_says_so(sharing, scoped, caplog, owned_logger):
    shared_skills.shared_skills_file().write_text('{"not": "a list"}')
    owned_logger("cowork.shared_skills", level=logging.WARNING)

    assert SHARED not in _listed(scoped, BOB)
    assert _listed(scoped, ALICE)[SHARED]["shared"] is False

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "cowork.shared_skills" and record.levelno == logging.WARNING
    ]
    assert warnings and str(shared_skills.shared_skills_file()) in warnings[0]


def test_only_a_scope_with_personal_skills_reads_the_list(sharing):
    """Where skills are shared the base store already holds everyone's: a
    person's scope that keeps skills shared, and the operator view, read no
    list."""
    bob = _person(partition=PARTITION_B, sub=SUB_B, resources=SHARED_RESOURCES)
    assert SkillService(bob).shared_with_owner() == {}
    assert SkillService(LOCAL_SCOPE).shared_with_owner() == {}


# ── a person shares or hides their own skill ───────────────────────────────


def _upload(scoped, scope: TenantScope, slug: str, body: str, *, zipped: bool, shared: bool) -> dict:
    raw = _zip(slug, body) if zipped else _skill_md(slug, body).encode()
    return skills_ep._upload_skill_bytes(
        raw, "skill.zip" if zipped else "SKILL.md", scoped(scope), None, shared=shared
    )


def test_alice_shares_her_skill_and_hides_it_again(trusted, scoped):
    """No managed folder: the list sits in the shared root."""
    SkillService(ALICE).create_skill(SHARED, SHARED_BODY)
    listing = skills_ep.list_skills(scoped(ALICE), None)
    listed = {skill["id"]: skill for skill in listing["skills"]}
    assert listing["canShare"] is True
    assert listed[SHARED]["capabilities"] == EVERY_CAPABILITY
    # A packaged built-in's copy is everyone's already.
    assert not any(listed[slug]["capabilities"]["canShare"] for slug in BUILTIN_SKILL_SLUGS)

    shared = _set_sharing(scoped, ALICE, ALICE_PRINCIPAL, SHARED, shared=True)

    assert (shared["shared"], shared["capabilities"]) == (True, EVERY_CAPABILITY)
    assert _list_entries() == [_entry(sub=SUB_A, partition=PARTITION_A, skill=SHARED)]
    bobs = _listed(scoped, BOB)[SHARED]
    assert (bobs["shared"], bobs["capabilities"], bobs["declarative"].strip()) == (True, NO_CAPABILITY, SHARED_BODY)

    hidden = _set_sharing(scoped, ALICE, ALICE_PRINCIPAL, SHARED, shared=False)

    assert (hidden["shared"], hidden["capabilities"]) == (False, EVERY_CAPABILITY)
    assert _list_entries() == []
    assert SHARED not in _listed(scoped, BOB)
    assert (SkillService(ALICE).root / SHARED / "SKILL.md").is_file()


def test_sharing_or_hiding_twice_writes_nothing(trusted, scoped):
    SkillService(ALICE).create_skill(SHARED, SHARED_BODY)
    path = shared_skills.shared_skills_file()

    _set_sharing(scoped, ALICE, ALICE_PRINCIPAL, SHARED, shared=True)
    written = path.stat().st_ino
    assert _set_sharing(scoped, ALICE, ALICE_PRINCIPAL, SHARED, shared=True)["shared"] is True
    assert path.stat().st_ino == written

    _set_sharing(scoped, ALICE, ALICE_PRINCIPAL, SHARED, shared=False)
    written = path.stat().st_ino
    assert _set_sharing(scoped, ALICE, ALICE_PRINCIPAL, SHARED, shared=False)["shared"] is False
    assert path.stat().st_ino == written


def test_bob_shares_neither_alices_shared_skill_nor_her_private_one(sharing, scoped):
    SkillService(ALICE).create_skill("alices-private", "Alice's alone.")
    before = _list_entries()

    for shared in (True, False):
        _answers(lambda: _set_sharing(scoped, BOB, BOB_PRINCIPAL, SHARED, shared=shared), 403, SHARED_SKILL_REFUSAL)
        _answers(
            lambda: _set_sharing(scoped, BOB, BOB_PRINCIPAL, "alices-private", shared=shared),
            404,
            "Skill 'alices-private' not found.",
        )
    with pytest.raises(HTTPException) as invalid:
        _set_sharing(scoped, BOB, BOB_PRINCIPAL, "Not A Slug!", shared=True)
    assert invalid.value.status_code == 404
    assert _list_entries() == before


def test_a_managed_skills_name_answers_403(sharing, scoped):
    (sharing.skills_dir / "operator-name").mkdir()
    (sharing.skills_dir / "operator-name" / "SKILL.md").write_text(_skill_md("operator-name", "The operator's."))

    _answers(
        lambda: _set_sharing(scoped, ALICE, ALICE_PRINCIPAL, "operator-name", shared=True), 403, SHARED_SKILL_REFUSAL
    )


def test_each_share_check_answers_409_saying_why(sharing, scoped, tmp_path):
    from anton.core.memory.skills import SkillStore

    alice = SkillService(ALICE)
    alice.ensure_builtin_skills()
    builtin = sorted(BUILTIN_SKILL_SLUGS)[0]
    agent_name = sorted(readonly_skill_dirs(SkillStore().builtin_root).readable)[0]
    alice.create_skill(agent_name, "Alice's own version.")
    alice.create_skill("linked", "Holds a link.")
    secret = tmp_path / "secret.txt"
    secret.write_text("not for everyone")
    (alice.root / "linked" / "refs").mkdir()
    (alice.root / "linked" / "refs" / "notes.txt").symlink_to(secret)
    alice.create_skill("drifted", "Its name drifted.")
    skill_file = alice.root / "drifted" / "SKILL.md"
    skill_file.write_text(skill_file.read_text().replace("name: drifted", "name: something-else"))
    # Made before Alice shared hers: a create after it is refused.
    bobs = SkillService(BOB).root / SHARED
    bobs.mkdir(parents=True)
    (bobs / "SKILL.md").write_text(_skill_md(SHARED, "Bob's own version."))

    for scope, principal, slug, detail in (
        (ALICE, ALICE_PRINCIPAL, builtin, f"{builtin} is a built-in skill that everyone already has, so it can't be shared."),
        (
            ALICE,
            ALICE_PRINCIPAL,
            agent_name,
            f"Everyone's agent already has a skill named {agent_name}, so sharing yours would reach nobody. "
            "Rename your skill before you share it.",
        ),
        (ALICE, ALICE_PRINCIPAL, "linked", "This skill holds a link, refs/notes.txt. Remove it before you share the skill."),
        (ALICE, ALICE_PRINCIPAL, "drifted", "Skill drifted can't be read in full, so it can't be shared."),
        (
            BOB,
            BOB_PRINCIPAL,
            SHARED,
            f"Someone else already shares a skill named {SHARED}. Rename your skill before you share it.",
        ),
    ):
        _answers(lambda: _set_sharing(scoped, scope, principal, slug, shared=True), 409, detail)
    assert _list_entries() == [_entry(sub=SUB_A, partition=PARTITION_A, skill=SHARED)]


def test_someone_elses_entry_for_a_gone_skill_never_blocks_a_share(sharing, scoped):
    SkillService._rmtree_direct_child(SkillService(ALICE).root, SHARED)
    SkillService(BOB).create_skill(SHARED, "Bob's own version.")

    shared = _set_sharing(scoped, BOB, BOB_PRINCIPAL, SHARED, shared=True)

    assert shared["shared"] is True
    assert _listed(scoped, ALICE)[SHARED]["declarative"].strip() == "Bob's own version."


def test_a_busy_list_answers_503_and_writes_nothing(trusted, scoped, monkeypatch):
    monkeypatch.setattr(shared_skills, "SHARED_SKILLS_LOCK_TIMEOUT", 0.2)
    SkillService(ALICE).create_skill(SHARED, SHARED_BODY)
    SkillService(ALICE).create_skill("private-notes", "Alice's.")
    _write_list([_entry(sub=SUB_A, partition=PARTITION_A, skill="gone-name")])
    alice = scoped(ALICE)

    with shared_skill_list_lock_held():
        for shared in (True, False):
            _answers(lambda: _set_sharing(scoped, ALICE, ALICE_PRINCIPAL, SHARED, shared=shared), 503, SKILL_SHARING_BUSY)
        _answers(
            lambda: skills_ep.create_skill(SkillCreateRequest(label="busy-notes", instructions="x", shared=True), alice, None),
            503,
            SKILL_SHARING_BUSY,
        )
        _answers(lambda: _upload(scoped, ALICE, "busy-upload", "x", zipped=True, shared=True), 503, SKILL_SHARING_BUSY)
        # A rename into a name her list still has for a gone skill must
        # clear that entry first.
        _answers(
            lambda: skills_ep.update_skill("private-notes", SkillUpdateRequest(label="gone-name"), alice, None),
            503,
            SKILL_SHARING_BUSY,
        )

    root = SkillService(ALICE).root
    assert not (root / "busy-notes").exists() and not (root / "busy-upload").exists()
    assert (root / "private-notes").is_dir() and not (root / "gone-name").exists()
    assert _list_entries() == [_entry(sub=SUB_A, partition=PARTITION_A, skill="gone-name")]


def test_deleting_a_shared_skill_stops_sharing_it(sharing, scoped):
    skills_ep.delete_skill(SHARED, scoped(ALICE), None)

    assert _list_entries() == []
    created = skills_ep.create_skill(SkillCreateRequest(label=SHARED, instructions="New."), scoped(ALICE), None)
    assert created["shared"] is False
    assert SHARED not in _listed(scoped, BOB)


def test_a_delete_still_succeeds_when_its_entry_cant_be_removed(sharing, scoped, monkeypatch, caplog, owned_logger):
    def broken(*, change):
        raise OSError("the list can't be written")

    monkeypatch.setattr(shared_skills, "change_shared_skills", broken)
    owned_logger("cowork.services.skills", level=logging.WARNING)

    skills_ep.delete_skill(SHARED, scoped(ALICE), None)

    assert not (SkillService(ALICE).root / SHARED).exists()
    warnings = [record.getMessage() for record in caplog.records if record.name == "cowork.services.skills"]
    assert any(SHARED in line and str(shared_skills.shared_skills_file()) in line for line in warnings)
    # The entry left behind shares nothing: its skill is gone.
    assert SHARED not in _listed(scoped, BOB)


async def test_a_turn_reads_shared_skills_without_a_managed_folder(trusted, monkeypatch, stores):
    SkillService(ALICE).create_skill(SHARED, SHARED_BODY)
    SkillService(ALICE).share(SHARED)

    async with built_turn(monkeypatch=monkeypatch, tmp_path=stores, scope=BOB) as turn:
        roots = turn.config.settings.skills_extra_roots
        assert roots == [HOST_SKILLS_DIR, turn.scratch / "shared-skills"]
        assert _links(roots[-1]) == {SHARED: SkillService(ALICE).root.resolve() / SHARED}


# ── sharing a skill as it is created ────────────────────────────────────────


@pytest.mark.parametrize("way", ["create", "upload", "upload-zip"])
def test_a_skill_created_shared_reaches_bob_read_only(trusted, scoped, way):
    if way == "create":
        created = skills_ep.create_skill(
            SkillCreateRequest(label=SHARED, instructions=SHARED_BODY, shared=True), scoped(ALICE), None
        )
    else:
        created = _upload(scoped, ALICE, SHARED, SHARED_BODY, zipped=way == "upload-zip", shared=True)

    assert (created["shared"], created["capabilities"]) == (True, EVERY_CAPABILITY)
    assert _list_entries() == [_entry(sub=SUB_A, partition=PARTITION_A, skill=SHARED)]
    alices = _listed(scoped, ALICE)[SHARED]
    assert (alices["shared"], alices["capabilities"]) == (True, EVERY_CAPABILITY)
    bobs = _listed(scoped, BOB)[SHARED]
    assert (bobs["shared"], bobs["capabilities"], bobs["declarative"].strip()) == (True, NO_CAPABILITY, SHARED_BODY)


def test_a_skill_created_with_only_me_is_shared_with_nobody(trusted, scoped):
    created = skills_ep.create_skill(SkillCreateRequest(label=SHARED, instructions=SHARED_BODY), scoped(ALICE), None)

    assert (created["shared"], created["capabilities"]) == (False, EVERY_CAPABILITY)
    assert not shared_skills.shared_skills_file().exists()
    assert SHARED not in _listed(scoped, BOB)


@pytest.mark.parametrize("way", ["create", "upload", "upload-zip"])
def test_a_skill_created_shared_under_an_agent_skill_name_answers_409(trusted, scoped, way):
    from anton.core.memory.skills import SkillStore

    slug = sorted(readonly_skill_dirs(SkillStore().builtin_root).readable)[0]
    detail = (
        f"Everyone's agent already has a skill named {slug}, so sharing yours would reach nobody. "
        "Rename your skill before you share it."
    )

    def make() -> dict:
        if way == "create":
            return skills_ep.create_skill(
                SkillCreateRequest(label=slug, instructions="x", shared=True), scoped(ALICE), None
            )
        return _upload(scoped, ALICE, slug, "x", zipped=way == "upload-zip", shared=True)

    _answers(make, 409, detail)

    assert not (SkillService(ALICE).root / slug).exists()
    assert not shared_skills.shared_skills_file().exists()


@pytest.mark.parametrize("way", ["create", "upload", "upload-zip"])
def test_a_skill_created_shared_is_removed_when_its_entry_cant_be_written(trusted, scoped, monkeypatch, way):
    def broken(*, change):
        raise OSError("the list can't be written")

    monkeypatch.setattr(shared_skills, "change_shared_skills", broken)

    with pytest.raises(OSError, match="the list can't be written"):
        if way == "create":
            skills_ep.create_skill(SkillCreateRequest(label="new-shared", instructions="x", shared=True), scoped(ALICE), None)
        else:
            _upload(scoped, ALICE, "new-shared", "x", zipped=way == "upload-zip", shared=True)

    assert not (SkillService(ALICE).root / "new-shared").exists()


def test_where_skills_are_shared_nobody_shares_one_at_a_time(all_shared, scoped):
    alice = scoped(ALICE_SHARED)
    skills_ep.create_skill(SkillCreateRequest(label="team-notes", instructions="Alice's."), alice, None)
    listing = skills_ep.list_skills(alice, None)
    assert listing["canShare"] is False
    assert not any(skill["capabilities"]["canShare"] for skill in listing["skills"])

    # The 409's text, which the Cowork toast shows and the README quotes.
    assert SKILL_SHARING_UNAVAILABLE == "Sharing one skill at a time isn't available here."
    # Unavailable answers before the skill is looked up.
    for slug in ("team-notes", "never-made"):
        for shared in (True, False):
            _answers(
                lambda: skills_ep.set_skill_sharing(slug, SkillSharingRequest(shared=shared), alice, ALICE_PRINCIPAL),
                409,
                SKILL_SHARING_UNAVAILABLE,
            )
    _answers(
        lambda: skills_ep.create_skill(SkillCreateRequest(label="new-shared", instructions="x", shared=True), alice, None),
        409,
        SKILL_SHARING_UNAVAILABLE,
    )
    _answers(lambda: _upload(scoped, ALICE_SHARED, "new-upload", "x", zipped=True, shared=True), 409, SKILL_SHARING_UNAVAILABLE)

    root = SkillService(LOCAL_SCOPE).root
    assert not (root / "new-shared").exists() and not (root / "new-upload").exists()
    assert not shared_skills.shared_skills_file().exists()


def test_the_operator_view_shares_nothing(trusted, scoped):
    # Nor does a person's partition acting with no person in it.
    no_person = TenantScope(personal=True, org_id=PARTITION_A, user_id=None, personal_resources=ALL_PERSONAL_RESOURCES)
    assert SkillService(no_person).sharing_available is False
    operator = scoped(LOCAL_SCOPE)
    SkillService(LOCAL_SCOPE).create_skill("ops-notes", "Ours.")

    listing = skills_ep.list_skills(operator, None)
    assert listing["canShare"] is False
    assert not any(skill["capabilities"]["canShare"] for skill in listing["skills"])
    _answers(
        lambda: skills_ep.create_skill(SkillCreateRequest(label="new-shared", instructions="x", shared=True), operator, None),
        409,
        SKILL_SHARING_UNAVAILABLE,
    )


# ── the sharing route on the wire ───────────────────────────────────────────


@pytest.fixture
def client(trusted, monkeypatch):
    monkeypatch.setenv("COWORK_AUTH_TOKEN", TOKEN)
    get_app_settings.cache_clear()
    return TestClient(cowork_server.create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321))


def _as(sub: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}", "X-User-Id": sub, "X-Organization-Id": DEPLOYMENT}


def test_the_sharing_route_and_create_fields_on_the_wire(client):
    SkillService(ALICE).create_skill(SHARED, SHARED_BODY)
    listing = client.get("/api/v1/skills/", headers=_as(SUB_A)).json()
    assert listing["canShare"] is True
    assert {skill["id"]: skill for skill in listing["skills"]}[SHARED]["capabilities"]["canShare"] is True

    response = client.put(f"/api/v1/skills/{SHARED}/sharing", headers=_as(SUB_A), json={"shared": True})

    assert response.status_code == 200
    assert (response.json()["shared"], response.json()["capabilities"]) == (True, EVERY_CAPABILITY)
    bobs = {skill["id"]: skill for skill in client.get("/api/v1/skills/", headers=_as(SUB_B)).json()["skills"]}
    assert (bobs[SHARED]["shared"], bobs[SHARED]["capabilities"]) == (True, NO_CAPABILITY)

    for body in ({}, {"shared": "maybe"}):
        assert client.put(f"/api/v1/skills/{SHARED}/sharing", headers=_as(SUB_A), json=body).status_code == 422

    created = client.post(
        "/api/v1/skills/", headers=_as(SUB_A), json={"label": "made-shared", "declarative": "x", "shared": True}
    )
    assert (created.status_code, created.json()["shared"]) == (201, True)
    uploaded = client.post(
        "/api/v1/skills/upload",
        headers=_as(SUB_A),
        files={"file": ("SKILL.md", _skill_md("uploaded-shared", "x").encode(), "text/markdown")},
        data={"shared": "true"},
    )
    assert (uploaded.status_code, uploaded.json()["shared"]) == (201, True)
    private = client.post(
        "/api/v1/skills/upload",
        headers=_as(SUB_A),
        files={"file": ("SKILL.md", _skill_md("uploaded-private", "x").encode(), "text/markdown")},
    )
    assert (private.status_code, private.json()["shared"]) == (201, False)
    assert [entry["skill"] for entry in _list_entries()] == [SHARED, "made-shared", "uploaded-shared"]


def test_the_sharing_route_answers_401_to_a_caller_with_no_person(trusted, monkeypatch):
    """The operator view: a request with the bearer token and no identity,
    served with no person under COWORK_IDENTITY_ENFORCE=audit."""
    monkeypatch.setenv("COWORK_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    get_app_settings.cache_clear()
    client = TestClient(cowork_server.create_app(), base_url="http://127.0.0.1:26866", client=("127.0.0.1", 54321))
    SkillService(LOCAL_SCOPE).create_skill("ops-notes", "Ours.")

    response = client.put(
        "/api/v1/skills/ops-notes/sharing", headers={"Authorization": f"Bearer {TOKEN}"}, json={"shared": True}
    )

    assert (response.status_code, response.json()) == (401, {"detail": "Unauthorized"})
    assert not shared_skills.shared_skills_file().exists()


def test_create_app_starts_when_a_fifo_stands_in_for_the_list(trusted, monkeypatch):
    """A FIFO at skills.json must not block the start: create_app reads the
    list once and a non-regular file reads as empty, never hanging."""
    monkeypatch.setenv("COWORK_AUTH_TOKEN", TOKEN)
    get_app_settings.cache_clear()
    path = shared_skills.shared_skills_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(path)

    ready: list[bool] = []

    def start() -> None:
        cowork_server.create_app()
        ready.append(True)

    thread = threading.Thread(target=start, daemon=True)
    thread.start()
    thread.join(30.0)

    assert not thread.is_alive(), "create_app() blocked reading a FIFO at the shared-skill list"
    assert ready == [True]


# ── every skill shared ──────────────────────────────────────────────────────


ALICE_SHARED = _person(partition=PARTITION_A, sub=SUB_A, resources=SHARED_RESOURCES)
BOB_SHARED = _person(partition=PARTITION_B, sub=SUB_B, resources=SHARED_RESOURCES)


def _every_change_but_create_is_refused(scoped, scope: TenantScope, slug: str) -> None:
    session = scoped(scope)
    _refused(lambda: skills_ep.update_skill(slug, SkillUpdateRequest(instructions="x"), session, None))
    _refused(lambda: skills_ep.update_skill(slug, SkillUpdateRequest(enabled=False), session, None))
    _refused(lambda: skills_ep.update_skill(slug, SkillUpdateRequest(label=f"{slug}-moved"), session, None))
    _refused(lambda: skills_ep.delete_skill(slug, session, None))


def test_where_all_are_shared_only_the_creator_changes_a_skill(all_shared, scoped):
    skills_ep.create_skill(SkillCreateRequest(label="team-notes", instructions="Alice's."), scoped(ALICE_SHARED), None)
    skills_ep._upload_skill_bytes(
        _skill_md("bob-upload", "Bob's.", created_by=SUB_A).encode(), "SKILL.md", scoped(BOB_SHARED), None
    )
    assert _stored(LOCAL_SCOPE, "team-notes")[META_CREATED_BY] == SUB_A
    assert _stored(LOCAL_SCOPE, "bob-upload")[META_CREATED_BY] == SUB_B

    _every_change_but_create_is_refused(scoped, BOB_SHARED, "team-notes")
    _every_change_but_create_is_refused(scoped, ALICE_SHARED, "bob-upload")

    # Everyone lists it, and the capabilities say who can change it. Bob
    # reads Alice's as shared with him, read-only.
    for scope, capabilities, shared in (
        (BOB_SHARED, NO_CAPABILITY, True),
        (LOCAL_SCOPE, CHANGE_CAPABILITY, False),
        (ALICE_SHARED, CHANGE_CAPABILITY, False),
    ):
        listed = _listed(scoped, scope)["team-notes"]
        assert (listed["capabilities"], listed["shared"]) == (capabilities, shared)
        read = skills_ep.get_skill("team-notes", scoped(scope), None)
        assert (read["capabilities"], read["shared"]) == (capabilities, shared)

    alice = scoped(ALICE_SHARED)
    skills_ep.update_skill("team-notes", SkillUpdateRequest(instructions="Changed."), alice, None)
    renamed = skills_ep.update_skill("team-notes", SkillUpdateRequest(label="team-notes-v2"), alice, None)
    assert renamed["capabilities"] == CHANGE_CAPABILITY
    assert _stored(LOCAL_SCOPE, "team-notes-v2")[META_CREATED_BY] == SUB_A
    skills_ep.delete_skill("team-notes-v2", alice, None)
    assert "team-notes-v2" not in _listed(scoped, BOB_SHARED)


def test_where_all_are_shared_the_operator_view_changes_anyones_skill(all_shared, scoped):
    """A creator who left can't hold their skill out of the operator's reach:
    the operator view edits, disables, renames and deletes it, and the stamp
    stays the creator's."""
    skills_ep.create_skill(SkillCreateRequest(label="team-notes", instructions="Alice's."), scoped(ALICE_SHARED), None)
    operator = scoped(LOCAL_SCOPE)

    skills_ep.update_skill("team-notes", SkillUpdateRequest(instructions="The operator's."), operator, None)
    skills_ep.update_skill("team-notes", SkillUpdateRequest(enabled=False), operator, None)
    renamed = skills_ep.update_skill("team-notes", SkillUpdateRequest(label="team-notes-v2"), operator, None)
    assert (renamed["declarative"], renamed["enabled"]) == ("The operator's.", False)
    assert _stored(LOCAL_SCOPE, "team-notes-v2")[META_CREATED_BY] == SUB_A

    skills_ep.delete_skill("team-notes-v2", operator, None)
    assert "team-notes-v2" not in _listed(scoped, ALICE_SHARED)


def test_where_all_are_shared_a_skill_with_no_creator_is_the_operators(all_shared, scoped):
    SkillService(LOCAL_SCOPE).create_skill("legacy", "Made before creators were recorded.")
    assert META_CREATED_BY not in _stored(LOCAL_SCOPE, "legacy")

    _every_change_but_create_is_refused(scoped, ALICE_SHARED, "legacy")
    _every_change_but_create_is_refused(scoped, BOB_SHARED, "legacy")

    operator = scoped(LOCAL_SCOPE)
    edited = skills_ep.update_skill("legacy", SkillUpdateRequest(instructions="The operator's."), operator, None)
    assert edited["capabilities"] == CHANGE_CAPABILITY
    assert META_CREATED_BY not in _stored(LOCAL_SCOPE, "legacy")
    skills_ep.delete_skill("legacy", operator, None)
    assert "legacy" not in _listed(scoped, LOCAL_SCOPE)


# ── the desktop and hosted org tenancy ─────────────────────────────────────


def _every_route_works(session, principal, *, upload_creator: str | None) -> None:
    """Create, upload, list, read, edit, disable, rename and delete one skill
    each, as the caller who made them. None is ``shared``, and nothing stamps
    a creator: an upload keeps what its file says (``upload_creator``)."""
    created = skills_ep.create_skill(SkillCreateRequest(label="my-notes", instructions="Mine."), session, principal)
    assert (created["shared"], created["capabilities"]) == (False, CHANGE_CAPABILITY)
    uploaded = skills_ep._upload_skill_bytes(
        _skill_md("uploaded", "Mine.", created_by=SUB_B).encode(), "SKILL.md", session, principal
    )
    assert uploaded["shared"] is False
    service = SkillService(session.scope)
    assert service.get_skill("my-notes").created_by is None
    assert service.get_skill("uploaded").created_by == upload_creator

    listing = skills_ep.list_skills(session, principal)
    assert listing["canShare"] is False
    listed = {s["id"]: s for s in listing["skills"]}
    assert {"my-notes", "uploaded"} <= set(listed)
    assert not any(skill["shared"] or skill["capabilities"]["canShare"] for skill in listed.values())
    assert skills_ep.get_skill("my-notes", session, principal)["shared"] is False

    edited = skills_ep.update_skill("my-notes", SkillUpdateRequest(instructions="Changed."), session, principal)
    assert edited["declarative"] == "Changed."
    disabled = skills_ep.update_skill("my-notes", SkillUpdateRequest(enabled=False), session, principal)
    assert disabled["enabled"] is False
    renamed = skills_ep.update_skill("my-notes", SkillUpdateRequest(label="my-renamed"), session, principal)
    assert (renamed["id"], renamed["shared"]) == ("my-renamed", False)
    skills_ep.delete_skill("my-renamed", session, principal)
    skills_ep.delete_skill("uploaded", session, principal)
    remaining = {s["id"] for s in skills_ep.list_skills(session, principal)["skills"]}
    assert not {"my-notes", "my-renamed", "uploaded"} & remaining


def test_the_desktop_keeps_every_skills_route(monkeypatch, stores, scoped):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    monkeypatch.delenv("COWORK_MANAGED_DIR", raising=False)
    get_app_settings.cache_clear()

    _every_route_works(scoped(LOCAL_SCOPE), None, upload_creator=SUB_B)


ORG_ID = "33333333-3333-4333-8333-333333333333"
ORG_USER = "44444444-4444-4444-8444-444444444444"


@pytest.fixture
def hosted(org_deployment, monkeypatch, tmp_path):
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    monkeypatch.setenv("COWORK_MEMORY_DIR", str(tmp_path / "memory"))
    get_app_settings.cache_clear()
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    session = ScopedSession(Session(engine), TenantScope(org_mode=True, org_id=ORG_ID, user_id=ORG_USER))
    try:
        yield session
    finally:
        session.close()
        get_app_settings.cache_clear()


def test_hosted_org_tenancy_keeps_every_skills_route(hosted):
    principal = Principal(user_id=ORG_USER, org_id=ORG_ID, email="member@example.com")

    _every_route_works(hosted, principal, upload_creator=SUB_B)

    assert SkillService(hosted.scope).root.parent.name == ORG_ID


def test_hosted_org_tenancy_shares_no_skill(hosted):
    principal = Principal(user_id=ORG_USER, org_id=ORG_ID, email="member@example.com")
    skills_ep.create_skill(SkillCreateRequest(label="my-notes", instructions="Mine."), hosted, principal)

    _answers(
        lambda: skills_ep.set_skill_sharing("my-notes", SkillSharingRequest(shared=True), hosted, principal),
        409,
        SKILL_SHARING_UNAVAILABLE,
    )
    _answers(
        lambda: skills_ep.create_skill(SkillCreateRequest(label="new-shared", instructions="x", shared=True), hosted, principal),
        409,
        SKILL_SHARING_UNAVAILABLE,
    )
    _answers(
        lambda: skills_ep._upload_skill_bytes(_zip("new-upload", "x"), "skill.zip", hosted, principal, shared=True),
        409,
        SKILL_SHARING_UNAVAILABLE,
    )
    root = SkillService(hosted.scope).root
    assert not (root / "new-shared").exists() and not (root / "new-upload").exists()
    # The refused upload reserved no claim: the name is still free.
    uploaded = skills_ep._upload_skill_bytes(_zip("new-upload", "x"), "skill.zip", hosted, principal)
    assert (uploaded["id"], uploaded["shared"]) == ("new-upload", False)


# ── the skills an artifact build follows without a recall ───────────────────


def _mark_for_artifacts(skill_dir: Path) -> None:
    """Add `applies_to: artifacts` to a stored SKILL.md's frontmatter."""
    path = skill_dir / "SKILL.md"
    text = path.read_text()
    path.write_text(text.replace("---\n", "---\napplies_to: artifacts\n", 1))


def _followed_by_artifact_builds(settings) -> list[str]:
    from anton.core.memory.skills import SkillStore

    store = SkillStore(
        root=settings.skills_root,
        extra_roots=settings.skills_extra_roots,
        artifact_roots=settings.skills_artifact_roots,
    )
    return [skill.label for skill in store.list_for_artifacts()]


async def test_a_build_follows_bobs_marked_skills_and_never_alices_shared_one(sharing, monkeypatch, stores):
    SkillService(BOB).create_skill("bobs-charts", "Use only Chart.js.")
    _mark_for_artifacts(SkillService(BOB).root / "bobs-charts")
    _mark_for_artifacts(SkillService(ALICE).root / SHARED)

    async with built_turn(monkeypatch=monkeypatch, tmp_path=stores, scope=BOB) as turn:
        roots = [Path(p).resolve() for p in turn.config.settings.skills_artifact_roots]
        assert roots[0] == HOST_SKILLS_DIR.resolve()
        assert (SkillService(BOB).root / "bobs-charts").resolve() in roots
        alice = SkillService(ALICE).root.resolve()
        assert not [p for p in roots if p.is_relative_to(alice)]
        followed = _followed_by_artifact_builds(turn.config.settings)
        assert "bobs-charts" in followed
        assert SHARED not in followed


async def test_a_build_follows_the_operators_marked_managed_skills(sharing, monkeypatch, stores):
    (sharing.skills_dir / "gateway-rules").mkdir()
    (sharing.skills_dir / "gateway-rules" / "SKILL.md").write_text(
        _skill_md("gateway-rules", "Call only the endpoints listed here.")
    )
    _mark_for_artifacts(sharing.skills_dir / "gateway-rules")

    async with built_turn(monkeypatch=monkeypatch, tmp_path=stores, scope=BOB) as turn:
        assert "gateway-rules" in _followed_by_artifact_builds(turn.config.settings)


async def test_in_one_shared_store_a_build_follows_only_the_askers_own(all_shared, monkeypatch, stores):
    SkillService(ALICE_SHARED).create_skill("alices-charts", "Use only ECharts.")
    SkillService(BOB_SHARED).create_skill("bobs-charts", "Use only Chart.js.")
    store = SkillService(BOB_SHARED).root
    assert store == SkillService(ALICE_SHARED).root
    for slug in ("alices-charts", "bobs-charts"):
        _mark_for_artifacts(store / slug)

    async with built_turn(monkeypatch=monkeypatch, tmp_path=stores, scope=BOB_SHARED) as turn:
        followed = _followed_by_artifact_builds(turn.config.settings)
        assert "bobs-charts" in followed
        assert "alices-charts" not in followed


async def test_a_turn_with_no_person_follows_no_persons_skill(all_shared, managed_dir, monkeypatch, stores):
    """A channel's turn or a shared schedule's runs in the operator view,
    which is no operator in a turn (turn_changer_for): it owns no stored
    skill, so it follows the managed folder's and not Alice's."""
    SkillService(ALICE_SHARED).create_skill("alices-charts", "Use only ECharts.")
    store = SkillService(ALICE_SHARED).root
    _mark_for_artifacts(store / "alices-charts")
    (managed_dir.skills_dir / "gateway-rules").mkdir()
    (managed_dir.skills_dir / "gateway-rules" / "SKILL.md").write_text(
        _skill_md("gateway-rules", "Call only the endpoints listed here.")
    )
    _mark_for_artifacts(managed_dir.skills_dir / "gateway-rules")

    def link_into_project(project_dir: Path) -> None:
        # The link reconcile_project gives each project of the one store.
        (project_dir / "skills").mkdir(exist_ok=True)
        (project_dir / "skills" / "alices-charts").symlink_to(store / "alices-charts", target_is_directory=True)

    async with built_turn(
        monkeypatch=monkeypatch, tmp_path=stores, scope=LOCAL_SCOPE, before_build=link_into_project
    ) as turn:
        settings = turn.config.settings
        assert "alices-charts" in [p.name for p in Path(settings.skills_root).iterdir()]
        followed = _followed_by_artifact_builds(settings)
        assert "alices-charts" not in followed
        assert "gateway-rules" in followed
