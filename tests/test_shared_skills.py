"""Skills that record their creator, and sharing one with everyone.

On a deployment that trusts identity headers, every skill created through the
server records its creator's sub in its metadata: a create or upload stamps
the person, never a value the request or file carries, an edit, rename or
project-rename rewrite keeps it, and the operator view stamps nobody.

With skills personal, the operator shares one person's skill by listing it in
the managed folder's ``shared/skills.json``. Everyone else lists it after the
built-ins, the managed skills and their own, reads it live from its creator's
store with ``shared`` true and no capability, and every change they try,
disabling included, answers 403 with the shared sentence. Their turns link it
read-only, after the managed skills, with the selection rule applied to the
skill's own metadata. Its creator sees it ``shared`` with every capability and
edits, disables and deletes it as before, and everyone sees each change; its
rename is refused until it is unshared.

With skills shared, everyone reads the one base store. A person changes only
a skill they created, and the operator view changes any. The desktop and
hosted org tenancy keep every skills route as before.
"""
from __future__ import annotations

import io
import json
import logging
import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import delete
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

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
from cowork.schemas.skills import SkillCreateRequest, SkillUpdateRequest
from cowork.services.skills import (
    BUILTIN_SKILL_SLUGS,
    SHARED_SKILL_REFUSAL,
    SHARED_SKILL_RENAME_REFUSAL,
    STILL_SHARED_SKILL_REFUSAL,
    SkillImmutableError,
    SkillService,
    agent_skill_names,
    link_turn_skills,
)
from tests._turns import built_turn

DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB_A = "11111111-1111-4111-8111-111111111111"
SUB_B = "22222222-2222-4222-8222-222222222222"
PARTITION_A = personal_partition_id(DEPLOYMENT, SUB_A)
PARTITION_B = personal_partition_id(DEPLOYMENT, SUB_B)
SHARED_RESOURCES = frozenset(ALL_PERSONAL_RESOURCES - {PersonalResource.SKILLS})
NO_CAPABILITY = {"canEdit": False, "canDelete": False, "canDisable": False}
EVERY_CAPABILITY = {"canEdit": True, "canDelete": True, "canDisable": True}

SHARED = "team-report"
SHARED_BODY = "Write the report the team's way."


def _person(*, partition: str, sub: str, resources=ALL_PERSONAL_RESOURCES) -> TenantScope:
    return TenantScope(personal=True, org_id=partition, user_id=sub, personal_resources=resources)


ALICE = _person(partition=PARTITION_A, sub=SUB_A)
BOB = _person(partition=PARTITION_B, sub=SUB_B)


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
    """Every resource personal, with a managed folder whose registry lists
    Alice's skill `team-report`, which she created herself."""
    get_app_settings.cache_clear()
    SkillService(ALICE).create_skill(SHARED, SHARED_BODY)
    _write_registry(managed_dir, [_entry(sub=SUB_A, partition=PARTITION_A, skill=SHARED)])
    return managed_dir


@pytest.fixture
def all_shared(personal_deployment, stores, monkeypatch):
    """Skills left off COWORK_PERSONAL_RESOURCES: one base store."""
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(SHARED_RESOURCES)))
    get_app_settings.cache_clear()
    return stores


def _entry(*, sub: str, partition: str, skill: str) -> dict[str, str]:
    return {"user_id": sub, "partition_id": partition, "skill": skill}


def _write_registry(managed_dir, entries: list[dict[str, str]]) -> None:
    managed_dir.shared_dir.mkdir(exist_ok=True)
    managed_dir.shared_skills_file.write_text(json.dumps(entries))


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


def _links(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        return {}
    return {entry.name: entry.resolve() for entry in folder.iterdir() if entry.is_symlink()}


def _refused(call, detail: str = SHARED_SKILL_REFUSAL) -> None:
    with pytest.raises(HTTPException) as refused:
        call()
    assert (refused.value.status_code, refused.value.detail) == (403, detail)


def _every_change_is_refused(scoped, scope: TenantScope, slug: str) -> None:
    """Every mutating skills route for ``scope`` on ``slug``: edit, disable,
    rename it, rename one of ``scope``'s own into it, create it, upload it as
    a SKILL.md and as a zip, and delete it. The service refuses each too."""
    session = scoped(scope)
    service = SkillService(scope)
    service.create_skill("own-notes", "Mine.")
    _refused(lambda: skills_ep.update_skill(slug, SkillUpdateRequest(instructions="x"), session, None))
    _refused(lambda: skills_ep.update_skill(slug, SkillUpdateRequest(enabled=False), session, None))
    _refused(lambda: skills_ep.update_skill(slug, SkillUpdateRequest(label="taken-over"), session, None))
    _refused(lambda: skills_ep.update_skill("own-notes", SkillUpdateRequest(label=slug), session, None))
    _refused(lambda: skills_ep.create_skill(SkillCreateRequest(label=slug, instructions="x"), session, None))
    _refused(lambda: skills_ep._upload_skill_bytes(_skill_md(slug, "x").encode(), "SKILL.md", session, None))
    _refused(lambda: skills_ep._upload_skill_bytes(_zip(slug, "x"), "skill.zip", session, None))
    _refused(lambda: skills_ep.delete_skill(slug, session, None))
    with pytest.raises(SkillImmutableError, match="shared with you"):
        service.update_skill(slug, instructions="x")
    with pytest.raises(SkillImmutableError, match="shared with you"):
        service.delete_skill(slug)
    with pytest.raises(SkillImmutableError, match="shared with you"):
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


def test_every_change_bob_tries_is_refused(sharing, scoped):
    before = (SkillService(ALICE).root / SHARED / "SKILL.md").read_bytes()

    _every_change_is_refused(scoped, BOB, SHARED)

    assert (SkillService(ALICE).root / SHARED / "SKILL.md").read_bytes() == before
    assert not (SkillService(BOB).root / SHARED).exists()
    assert not (SkillService(BOB).root / "taken-over").exists()


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
    _write_registry(sharing, [])

    assert SHARED not in _listed(scoped, BOB)
    alices = _listed(scoped, ALICE)[SHARED]
    assert (alices["shared"], alices["capabilities"]) == (False, EVERY_CAPABILITY)


def test_alice_renames_her_shared_skill_only_after_unsharing_it(sharing, scoped):
    """The registry names a skill by slug, and the server never writes it, so
    a rename while shared would silently stop sharing it: it is refused."""
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

    _write_registry(sharing, [])
    renamed = skills_ep.update_skill(SHARED, SkillUpdateRequest(label="team-report-v2"), alice, None)
    assert renamed["id"] == "team-report-v2"


def test_a_name_still_shared_for_a_deleted_skill_is_never_taken_again(sharing, scoped):
    """Alice deletes her shared skill, and its registry entry stays until the
    operator unshares it. A create, upload or rename into that name would
    share the new skill with everyone without the operator sharing it, so
    each is refused until then."""
    alice = scoped(ALICE)
    skills_ep.delete_skill(SHARED, alice, None)
    SkillService(ALICE).create_skill("private-notes", "Alice's private notes, never shared.")

    for call in (
        lambda: skills_ep.update_skill("private-notes", SkillUpdateRequest(label=SHARED), alice, None),
        lambda: skills_ep.create_skill(SkillCreateRequest(label=SHARED, instructions="x"), alice, None),
        lambda: skills_ep._upload_skill_bytes(_skill_md(SHARED, "x").encode(), "SKILL.md", alice, None),
        lambda: skills_ep._upload_skill_bytes(_zip(SHARED, "x"), "skill.zip", alice, None),
    ):
        _refused(call, STILL_SHARED_SKILL_REFUSAL)
    assert not (SkillService(ALICE).root / SHARED).exists()
    assert SHARED not in _listed(scoped, BOB)

    _write_registry(sharing, [])
    renamed = skills_ep.update_skill("private-notes", SkillUpdateRequest(label=SHARED), alice, None)
    assert (renamed["id"], renamed["shared"]) == (SHARED, False)
    assert SHARED not in _listed(scoped, BOB)


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
        assert roots == [HOST_SKILLS_DIR, sharing.skills_dir, turn.scratch / "shared-skills"]
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
    _write_registry(
        sharing,
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

    slug = sorted(readonly_skill_dirs(SkillStore().builtin_root if root == "anton" else HOST_SKILLS_DIR))[0]
    assert slug in agent_skill_names()
    SkillService(ALICE).create_skill(slug, "Alice's version, for everyone.")
    _write_registry(sharing, [_entry(sub=SUB_A, partition=PARTITION_A, skill=slug)])
    shared_dir = stores / "bob-turn" / "shared-skills"

    assert slug not in SkillService(BOB).shared_with_owner()
    assert slug not in _listed(scoped, BOB)
    link_turn_skills(
        scope=BOB, project_name="general", skills_dir=stores / "bob-turn" / "skills", shared_skills_dir=shared_dir
    )
    assert slug not in _links(shared_dir)
    created = skills_ep.create_skill(SkillCreateRequest(label=slug, instructions="Bob's."), scoped(BOB), None)
    assert (created["id"], created["shared"]) == (slug, False)


def test_only_registry_entries_of_existing_skills_are_read(sharing, scoped):
    SkillService(ALICE).create_skill("alices-private", "Alice's alone.")
    _write_registry(
        sharing,
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


def test_a_malformed_registry_shares_nothing_and_says_so(sharing, scoped, caplog, owned_logger):
    sharing.shared_skills_file.write_text('{"not": "a list"}')
    owned_logger("cowork.services.skills", level=logging.ERROR)

    assert SHARED not in _listed(scoped, BOB)
    assert _listed(scoped, ALICE)[SHARED]["shared"] is False

    errors = [
        record.getMessage()
        for record in caplog.records
        if record.name == "cowork.services.skills" and record.levelno == logging.ERROR
    ]
    assert errors and str(sharing.shared_skills_file) in errors[0]


def test_only_a_scope_with_personal_skills_reads_the_registry(sharing):
    """Where skills are shared the base store already holds everyone's: a
    person's scope that keeps skills shared, and the operator view, read no
    registry."""
    bob = _person(partition=PARTITION_B, sub=SUB_B, resources=SHARED_RESOURCES)
    assert SkillService(bob).shared_with_owner() == {}
    assert SkillService(LOCAL_SCOPE).shared_with_owner() == {}


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

    # Everyone lists it, and the capabilities say who can change it.
    for scope, capabilities in (
        (BOB_SHARED, NO_CAPABILITY),
        (LOCAL_SCOPE, EVERY_CAPABILITY),
        (ALICE_SHARED, EVERY_CAPABILITY),
    ):
        listed = _listed(scoped, scope)["team-notes"]
        assert (listed["capabilities"], listed["shared"]) == (capabilities, False)

    alice = scoped(ALICE_SHARED)
    skills_ep.update_skill("team-notes", SkillUpdateRequest(instructions="Changed."), alice, None)
    renamed = skills_ep.update_skill("team-notes", SkillUpdateRequest(label="team-notes-v2"), alice, None)
    assert renamed["capabilities"] == EVERY_CAPABILITY
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
    assert edited["capabilities"] == EVERY_CAPABILITY
    assert META_CREATED_BY not in _stored(LOCAL_SCOPE, "legacy")
    skills_ep.delete_skill("legacy", operator, None)
    assert "legacy" not in _listed(scoped, LOCAL_SCOPE)


# ── the desktop and hosted org tenancy ─────────────────────────────────────


def _every_route_works(session, principal, *, upload_creator: str | None) -> None:
    """Create, upload, list, read, edit, disable, rename and delete one skill
    each, as the caller who made them. None is ``shared``, and nothing stamps
    a creator: an upload keeps what its file says (``upload_creator``)."""
    created = skills_ep.create_skill(SkillCreateRequest(label="my-notes", instructions="Mine."), session, principal)
    assert (created["shared"], created["capabilities"]) == (False, EVERY_CAPABILITY)
    uploaded = skills_ep._upload_skill_bytes(
        _skill_md("uploaded", "Mine.", created_by=SUB_B).encode(), "SKILL.md", session, principal
    )
    assert uploaded["shared"] is False
    service = SkillService(session.scope)
    assert service.get_skill("my-notes").created_by is None
    assert service.get_skill("uploaded").created_by == upload_creator

    listed = {s["id"]: s for s in skills_ep.list_skills(session, principal)["skills"]}
    assert {"my-notes", "uploaded"} <= set(listed)
    assert not any(skill["shared"] for skill in listed.values())
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
