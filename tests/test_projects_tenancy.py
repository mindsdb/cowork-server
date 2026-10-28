"""Cross-tenant behaviour of the swept ProjectService.

Two orgs against one database: everything org A creates must be invisible
and untouchable for org B — and indistinguishable from nonexistent (404-shaped
None/ValueError, never a "forbidden"). Also covers the default-project
contract: one per org, no duplicates, no created_by attribution.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from sqlmodel import Session, select

from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.models.project import Project
from cowork.services.projects import GENERAL_PROJECT, GENERAL_PROJECT_ID, ProjectService

ORG_A = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
ORG_B = "0f7f0b6a-3f0f-4c58-9e0c-6dbb3ac0f0a1"


def _scope(org: str, user: str = "user-1") -> TenantScope:
    return TenantScope(org_mode=True, org_id=org, user_id=user)


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Isolated engine + projects root, seeded with the GENERAL row (NULL org)."""
    # Org mode roots at cowork_home() regardless of COWORK_PROJECTS_DIR (see
    # scoped_storage_root), so it must be pinned here too or org-scoped tests
    # fall through to the real ~/.cowork.
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("COWORK_SHARED_DIR", str(tmp_path))
    from cowork.common.settings.app_settings import get_app_settings
    get_app_settings.cache_clear()

    from sqlalchemy.pool import StaticPool
    from sqlmodel import SQLModel, create_engine

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as seed:
        seed.add(Project(id=GENERAL_PROJECT_ID, name=GENERAL_PROJECT, path=str(tmp_path / "projects" / "general"), is_active=True))
        seed.commit()
    yield engine
    get_app_settings.cache_clear()


def _svc(engine, scope: TenantScope) -> ProjectService:
    # One session per scope, like production (one request = one session).
    return ProjectService(ScopedSession(Session(engine), scope))


def _raw(engine) -> Session:
    return Session(engine)


def test_creation_stamps_org_and_creator(db):
    svc = _svc(db, _scope(ORG_A, "alice"))
    project = svc.create_project("reports")
    assert project.org_id == ORG_A
    assert project.created_by == "alice"


def test_other_org_cannot_see_list_get_rename_or_delete(db):
    a = _svc(db, _scope(ORG_A))
    b = _svc(db, _scope(ORG_B))
    project = a.create_project("secret-plans")

    assert "secret-plans" not in {p.name for p in b.list_projects()}
    with pytest.raises(ValueError, match="not found"):
        b.get_project(project.id)
    with pytest.raises(ValueError, match="not found"):
        b.get_project_by_name("secret-plans")
    with pytest.raises(ValueError, match="not found"):
        b.update_project(project.id, name="stolen")
    assert b.delete_project(project.id) is False  # same answer as nonexistent

    # and org A still has it, untouched
    assert a.get_project(project.id).name == "secret-plans"


def test_project_names_are_per_org(db):
    a = _svc(db, _scope(ORG_A))
    b = _svc(db, _scope(ORG_B))
    assert a.create_project("reports").name == "reports"
    # Name uniqueness is org-scoped: B is blind to A's "reports", so no -2
    # suffix. Directories are separated by the org-keyed root now, not by
    # "one org per deployment" (that died with the shared-instance decision).
    assert b._unique_name("reports") == "reports"


@pytest.mark.parametrize("evil", ["../evil", "a/../../b", "/etc/passwd", "..", "a/b"])
def test_project_paths_cannot_escape_the_root(db, tmp_path, evil):
    svc = _svc(db, _scope(ORG_A))
    project = svc.create_project(evil)
    # The created dir must sit directly under this ORG's projects root (the root
    # is org-keyed, like the skills/memory stores), not wherever the (sanitized)
    # name happened to point.
    projects_root = (tmp_path / str(ORG_A) / "projects").resolve()
    assert Path(project.path).resolve().parent == projects_root
    assert "/" not in project.name and ".." != project.name


def test_project_path_guard_rejects_escapes_directly(db):
    svc = _svc(db, _scope(ORG_A))
    with pytest.raises(ValueError):
        svc._project_path("../evil")
    with pytest.raises(ValueError):
        svc._project_path(".")


def test_local_mode_sees_everything(db):
    a = _svc(db, _scope(ORG_A))
    a.create_project("cloud-thing")
    local = _svc(db, LOCAL_SCOPE)
    assert "cloud-thing" in {p.name for p in local.list_projects()}


# ── GENERAL project bootstrap ───────────────────────────────────────────────

def test_general_is_idempotent_within_an_org(db):
    a = _svc(db, _scope(ORG_A))
    first = a.ensure_general_for_scope()
    second = a.ensure_general_for_scope()
    assert first is not None and second is not None
    assert first.id == second.id
    assert first.org_id == ORG_A
    # The member who first reaches the org provisions General and is recorded as
    # its creator, so its instructions are editable without an admin.
    assert first.created_by == "user-1"


def test_general_keeps_its_first_provisioner_as_creator(db):
    """A later member never takes over General's recorded creator."""
    first = _svc(db, _scope(ORG_A, "user-1")).ensure_general_for_scope()
    later = _svc(db, _scope(ORG_A, "user-2")).ensure_general_for_scope()
    assert first is not None and later is not None
    assert later.id == first.id
    assert later.created_by == "user-1"


def test_every_org_gets_its_own_general(db):
    """One seeded id can't serve N tenants: the first org claimed it and every
    later org got None → 404."""
    a = _svc(db, _scope(ORG_A))
    b = _svc(db, _scope(ORG_B))

    ga = a.ensure_general_for_scope()
    gb = b.ensure_general_for_scope()

    assert ga is not None and gb is not None
    assert ga.id != gb.id  # separate rows, not a contested one
    assert (ga.org_id, gb.org_id) == (ORG_A, ORG_B)
    assert Path(ga.path) != Path(gb.path)  # and separate directories
    # Each org sees exactly its own.
    assert GENERAL_PROJECT in {p.name for p in a.list_projects()}
    assert GENERAL_PROJECT in {p.name for p in b.list_projects()}


def test_general_does_not_adopt_a_legacy_null_org_row(db):
    # A legacy NULL-org project must NOT be swept into an org.
    raw = _raw(db)
    raw.add(Project(name="legacy", path="/tmp/legacy", is_active=False))
    raw.commit()
    a = _svc(db, _scope(ORG_A))
    a.ensure_general_for_scope()

    legacy = _raw(db).exec(select(Project).where(Project.name == "legacy")).one()
    assert legacy.org_id is None


def test_general_never_duplicates_across_concurrent_sessions(db):
    """No unique constraint + 2 replicas + called on every request: two sessions
    racing first load must converge on ONE row."""
    a1 = _svc(db, _scope(ORG_A, "alice"))
    a2 = _svc(db, _scope(ORG_A, "bob"))  # a second replica, same org

    # Both past the pre-check before either writes. Calling
    # ensure_general_for_scope twice would short-circuit and miss the race.
    path = a1._project_path(GENERAL_PROJECT)
    a1._insert_general_if_absent(path)
    a2._insert_general_if_absent(path)

    rows = _raw(db).exec(
        select(Project).where(Project.org_id == ORG_A, Project.name == GENERAL_PROJECT)
    ).all()
    assert len(rows) == 1
    assert a1.ensure_general_for_scope().id == rows[0].id


def test_general_recreates_a_missing_directory(db):
    """The live 404: row resolved, directory did not exist."""
    a = _svc(db, _scope(ORG_A))
    general = a.ensure_general_for_scope()
    assert general is not None
    shutil.rmtree(general.path)
    assert not Path(general.path).is_dir()

    again = a.ensure_general_for_scope()

    assert again is not None and Path(again.path).is_dir()


def test_general_in_local_mode_needs_no_claim(db):
    local = _svc(db, LOCAL_SCOPE)
    general = local.ensure_general_for_scope()
    assert general is not None
    row = _raw(db).exec(select(Project).where(Project.id == GENERAL_PROJECT_ID)).one()
    assert row.org_id is None  # local mode never stamps


def test_general_repoints_a_legacy_unkeyed_path_with_no_content(db, tmp_path):
    """Rows predating the org-keyed root point outside it. With no content there,
    re-point them — otherwise the org stays 404'd."""
    a = _svc(db, _scope(ORG_A))
    legacy = tmp_path / "projects" / GENERAL_PROJECT  # un-keyed, never created
    raw = _raw(db)
    raw.add(Project(name=GENERAL_PROJECT, path=str(legacy), is_active=False, org_id=ORG_A))
    raw.commit()

    general = a.ensure_general_for_scope()

    assert general is not None
    assert Path(general.path).parent == (tmp_path / str(ORG_A) / "projects")
    assert Path(general.path).is_dir()


def test_general_keeps_a_legacy_path_it_cannot_read(db, tmp_path, monkeypatch):
    """An unreadable directory is not an empty one. A disconnected mount or a
    folder the process cannot list must not read as "no content" and cost the
    org the path its work is on."""
    a = _svc(db, _scope(ORG_A))
    legacy = tmp_path / "projects" / GENERAL_PROJECT
    legacy.mkdir(parents=True)
    raw = _raw(db)
    raw.add(Project(name=GENERAL_PROJECT, path=str(legacy), is_active=False, org_id=ORG_A))
    raw.commit()

    readable = Path.iterdir

    def refuse(self):
        if self == legacy:
            raise OSError(116, "Stale file handle")
        return readable(self)

    monkeypatch.setattr(Path, "iterdir", refuse)

    general = a.ensure_general_for_scope()

    assert general is not None
    assert Path(general.path) == legacy


def test_general_keeps_a_legacy_path_that_still_has_content(db, tmp_path):
    """The mirror case: a dir with real content must not be abandoned."""
    a = _svc(db, _scope(ORG_A))
    legacy = tmp_path / "projects" / GENERAL_PROJECT
    legacy.mkdir(parents=True)
    (legacy / "notes.md").write_text("real work")
    raw = _raw(db)
    raw.add(Project(name=GENERAL_PROJECT, path=str(legacy), is_active=False, org_id=ORG_A))
    raw.commit()

    general = a.ensure_general_for_scope()

    assert general is not None
    assert Path(general.path) == legacy
    assert (Path(general.path) / "notes.md").read_text() == "real work"


def test_same_project_name_in_two_orgs_gets_separate_directories(db):
    """Rows were always keyed by org_id; the filesystem was not — two orgs using
    `reports` shared one directory."""
    a = _svc(db, _scope(ORG_A))
    b = _svc(db, _scope(ORG_B))

    pa = a.create_project("reports")
    pb = b.create_project("reports")

    assert pa.name == pb.name == "reports"  # no -2 suffix: names are per org
    assert Path(pa.path) != Path(pb.path)
    (Path(pa.path) / "a.md").write_text("org A")
    (Path(pb.path) / "b.md").write_text("org B")
    assert not (Path(pa.path) / "b.md").exists()  # no shared directory


def test_general_id_round_trips_through_get_project(db):
    """Review: sa.literal(str(uuid4())) binds as String and skips the Uuid bind
    processor, so the stored id doesn't match how every other lookup binds.

    Must use a FRESH session: the writing session's identity map would serve the
    row from memory and never round-trip through the column type.
    """
    general = _svc(db, _scope(ORG_A)).ensure_general_for_scope()
    assert general is not None

    fresh = _svc(db, _scope(ORG_A))
    assert fresh.get_project(general.id).id == general.id


def test_unique_index_rejects_a_second_default_for_one_org(db):
    """What arbitrates on Postgres, where two replicas can both pass NOT EXISTS."""
    import sqlalchemy as sa
    raw = _raw(db)
    raw.exec(sa.text(
        "CREATE UNIQUE INDEX uq_projects_default_per_org ON projects "
        "(coalesce(org_id, '')) WHERE name = 'general'"
    ))
    _svc(db, _scope(ORG_A)).ensure_general_for_scope()

    with pytest.raises(sa.exc.IntegrityError):
        dup = _raw(db)
        dup.add(Project(name=GENERAL_PROJECT, path="/tmp/dup", is_active=False, org_id=ORG_A))
        dup.commit()


def test_a_losing_insert_adopts_the_winners_row(db, monkeypatch):
    """The loser must return the winner's row, not assume its own insert landed."""
    import sqlalchemy as sa
    winner = _svc(db, _scope(ORG_A, "alice")).ensure_general_for_scope()
    assert winner is not None

    loser = _svc(db, _scope(ORG_A, "bob"))
    monkeypatch.setattr(
        type(loser),
        "get_project_by_name_or_none",
        lambda self, name: None,  # pretend the pre-check saw nothing
        raising=True,
    )
    monkeypatch.setattr(
        type(loser),
        "_execute_general_insert",
        lambda self, raw, path, org_id, created_by: (_ for _ in ()).throw(
            sa.exc.IntegrityError("insert", {}, Exception("duplicate"))
        ),
        raising=True,
    )

    loser._insert_general_if_absent(loser._project_path(GENERAL_PROJECT))  # must not raise

    rows = _raw(db).exec(
        select(Project).where(Project.org_id == ORG_A, Project.name == GENERAL_PROJECT)
    ).all()
    assert len(rows) == 1 and rows[0].id == winner.id


def test_fresh_org_has_an_active_project(db):
    """Review: provisioned with is_active=False, so a fresh org had zero active
    projects and get_active_project raised on the first request."""
    a = _svc(db, _scope(ORG_A))
    a.ensure_general_for_scope()
    assert a.get_active_project().name == GENERAL_PROJECT


def test_a_member_cannot_take_the_default_projects_name(db):
    """Review: a member could POST /projects named `general` before the system row
    exists — it became user-attributed and undeletable, and the name-based lookup
    then blocked the real default forever."""
    a = _svc(db, _scope(ORG_A, "alice"))
    hijack = a.create_project(GENERAL_PROJECT)

    assert hijack.name != GENERAL_PROJECT  # the name is reserved
    general = a.ensure_general_for_scope()
    assert general is not None
    assert general.id != hijack.id  # the real system row, not the member's
    assert general.created_by == "alice"  # provisioned by, not hijacked by


def test_repointing_does_not_attribute_the_system_project(db, tmp_path):
    """Review: ScopedSession.add stamps created_by when it's None, so healing a
    stale path attributed the org's default to whoever triggered the heal."""
    legacy = tmp_path / "projects" / GENERAL_PROJECT
    raw = _raw(db)
    raw.add(Project(name=GENERAL_PROJECT, path=str(legacy), is_active=False, org_id=ORG_A))
    raw.commit()

    general = _svc(db, _scope(ORG_A, "alice")).ensure_general_for_scope()

    assert general is not None
    assert general.created_by is None  # still system-created


def test_repoints_off_an_empty_legacy_directory(db, tmp_path):
    """An empty dir is not content: it must move to the org-keyed path, not pin
    the row to a pre-org-keyed location forever."""
    legacy = tmp_path / "projects" / GENERAL_PROJECT
    legacy.mkdir(parents=True)  # exists, but empty
    raw = _raw(db)
    raw.add(Project(name=GENERAL_PROJECT, path=str(legacy), is_active=False, org_id=ORG_A))
    raw.commit()

    general = _svc(db, _scope(ORG_A)).ensure_general_for_scope()

    assert general is not None
    assert Path(general.path).parent == (tmp_path / str(ORG_A) / "projects")


def test_any_project_recovers_a_missing_directory(db):
    """Review: the heal was wired only into the default project, so any other
    project whose directory disappeared with the pod stayed broken."""
    a = _svc(db, _scope(ORG_A))
    reports = a.create_project("reports")
    shutil.rmtree(reports.path)

    a.ensure_dir_exists(reports)

    assert Path(reports.path).is_dir()


def test_rename_works_when_the_org_root_does_not_exist_yet(db, tmp_path):
    """Review: rename into a not-yet-created <root>/<org_id> raised
    FileNotFoundError, which the endpoint's `except ValueError` turns into a 500."""
    legacy = tmp_path / "projects" / "legacy-proj"
    legacy.mkdir(parents=True)
    (legacy / "keep.md").write_text("content")
    raw = _raw(db)
    raw.add(Project(name="legacy-proj", path=str(legacy), is_active=False, org_id=ORG_A))
    raw.commit()
    a = _svc(db, _scope(ORG_A))
    assert not (tmp_path / str(ORG_A) / "projects").exists()  # org root absent

    renamed = a.update_project(a.get_project_by_name("legacy-proj").id, name="renamed")

    assert renamed.name == "renamed"
    assert (Path(renamed.path) / "keep.md").read_text() == "content"


def test_a_leftover_directory_is_not_adopted(db):
    """Review: exist_ok=True dropped the backstop, so a create silently adopted a
    leftover directory (stale contents included) or shared one with a racing
    create — where deleting either would rmtree the other's files."""
    a = _svc(db, _scope(ORG_A))
    first = a.create_project("reports")
    first_path = Path(first.path)

    # Row gone, directory left behind (a crashed delete, a restored volume).
    raw = _raw(db)
    raw.delete(raw.exec(select(Project).where(Project.name == "reports")).one())
    raw.commit()
    first_path.mkdir(parents=True, exist_ok=True)
    (first_path / "stale.md").write_text("previous tenant of this directory")

    second = _svc(db, _scope(ORG_A)).create_project("reports")

    assert Path(second.path) != first_path  # a fresh directory
    assert not (Path(second.path) / "stale.md").exists()


def test_delete_removes_the_projects_own_directory(db):
    a = _svc(db, _scope(ORG_A))
    p = a.create_project("reports")
    path = Path(p.path)
    assert path.is_dir()

    assert a.delete_project(p.id) is True

    assert not path.exists()


def test_delete_never_rmtrees_a_path_outside_the_sanitizer(db, tmp_path):
    """A tampered stored path must not reach rmtree: delete_project rebuilds the
    target from the sanitized name and only deletes it when it matches the row.
    The mismatched directory is left in place; the row still goes away."""
    a = _svc(db, _scope(ORG_A))
    p = a.create_project("reports")
    # Simulate a poisoned row pointing outside the org root.
    victim = tmp_path / "victim-dir"
    victim.mkdir()
    (victim / "precious.md").write_text("someone else's data")
    raw = _raw(db)
    row = raw.exec(select(Project).where(Project.name == "reports")).one()
    row.path = str(victim)
    raw.add(row); raw.commit()

    assert _svc(db, _scope(ORG_A)).delete_project(p.id) is True

    assert (victim / "precious.md").exists()  # the foreign directory survived
    assert a.get_project_by_name_or_none("reports") is None  # the row is gone


def test_ensure_dir_never_creates_a_path_outside_the_sanitizer(db, tmp_path):
    """A tampered stored path must not reach mkdir: ensure_dir_exists rebuilds the
    target from the sanitized name and only creates it when it matches the row."""
    a = _svc(db, _scope(ORG_A))
    p = a.create_project("reports")
    shutil.rmtree(p.path)
    # Simulate a poisoned row pointing outside the org root.
    raw = _raw(db)
    row = raw.exec(select(Project).where(Project.name == "reports")).one()
    escape = tmp_path / "escape-target"
    row.path = str(escape)
    raw.add(row); raw.commit()

    a.ensure_dir_exists(a.get_project_by_name("reports"))

    assert not escape.exists()  # the tampered path was never created


# ── Resolving `general` by name before it is provisioned ────────────────────

def test_get_or_provision_by_name_provisions_general_before_first_list(db):
    """The bug: a send/task-create names `general` on an org whose default row has
    not been provisioned yet (no prior GET /projects/). A plain by-name lookup 404s;
    get_or_provision_by_name must create it instead."""
    a = _svc(db, _scope(ORG_A))
    # No GET /projects/ ran for this org, so the exact-match lookup still misses.
    with pytest.raises(ValueError, match="not found"):
        a.get_project_by_name(GENERAL_PROJECT)

    general = a.get_or_provision_by_name(GENERAL_PROJECT)

    assert general.name == GENERAL_PROJECT
    assert general.org_id == ORG_A
    assert Path(general.path).is_dir()


def test_get_or_provision_by_name_still_404s_for_a_real_missing_project(db):
    """Only the reserved default self-heals; every other name stays an exact match."""
    a = _svc(db, _scope(ORG_A))
    with pytest.raises(ValueError, match="not found"):
        a.get_or_provision_by_name("does-not-exist")


def test_conversation_project_by_name_provisions_general(db):
    """The create/update/move task endpoints resolve the project name through
    ConversationService.project_by_name — it must provision `general` too, so a task
    created in the default project on a fresh org doesn't 404."""
    from cowork.services.conversations import ConversationService

    svc = ConversationService(ScopedSession(Session(db), _scope(ORG_A)))
    project = svc.project_by_name(GENERAL_PROJECT)

    assert project is not None
    assert project.name == GENERAL_PROJECT and project.org_id == ORG_A


def test_delete_project_cascades_all_members_conversations(db):
    """Regression (owner-scoping PR): delete_project is org-wide cleanup, so it
    must delete EVERY member's conversation in the project — owner-scoping the
    cascade would orphan foreign members' rows/bytes (ENG-701)."""
    from cowork.services.conversations import ConversationService
    from cowork.models.conversation import Conversation
    from sqlmodel import select as _select

    alice = _svc(db, _scope(ORG_A, "alice"))
    proj = alice.create_project("shared-reports")

    a_conv = ConversationService(ScopedSession(Session(db), _scope(ORG_A, "alice"))).create_conversation(
        topic="alice", project_id=proj.id)
    b_conv = ConversationService(ScopedSession(Session(db), _scope(ORG_A, "bob"))).create_conversation(
        topic="bob", project_id=proj.id)

    # Bob's conversation lives in Alice's project; deleting the project (as any
    # org member) must remove BOTH, leaving no orphaned conversation rows.
    assert alice.delete_project(proj.id) is True
    with Session(db) as s:
        remaining = s.exec(_select(Conversation).where(Conversation.project_id == proj.id)).all()
    assert remaining == []


# ── personal partitions (local tenancy with trusted identity headers) ──────

PARTITION_A = "44e92829-e701-599b-88f9-69f5bf1165eb"
PARTITION_B = "55e92829-e701-599b-88f9-69f5bf1165eb"
SUB_A = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
SUB_B = "8a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


def _person(*, partition: str, sub: str, resources=None) -> TenantScope:
    from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES

    return TenantScope(
        personal=True,
        org_id=partition,
        user_id=sub,
        personal_resources=ALL_PERSONAL_RESOURCES if resources is None else resources,
    )


def test_every_person_gets_their_own_general(db, personal_deployment, tmp_path):
    alice = _svc(db, _person(partition=PARTITION_A, sub=SUB_A))
    bob = _svc(db, _person(partition=PARTITION_B, sub=SUB_B))

    ga = alice.ensure_general_for_scope()
    gb = bob.ensure_general_for_scope()

    assert ga is not None and gb is not None
    assert ga.id != gb.id and GENERAL_PROJECT_ID not in {ga.id, gb.id}
    assert (ga.org_id, ga.created_by) == (PARTITION_A, SUB_A)
    assert (gb.org_id, gb.created_by) == (PARTITION_B, SUB_B)
    assert Path(ga.path) == tmp_path / PARTITION_A / "projects" / GENERAL_PROJECT
    assert alice.ensure_general_for_scope().id == ga.id
    assert [p.id for p in alice.list_projects()] == [ga.id]
    # The operator view still has the seeded deployment row.
    assert _svc(db, LOCAL_SCOPE).ensure_general_for_scope().id == GENERAL_PROJECT_ID


def test_a_person_with_shared_projects_uses_the_seeded_general(db, personal_deployment):
    shared = _person(partition=PARTITION_A, sub=SUB_A, resources=frozenset())
    general = _svc(db, shared).ensure_general_for_scope()
    assert general is not None and general.id == GENERAL_PROJECT_ID
    assert general.org_id is None


def test_a_partition_refuses_to_adopt_a_server_folder(db, personal_deployment, tmp_path):
    from cowork.services.projects import ProjectPathNotAllowedError

    chosen = tmp_path / "outside" / "notes"
    chosen.mkdir(parents=True)
    with pytest.raises(ProjectPathNotAllowedError):
        _svc(db, _person(partition=PARTITION_A, sub=SUB_A)).create_project("notes", path=chosen)
    assert _raw(db).exec(select(Project).where(Project.name == "notes")).first() is None


@pytest.mark.parametrize("who", ["person", "operator"])
def test_a_trusted_deployment_refuses_to_adopt_a_folder_with_projects_shared(
    db, personal_deployment, tmp_path, who
):
    """Projects shared change nothing: skill links reach no adopted folder on
    a deployment that names people, so its turns would see no saved skills."""
    from cowork.services.projects import ProjectPathNotAllowedError

    chosen = tmp_path / "outside" / "notes"
    chosen.mkdir(parents=True)
    scope = (
        _person(partition=PARTITION_A, sub=SUB_A, resources=frozenset({PersonalResource.MEMORY}))
        if who == "person"
        else LOCAL_SCOPE
    )
    with pytest.raises(ProjectPathNotAllowedError):
        _svc(db, scope).create_project("notes", path=chosen)
    assert _raw(db).exec(select(Project).where(Project.name == "notes")).first() is None


def test_the_desktop_still_adopts_a_folder(db, tmp_path):
    chosen = tmp_path / "outside" / "notes"
    chosen.mkdir(parents=True)
    desktop = _svc(db, LOCAL_SCOPE)
    project = desktop.create_project("notes", path=chosen)
    assert Path(project.path) == chosen.resolve()
    assert desktop.directory_is_external(project) is True


def test_no_partition_project_reads_as_an_adopted_folder(db, personal_deployment):
    alice = _svc(db, _person(partition=PARTITION_A, sub=SUB_A))
    project = alice.create_project("reports")
    # Even a row whose path sits outside the partition root is a stale row here,
    # not a chosen folder.
    assert alice.directory_is_external(project) is False
    raw = _raw(db)
    row = raw.get(Project, project.id)
    row.path = "/somewhere/else/reports"
    raw.add(row)
    raw.commit()
    assert alice.directory_is_external(alice.get_project(project.id)) is False


def test_partition_creates_and_renames_link_no_skills(db, personal_deployment, monkeypatch):
    """Skill links point from the unkeyed store into the unkeyed projects root,
    found by name. Two people who both own `notes` must not receive each
    other's links, so a partition distributes none."""
    calls = []
    monkeypatch.setattr(
        "cowork.services.skill_links.reconcile_project",
        lambda *args, **kwargs: calls.append(kwargs.get("project_name")),
    )
    alice = _svc(db, _person(partition=PARTITION_A, sub=SUB_A))
    bob = _svc(db, _person(partition=PARTITION_B, sub=SUB_B))
    bob.create_project("notes")
    mine = alice.create_project("notes")
    alice.update_project(mine.id, name="notes-renamed")

    assert calls == []


def test_shared_projects_and_skills_keep_todays_links(db, personal_deployment, monkeypatch):
    from cowork.common.settings.app_settings import get_app_settings

    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", "")
    get_app_settings.cache_clear()
    calls = []
    monkeypatch.setattr(
        "cowork.services.skill_links.reconcile_project",
        lambda *args, **kwargs: calls.append(kwargs.get("project_name")),
    )
    shared = _svc(db, _person(partition=PARTITION_A, sub=SUB_A, resources=frozenset()))
    project = shared.create_project("notes")
    shared.update_project(project.id, name="notes-renamed")

    assert calls == ["notes", "notes-renamed"]


def test_a_partition_still_takes_the_process_name_lock(db, personal_deployment, monkeypatch):
    """The distributed lock that lets org mode skip this one does nothing
    outside org mode, so a partition's creates and renames must still queue on
    the process-local lock."""
    from cowork.services import projects as projects_module

    monkeypatch.setattr(projects_module, "_NAME_LOCK_TIMEOUT_SECONDS", 0.01)
    assert projects_module._LOCAL_NAME_LOCK.acquire(timeout=1)
    try:
        with pytest.raises(projects_module.ProjectNameLockBusyError):
            _svc(db, _person(partition=PARTITION_A, sub=SUB_A)).create_project("busy")
    finally:
        projects_module._LOCAL_NAME_LOCK.release()


def test_a_partition_refuses_adoption_before_touching_the_filesystem(db, personal_deployment, tmp_path):
    """Refused before any stat: a path that does not exist gets the same
    refusal as one that does, never "Choose an existing local folder", so the
    answer says nothing about the server's filesystem."""
    from cowork.services.projects import ProjectPathNotAllowedError

    missing = tmp_path / "no" / "such" / "folder"
    with pytest.raises(ProjectPathNotAllowedError):
        _svc(db, _person(partition=PARTITION_A, sub=SUB_A)).create_project("notes", path=missing)
    assert not missing.exists()


def test_a_partition_row_is_no_adopted_folder_to_the_operator_view(db, personal_deployment, tmp_path):
    """The operator view reads every row unfiltered. A person's project lives
    under their partition, outside the unkeyed root, which would read as a
    folder somebody chose; a row with an org_id never is one."""
    alice = _svc(db, _person(partition=PARTITION_A, sub=SUB_A))
    theirs = alice.create_project("reports")
    operator = _svc(db, LOCAL_SCOPE)

    assert operator.directory_is_external(operator.get_project(theirs.id)) is False

    # A NULL-org row outside the root is still an adopted folder there.
    chosen = tmp_path / "outside" / "chosen"
    chosen.mkdir(parents=True)
    raw = _raw(db)
    adopted = Project(name="chosen", path=str(chosen), is_active=True)
    raw.add(adopted)
    raw.commit()
    assert operator.directory_is_external(operator.get_project(adopted.id)) is True


# ── a deleted project's memory, and a rename's skill store ───────────────────


def _lesson_in_tier(scope: TenantScope, project: Project, text: str) -> Path:
    from cowork.harnesses.memory.registry import MemorySlot
    from cowork.harnesses.memory.store import project_memory_store

    store = project_memory_store(scope=scope, project_path=Path(project.path), project_id=project.id)
    store.write(MemorySlot.LESSONS, text)
    return store.root


def test_deleting_a_project_removes_the_persons_memory_tier(db, personal_deployment, tmp_path):
    """A person's project tier lives under their global tier, outside the
    project's folder, so it goes with the project rather than staying unlisted."""
    alice_scope = _person(partition=PARTITION_A, sub=SUB_A)
    alice = _svc(db, alice_scope)
    doomed = alice.create_project("doomed")
    kept = alice.create_project("kept")
    doomed_tier = _lesson_in_tier(alice_scope, doomed, "- Pallets of 40")
    kept_tier = _lesson_in_tier(alice_scope, kept, "- Ships on Fridays")
    assert doomed_tier == tmp_path / PARTITION_A / "memory" / "users" / SUB_A / "projects" / str(doomed.id)

    assert alice.delete_project(doomed.id) is True

    assert not doomed_tier.exists()
    assert (kept_tier / "lessons.md").read_text().strip() == "- Ships on Fridays"


def test_deleting_a_shared_project_removes_every_persons_tier(db, personal_deployment, tmp_path):
    shared = frozenset(ALL_PERSONAL_RESOURCES - {PersonalResource.PROJECTS})
    alice_scope = _person(partition=PARTITION_A, sub=SUB_A, resources=shared)
    bob_scope = _person(partition=PARTITION_B, sub=SUB_B, resources=shared)
    alice = _svc(db, alice_scope)
    team = alice.create_project("team")
    other = alice.create_project("other")
    tiers = [_lesson_in_tier(scope, team, "- Mine") for scope in (alice_scope, bob_scope)]
    other_tier = _lesson_in_tier(bob_scope, other, "- Elsewhere")
    # A link planted where a person's folder belongs is never followed.
    outside = tmp_path / "outside-memory" / "projects" / str(team.id)
    outside.mkdir(parents=True)
    (tmp_path / PARTITION_B / "memory" / "users" / "planted").symlink_to(tmp_path / "outside-memory")

    assert _svc(db, bob_scope).delete_project(team.id) is True

    assert not any(tier.exists() for tier in tiers)
    assert other_tier.is_dir() and outside.is_dir()


def test_deleting_a_project_with_memory_shared_removes_its_shared_tier_and_kept_personal_ones(
    db, personal_deployment, monkeypatch, tmp_path
):
    """Memory shared: the deleted project's tier in the deployment store goes,
    and so does a person's tier for it kept from before the switch. Another
    project's tiers stay, and a link planted in the tier's place is unlinked
    without being followed."""
    from cowork.common.settings.app_settings import get_app_settings

    shares_memory = frozenset(ALL_PERSONAL_RESOURCES - {PersonalResource.MEMORY})
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(shares_memory)))
    get_app_settings.cache_clear()
    shared_alice = _person(partition=PARTITION_A, sub=SUB_A, resources=shares_memory)
    alice = _svc(db, shared_alice)
    doomed = alice.create_project("doomed")
    kept = alice.create_project("kept")
    linked = alice.create_project("linked")
    kept_personal = _lesson_in_tier(_person(partition=PARTITION_A, sub=SUB_A), doomed, "- From before")
    shared_doomed = _lesson_in_tier(shared_alice, doomed, "- Shared pallets")
    shared_kept = _lesson_in_tier(shared_alice, kept, "- Shared Fridays")
    assert shared_doomed == tmp_path / "memory" / "projects" / str(doomed.id)
    outside = tmp_path / "outside-memory"
    outside.mkdir()
    (outside / "lessons.md").write_text("- Not a tier\n")
    (tmp_path / "memory" / "projects" / str(linked.id)).symlink_to(outside, target_is_directory=True)

    assert alice.delete_project(doomed.id) is True
    assert alice.delete_project(linked.id) is True

    assert not shared_doomed.exists() and not kept_personal.exists()
    assert (shared_kept / "lessons.md").read_text().strip() == "- Shared Fridays"
    assert not (tmp_path / "memory" / "projects" / str(linked.id)).is_symlink()
    assert (outside / "lessons.md").read_text() == "- Not a tier\n"


def test_deleting_a_project_with_memory_per_person_removes_a_kept_shared_tier(
    db, personal_deployment, monkeypatch, tmp_path
):
    """The tier written into the deployment store while memory was shared stays
    there after the switch back, and goes when its project is deleted."""
    from cowork.common.settings.app_settings import get_app_settings

    shares_memory = frozenset(ALL_PERSONAL_RESOURCES - {PersonalResource.MEMORY})
    alice_scope = _person(partition=PARTITION_A, sub=SUB_A)
    alice = _svc(db, alice_scope)
    doomed = alice.create_project("doomed")
    kept = alice.create_project("kept")
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(shares_memory)))
    get_app_settings.cache_clear()
    shared_scope = _person(partition=PARTITION_A, sub=SUB_A, resources=shares_memory)
    kept_shared = _lesson_in_tier(shared_scope, doomed, "- From the shared days")
    assert kept_shared == tmp_path / "memory" / "projects" / str(doomed.id)
    other_shared = _lesson_in_tier(shared_scope, kept, "- Stays")
    monkeypatch.delenv("COWORK_PERSONAL_RESOURCES")
    get_app_settings.cache_clear()

    assert alice.delete_project(doomed.id) is True

    assert not kept_shared.exists()
    assert (other_shared / "lessons.md").read_text().strip() == "- Stays"


def test_one_persons_planted_link_never_keeps_a_deleted_projects_other_tiers(
    db, personal_deployment, monkeypatch, caplog, tmp_path
):
    """Memory and projects shared: a link planted in place of one person's
    project tiers is refused, never followed, and logged. The deployment's
    tier for the deleted project still goes, so no one's Memories listing
    holds it, and so does every other person's kept tier."""
    import logging

    from cowork.common.settings.app_settings import get_app_settings
    from cowork.harnesses.memory.store import deployment_memory_root, project_tier_ids

    shared_both = frozenset(ALL_PERSONAL_RESOURCES - {PersonalResource.MEMORY, PersonalResource.PROJECTS})
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(shared_both)))
    get_app_settings.cache_clear()
    shared_alice = _person(partition=PARTITION_A, sub=SUB_A, resources=shared_both)
    alice = _svc(db, shared_alice)
    doomed = alice.create_project("doomed")
    shared_tier = _lesson_in_tier(shared_alice, doomed, "- Shared pallets")
    assert shared_tier == tmp_path / "memory" / "projects" / str(doomed.id)
    # Bob kept a tier of his own from when memory was per person.
    bob_personal = _person(
        partition=PARTITION_B, sub=SUB_B, resources=ALL_PERSONAL_RESOURCES - {PersonalResource.PROJECTS}
    )
    bobs_kept = _lesson_in_tier(bob_personal, doomed, "- From before")
    assert bobs_kept == tmp_path / PARTITION_B / "memory" / "users" / SUB_B / "projects" / str(doomed.id)
    # Alice's partition sorts first, so her refused chain comes before Bob's.
    outside = tmp_path / "outside-memory"
    (outside / str(doomed.id)).mkdir(parents=True)
    (outside / str(doomed.id) / "lessons.md").write_text("- Not a tier\n")
    alice_users = tmp_path / PARTITION_A / "memory" / "users" / SUB_A
    alice_users.mkdir(parents=True)
    (alice_users / "projects").symlink_to(outside, target_is_directory=True)

    with caplog.at_level(logging.WARNING, logger="cowork.services.memory"):
        assert alice.delete_project(doomed.id) is True

    assert not shared_tier.exists()
    assert doomed.id not in project_tier_ids(global_root=deployment_memory_root())
    assert not bobs_kept.exists()
    assert (alice_users / "projects").is_symlink()
    assert (outside / str(doomed.id) / "lessons.md").read_text() == "- Not a tier\n"
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "cowork.services.memory" and record.levelno == logging.WARNING
    ] == [f"memory: could not remove the tier of deleted project {doomed.id} in {alice_users / 'projects'}"]


def test_the_desktop_leaves_project_memory_to_the_project_folder(db, tmp_path):
    """Without trusted identity no project tier lives outside its project, so
    a delete removes none."""
    from uuid import uuid4

    from cowork.services.memory import remove_project_memory_tiers

    project_id = uuid4()
    stray = tmp_path / "memory" / "projects" / str(project_id)
    stray.mkdir(parents=True)

    assert remove_project_memory_tiers(scope=LOCAL_SCOPE, project_id=project_id) == 0
    assert stray.is_dir()


def test_operator_rename_leaves_a_shared_skill_store_alone(db, personal_deployment, monkeypatch, tmp_path):
    """Skills shared, projects personal: a v1.2.0 base project and Alice's own
    share a name, and a shared skill names it. The operator view's rename of
    the base project must not take the skill from Alice's."""
    from cowork.common.settings.app_settings import get_app_settings
    from cowork.services.skills import SkillService

    resources = frozenset(ALL_PERSONAL_RESOURCES - {PersonalResource.SKILLS})
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", ",".join(sorted(resources)))
    monkeypatch.setenv("COWORK_SKILLS_DIR", str(tmp_path / "skills"))
    get_app_settings.cache_clear()
    alice_scope = _person(partition=PARTITION_A, sub=SUB_A, resources=resources)
    operator = _svc(db, LOCAL_SCOPE)
    alice = _svc(db, alice_scope)
    base = operator.create_project("reports")
    mine = alice.create_project("reports")
    assert base.name == mine.name == "reports"
    SkillService(alice_scope).create_skill("sales", "Everyone's.", projects=["reports"])

    renamed = operator.update_project(base.id, name="q4")

    assert renamed.name == "q4"
    assert SkillService(alice_scope).get_skill("sales").projects == ["reports"]
