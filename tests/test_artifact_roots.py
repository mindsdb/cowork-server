"""Resolving artifact roots — the one place that branches on tenancy mode.

Org mode resolves from the DB through a ScopedSession, so a project belonging to
another organization simply is not found. Desktop keeps the filesystem scan but
still resolves by id, which is what keeps a slug-addressed delete from acting on
whichever project happens to sort first.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.services import artifact_identity
from cowork.services.artifact_draft_review import disable_draft_review, enable_draft_review
from cowork.services.artifact_roots import (
    artifacts_sources_for_project,
    artifacts_sources_for_scope,
    conversation_artifacts_base,
    owned_conversation_ids,
    plan_scope_roots,
    project_artifacts_base,
)

ORG_A = "11111111-1111-1111-1111-111111111111"
ORG_B = "22222222-2222-2222-2222-222222222222"
USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
# A second member of ORG_A: same tenant, different person.
USER_A2 = "a2a2a2a2-a2a2-a2a2-a2a2-a2a2a2a2a2a2"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def _project(session, tmp_path, *, name: str, org_id: str | None) -> Project:
    path = tmp_path / (org_id or "local") / name
    path.mkdir(parents=True, exist_ok=True)
    row = Project(id=uuid.uuid4(), name=name, path=str(path), org_id=org_id)
    session.add(row)
    session.commit()
    return row


def _conversation(session, project, *, owner: str) -> uuid.UUID:
    """A conversation row plus the workspace it writes into.

    The resolver joins the two: the directory name is the conversation's id and
    the row is what says who owns what is inside, so a test that writes only the
    directory is testing a shape the cloud never produces.
    """
    row = Conversation(
        id=uuid.uuid4(),
        topic="chat",
        project_id=project.id,
        org_id=project.org_id,
        created_by=owner,
    )
    session.add(row)
    session.commit()
    (Path(project.path) / "conversations" / str(row.id) / ".anton" / "artifacts").mkdir(
        parents=True
    )
    return row.id


def _base_for(project, conversation_id) -> Path:
    return Path(project.path) / "conversations" / str(conversation_id) / ".anton" / "artifacts"


def _project_base(project) -> Path:
    """The project-level base — where new artifacts land since ENG-2056."""
    return Path(project.path) / ".anton" / "artifacts"


@pytest.fixture
def org_mode(monkeypatch):
    # `_org_mode` is imported into artifact_roots as a module-level name, so
    # patching it there is what the resolver actually reads.
    monkeypatch.setattr("cowork.services.artifact_roots._org_mode", lambda: True)


def test_sources_for_project_cover_the_project_base_and_each_conversation(session, tmp_path, org_mode):
    """Org mode: new artifacts land in the shared project base (ENG-2056), while
    artifacts written before that change sit under
    `conversations/<id>/.anton/artifacts` — so the roots are the project base
    plus one legacy root per owned conversation."""
    row = _project(session, tmp_path, name="proj-a", org_id=ORG_A)
    convs = [_conversation(session, row, owner=USER_A) for _ in range(2)]
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A))

    sources = artifacts_sources_for_project(scoped, row.id)

    assert {s.base for s in sources} == {_project_base(row)} | {_base_for(row, cid) for cid in convs}
    # Every root reports the SAME project: a conversation is where the bytes sit,
    # not something the client addresses by.
    assert {s.project_id for s in sources} == {str(row.id)}
    assert {s.project_name for s in sources} == {"proj-a"}


def test_sources_for_project_skip_another_members_legacy_conversations(session, tmp_path, org_mode):
    """A project belongs to the organization; the LEGACY conversation workspaces
    under it belong to whoever started them. Listing every workspace under a
    shared project is how one member came to see another member's whole
    workspace tree — the shared project base (ENG-2056) is the deliberate,
    artifacts-only exception, not that."""
    row = _project(session, tmp_path, name="shared", org_id=ORG_A)
    mine = _conversation(session, row, owner=USER_A)
    theirs = _conversation(session, row, owner=USER_A2)
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A))

    bases = {s.base for s in artifacts_sources_for_project(scoped, row.id)}

    assert bases == {_project_base(row), _base_for(row, mine)}
    assert _base_for(row, theirs) not in bases


def test_project_base_is_shared_across_members(session, tmp_path, org_mode):
    """ENG-2056, the product decision: project-wide artifact visibility. A member
    who owns no conversation in the project still reads the project base, so the
    panel shows them every artifact the project's tasks produced."""
    row = _project(session, tmp_path, name="shared-wide", org_id=ORG_A)
    _conversation(session, row, owner=USER_A)  # someone else's task wrote here
    other = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A2))

    assert _project_base(row) in {s.base for s in artifacts_sources_for_project(other, row.id)}


def test_a_directory_naming_no_owned_conversation_is_skipped(session, tmp_path, org_mode):
    """Everything the pod writes is `str(conversation_id)`, so a name that will
    not parse names nothing, and a well-formed id whose row is gone names
    nothing either. Both get no reader.

    Deliberately unlike `_conversation_workspace_ok` on project files, which
    treats an unrecognized name as a shared file: that tree really does hold
    shared files beside the workspaces, and this one holds nothing else.
    """
    row = _project(session, tmp_path, name="stray", org_id=ORG_A)
    mine = _conversation(session, row, owner=USER_A)
    for stray in ("not-a-uuid", str(uuid.uuid4())):
        (Path(row.path) / "conversations" / stray / ".anton" / "artifacts").mkdir(parents=True)
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A))

    assert {s.base for s in artifacts_sources_for_project(scoped, row.id)} == {
        _project_base(row), _base_for(row, mine)
    }


def test_a_moved_tasks_legacy_root_stays_with_its_owner(session, tmp_path, org_mode):
    """Moving a task to another project changes its row's project and leaves
    its workspace directory where it was. The owner filter reads which
    conversations the caller owns before it walks the file store, so that
    read must not be narrowed to the project being walked."""
    source = _project(session, tmp_path, name="moved-from", org_id=ORG_A)
    destination = _project(session, tmp_path, name="moved-to", org_id=ORG_A)
    moved = _conversation(session, source, owner=USER_A)
    row = session.get(Conversation, moved)
    row.project_id = destination.id
    session.add(row)
    session.commit()
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A))

    assert _base_for(source, moved) in {s.base for s in artifacts_sources_for_project(scoped, source.id)}


def test_the_owner_filter_reads_only_the_callers_own_conversations_in_their_organization(
    session, tmp_path, org_mode
):
    mine = _conversation(session, _project(session, tmp_path, name="own", org_id=ORG_A), owner=USER_A)
    _conversation(session, _project(session, tmp_path, name="co-member", org_id=ORG_A), owner=USER_A2)
    # The same person's id in another organization is still another tenant.
    _conversation(session, _project(session, tmp_path, name="elsewhere", org_id=ORG_B), owner=USER_A)
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A))

    owned = owned_conversation_ids(scoped)

    assert mine in owned
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        others = raw.exec(
            select(Conversation.id).where(
                (Conversation.created_by != USER_A) | (Conversation.org_id != ORG_A)
            )
        ).all()
    assert owned.isdisjoint(others)


def test_sources_for_project_is_just_the_project_base_before_any_conversation_wrote(
    session, tmp_path, org_mode
):
    """The legacy `conversations/` dir only appears once a pre-ENG-2056 pod
    mounted one; its absence is not an error. The project base is always a root
    — the listing tolerates it not existing on disk yet, and the pre-turn
    staging creates it before the first pod needs the mount."""
    row = _project(session, tmp_path, name="fresh", org_id=ORG_A)
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A))

    assert [s.base for s in artifacts_sources_for_project(scoped, row.id)] == [_project_base(row)]


def test_source_for_foreign_project_raises(session, tmp_path, org_mode):
    row = _project(session, tmp_path, name="proj-b", org_id=ORG_B)
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A))

    with pytest.raises(ValueError):
        artifacts_sources_for_project(scoped, row.id)


def test_sources_for_scope_covers_only_own_org(session, tmp_path, org_mode):
    mine = _project(session, tmp_path, name="mine", org_id=ORG_A)
    theirs = _project(session, tmp_path, name="theirs", org_id=ORG_B)
    # Both need a conversation to yield a root at all: a project with an empty
    # tree contributes nothing, which would make this pass vacuously. The other
    # org's conversation belongs to the same user id on purpose: the org filter
    # has to hold it out, not the owner filter.
    for row in (mine, theirs):
        _conversation(session, row, owner=USER_A)
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A))

    sources = artifacts_sources_for_scope(scoped)
    names = {s.project_name for s in sources}

    assert "mine" in names
    assert "theirs" not in names
    assert str(mine.id) in {s.project_id for s in sources}


def test_source_for_project_works_in_desktop_mode_too(session, tmp_path, monkeypatch):
    """Desktop resolves by id as well — that is what keeps slug-addressed delete
    from acting on whichever project sorts first."""
    monkeypatch.setattr("cowork.services.artifact_roots._org_mode", lambda: False)
    row = _project(session, tmp_path, name="solo", org_id=None)
    scoped = ScopedSession(session, LOCAL_SCOPE)

    sources = artifacts_sources_for_project(scoped, row.id)

    # Exactly one, and no `conversations` segment: on the desktop the workspace IS
    # the project directory and every conversation shares one artifacts folder.
    assert [s.base for s in sources] == [tmp_path / "local" / "solo" / ".anton" / "artifacts"]
    assert sources[0].project_id == str(row.id)


def test_sources_for_scope_falls_back_to_the_scan_in_desktop_mode(session, monkeypatch):
    """Local mode must keep listing exactly what it listed before — the scan,
    not a DB read (a desktop install has no org rows to scope by)."""
    monkeypatch.setattr("cowork.services.artifact_roots._org_mode", lambda: False)
    called = []
    monkeypatch.setattr(
        "cowork.services.artifact_roots.artifacts_sources_for_scan",
        lambda *, scope: called.append(scope) or [],
    )

    assert artifacts_sources_for_scope(ScopedSession(session, LOCAL_SCOPE)) == []
    assert called == [LOCAL_SCOPE]


# ─── the turn-end root ─────────────────────────────────────────────────────
#
# `_remote_artifacts_context` resolves through `project_artifacts_base`
# (ENG-2056), so this is the layout the end-of-turn diff, the cards and
# autopublish all key off. Getting it wrong is silent: the snapshot and the
# diff are both empty, no card is emitted, and the reconciler has nothing to
# publish — exactly what a missing `conversations/` segment produced on staging
# under the old per-conversation layout.

def test_project_base_has_no_conversation_segment(tmp_path):
    """ENG-2056: new artifacts land at the project level in both modes — the pod
    mounts this dir at /project-artifacts, so it must match what anton writes."""
    assert project_artifacts_base(str(tmp_path / "proj")) == (
        tmp_path / "proj" / ".anton" / "artifacts"
    )


def test_legacy_conversation_base_is_scoped_to_the_conversation_in_org_mode(org_mode, tmp_path):
    base = conversation_artifacts_base(str(tmp_path / "proj"), "conv-7")

    assert base == tmp_path / "proj" / "conversations" / "conv-7" / ".anton" / "artifacts"


def test_conversation_base_ignores_the_conversation_on_desktop(monkeypatch, tmp_path):
    """Desktop's workspace IS the project dir — every conversation shares one
    artifacts folder, so the id is accepted and dropped rather than refused."""
    monkeypatch.setattr("cowork.services.artifact_roots._org_mode", lambda: False)

    base = conversation_artifacts_base(str(tmp_path / "proj"), "conv-7")

    assert base == tmp_path / "proj" / ".anton" / "artifacts"


def test_turn_base_agrees_with_the_listed_roots(session, tmp_path, org_mode):
    """The turn writes where the list reads. These are separate code paths and a
    drift between them shows up as artifacts that exist but never appear. Both
    the ENG-2056 project base (where new turns write) and the legacy
    per-conversation base (where old artifacts still sit) must be listed."""
    row = _project(session, tmp_path, name="agree", org_id=ORG_A)
    conversation_id = _conversation(session, row, owner=USER_A)
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A))

    listed = [s.base for s in artifacts_sources_for_project(scoped, row.id)]

    assert project_artifacts_base(row.path) in listed
    assert conversation_artifacts_base(row.path, conversation_id) in listed


def _write_artifact(base: Path, slug: str) -> None:
    folder = base / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.html").write_text("<html>report</html>")
    (folder / "metadata.json").write_text(json.dumps({"slug": slug, "type": "html-app"}))


def test_listing_shows_project_base_artifacts_to_every_member_and_keeps_legacy_ones(
    session, tmp_path, org_mode
):
    """End-to-end through `list_artifacts` (ENG-2056): an artifact in the shared
    project base appears in a DIFFERENT member's panel, while a legacy
    per-conversation artifact stays visible to its owner and stays hidden from
    everyone else."""
    from cowork.services.artifacts import list_artifacts

    row = _project(session, tmp_path, name="panel", org_id=ORG_A)
    conversation_id = _conversation(session, row, owner=USER_A)
    _write_artifact(_base_for(row, conversation_id), "legacy-report")
    _write_artifact(_project_base(row), "shared-report")
    project_id = row.id

    def _slugs_for(user_id: str) -> set[str]:
        # One raw session per caller: a session is pinned to the first tenant
        # scope it is wrapped with, so the two members cannot share one.
        engine = get_engine(get_app_settings().database.uri)
        with Session(engine) as caller_session:
            scoped = ScopedSession(
                caller_session, TenantScope(org_mode=True, org_id=ORG_A, user_id=user_id)
            )
            return {c["slug"] for c in list_artifacts(artifacts_sources_for_project(scoped, project_id))}

    assert _slugs_for(USER_A) == {"legacy-report", "shared-report"}
    assert _slugs_for(USER_A2) == {"shared-report"}


def test_the_unfiltered_scan_yields_nothing_for_an_org_project(session, tmp_path, org_mode, monkeypatch):
    """A tripwire, not a behaviour test.

    `artifacts_sources_for_scan` is this module's third public resolver and the
    only one taking no session, so it applies no owner filter. It walks
    `_projects_root(scope=...)`, the scope's own projects root: an org's is
    `<shared_root>/<org_id>/projects`, so the org filter is the root itself.
    `/api/v1/search` skips the scan in org mode explicitly.

    The owner filter is not a check either. `_scan_artifact_dirs` looks only
    for `<project>/.anton/artifacts`, the desktop layout, while an org
    project's LEGACY artifacts sit one level deeper under `conversations/<id>/`.
    Since ENG-2056 new org artifacts land at `<project>/.anton/artifacts`,
    project-wide by decision, so there is no cross-MEMBER read to reopen there.

    So this points the scan straight at an org project's real tree and asserts
    the legacy layout is still skipped. Whoever teaches the scan the
    conversation layout changes that calculus, and this fails when they do.
    """
    from cowork.services import artifacts as artifacts_service
    from cowork.services.artifact_roots import artifacts_sources_for_scan

    org_scope = TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A)
    project = _project(session, tmp_path, name="scanned", org_id=ORG_A)
    _conversation(session, project, owner=USER_A2)
    monkeypatch.setattr(
        artifacts_service, "_projects_root", lambda *, scope: Path(project.path).parent
    )

    assert artifacts_sources_for_scan(scope=org_scope) == [], (
        "the scan reached an org project's conversation workspaces; a co-member's "
        "legacy artifacts would surface wherever the scan is used"
    )

    # Not vacuous: the same scan, same root, does find a desktop-layout tree, so
    # the empty result above is the org layout being skipped rather than the
    # scan being broken or the root being wrong.
    (Path(project.path) / ".anton" / "artifacts").mkdir(parents=True)
    assert [s.base for s in artifacts_sources_for_scan(scope=org_scope)] == [
        Path(project.path).resolve() / ".anton" / "artifacts"
    ]


# ─── personal partitions ───────────────────────────────────────────────────

PARTITION_A = "44e92829-e701-599b-88f9-69f5bf1165eb"
PARTITION_B = "55e92829-e701-599b-88f9-69f5bf1165eb"


def _personal(*, partition: str, user: str, resources=None) -> TenantScope:
    from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES

    return TenantScope(
        personal=True,
        org_id=partition,
        user_id=user,
        personal_resources=ALL_PERSONAL_RESOURCES if resources is None else resources,
    )


def test_sources_for_scope_lists_only_the_persons_own_projects(session, tmp_path, monkeypatch):
    """A person's list reads their project rows, as org mode does, so each
    card carries its project id; the scan of their partition's projects root
    would leave cards path-addressed. Another person's project is not among
    them."""
    monkeypatch.setattr("cowork.services.artifact_roots._org_mode", lambda: False)
    monkeypatch.setattr(
        "cowork.services.artifact_roots.artifacts_sources_for_scan",
        lambda *, scope: pytest.fail("a personal partition must not fall back to the scan"),
    )
    mine = _project(session, tmp_path, name="alice-notes", org_id=PARTITION_A)
    _project(session, tmp_path, name="bob-notes", org_id=PARTITION_B)

    sources = artifacts_sources_for_scope(ScopedSession(session, _personal(partition=PARTITION_A, user=USER_A)))

    assert [s.project_name for s in sources] == ["alice-notes"]
    # The in-process layout: one project-wide root, no conversation segment.
    assert [s.base for s in sources] == [_project_base(mine)]
    assert sources[0].project_id == str(mine.id)


def test_sources_for_scope_scans_when_projects_are_shared(session, monkeypatch):
    """Projects left off the personal list stay in the unkeyed root, so the
    person's list is the desktop scan, as on today's shared stack."""
    monkeypatch.setattr("cowork.services.artifact_roots._org_mode", lambda: False)
    called = []
    monkeypatch.setattr(
        "cowork.services.artifact_roots.artifacts_sources_for_scan",
        lambda *, scope: called.append("scan") or [],
    )
    # Earlier tests leave project rows behind in the shared database; the
    # adopted-folder half of the desktop path is not what this pins.
    monkeypatch.setattr(
        "cowork.services.artifact_roots._adopted_root",
        lambda _project: called.append("adopted") or None,
    )

    scope = _personal(partition=PARTITION_A, user=USER_A, resources=frozenset())
    plan = plan_scope_roots(ScopedSession(session, scope))
    assert plan.scan and plan.scan_scope == scope and plan.projects == ()
    assert artifacts_sources_for_scope(ScopedSession(session, scope)) == []
    assert called[0] == "scan"


# ─── the review resolver ───────────────────────────────────────────────────
#
# `artifact_scope.review_artifact_for_request` is the one lookup that searches
# a co-member's legacy conversation root, and only the owner's draft-review
# grant makes a hit there visible. The route tests stub it, so these drive it
# for real: org tenancy, database rows, and folders on disk.

REVIEWED = "0f9e8d7c6b5a49388271605f4e3d2c1b"


@pytest.fixture
def owners_legacy_artifact(session, tmp_path, org_deployment, cleanup_tmp_projects):
    """USER_A's artifact in their own legacy conversation root of an ORG_A
    project. Returns (project, folder)."""
    row = _project(session, tmp_path, name="reviewed", org_id=ORG_A)
    folder = _base_for(row, _conversation(session, row, owner=USER_A)) / "brief"
    folder.mkdir()
    (folder / "metadata.json").write_text(
        json.dumps({"id": REVIEWED, "type": "html-app", "primary": "index.html"})
    )
    (folder / "index.html").write_text("<p>brief</p>")
    artifact_identity._clear_identity_indexes()
    yield row, folder
    artifact_identity._clear_identity_indexes()


async def _review(project, *, org_id: str, user_id: str, artifact_id: str = REVIEWED):
    from cowork.api.v1.artifact_scope import review_artifact_for_request

    return await review_artifact_for_request(
        TenantScope(org_mode=True, org_id=org_id, user_id=user_id), project.id.hex, artifact_id
    )


async def _refusal(project, **who) -> tuple[int, str]:
    with pytest.raises(HTTPException) as refused:
        await _review(project, **who)
    return refused.value.status_code, refused.value.detail


async def test_the_owner_reviews_their_legacy_artifact_as_their_own(owners_legacy_artifact):
    row, folder = owners_legacy_artifact

    resolved = await _review(row, org_id=ORG_A, user_id=USER_A)

    assert resolved.is_own is True
    assert resolved.folder.name == folder.name
    assert resolved.source.base == folder.parent


async def test_a_co_member_with_the_owners_grant_reviews_it_as_a_reviewer(owners_legacy_artifact):
    row, folder = owners_legacy_artifact
    enable_draft_review(folder, org_id=ORG_A, enabled_by=USER_A)

    resolved = await _review(row, org_id=ORG_A, user_id=USER_A2)

    assert resolved.is_own is False
    assert resolved.folder.name == folder.name


async def test_a_co_member_without_a_grant_gets_the_same_404_as_a_missing_artifact(
    owners_legacy_artifact,
):
    row, _folder = owners_legacy_artifact

    hidden = await _refusal(row, org_id=ORG_A, user_id=USER_A2)
    missing = await _refusal(row, org_id=ORG_A, user_id=USER_A2, artifact_id=uuid.uuid4().hex)

    assert hidden == missing == (404, "Artifact not found")


async def test_a_grant_never_reaches_another_organization(owners_legacy_artifact):
    row, folder = owners_legacy_artifact
    enable_draft_review(folder, org_id=ORG_A, enabled_by=USER_A)

    status_code, _detail = await _refusal(row, org_id=ORG_B, user_id=USER_B)

    assert status_code == 404
    # A grant that names another organization, as one left behind when the
    # project row moved, opens nothing to the members of this one.
    disable_draft_review(folder)
    enable_draft_review(folder, org_id=ORG_B, enabled_by=USER_A)
    assert await _refusal(row, org_id=ORG_A, user_id=USER_A2) == (404, "Artifact not found")


def _count_hops(monkeypatch, *modules) -> list[str]:
    """Record each run_db and run_efs call the named modules make, in order."""
    from cowork.db import units

    hops: list[str] = []

    async def run_db(fn, *, scope):
        hops.append("run_db")
        return await units.run_db(fn, scope=scope)

    async def run_efs(fn, *, scope, deadline=True):
        hops.append("run_efs")
        return await units.run_efs(fn, scope=scope, deadline=deadline)

    for module in modules:
        monkeypatch.setattr(module, "run_db", run_db)
        monkeypatch.setattr(module, "run_efs", run_efs)
    return hops


async def test_a_review_lookup_waits_for_the_database_once_and_the_file_store_once(
    owners_legacy_artifact, monkeypatch,
):
    """The project row and the conversations the caller owns in one database
    unit, then the roots, the owner filter, the folder and a reviewer's grant
    in one file store call. Each wait for a slot can refuse the request, so a
    lookup that waited again after its file store work could be refused with
    that work done."""
    from cowork.api.v1 import artifact_scope

    row, folder = owners_legacy_artifact
    enable_draft_review(folder, org_id=ORG_A, enabled_by=USER_A)
    hops = _count_hops(monkeypatch, artifact_scope)

    for user_id, is_own in ((USER_A, True), (USER_A2, False)):
        hops.clear()
        assert (await _review(row, org_id=ORG_A, user_id=user_id)).is_own is is_own
        assert hops == ["run_db", "run_efs"], user_id


async def test_a_project_list_reads_its_roots_and_cards_in_one_file_store_call(
    owners_legacy_artifact, monkeypatch,
):
    from cowork.api.v1 import artifact_scope
    from cowork.api.v1.endpoints import artifacts

    row, _folder = owners_legacy_artifact
    hops = _count_hops(monkeypatch, artifact_scope, artifacts)
    caller = TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A)

    cards = await artifacts.list_artifacts(caller, project_id=row.id, project_path=None)

    # The owner's legacy root is still listed: the owner filter kept it.
    assert [card["slug"] for card in cards] == ["brief"]
    assert hops == ["run_db", "run_efs", "run_db"]
