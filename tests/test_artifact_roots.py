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
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.services.artifact_roots import (
    artifacts_sources_for_project,
    artifacts_sources_for_scope,
    conversation_artifacts_base,
    project_artifacts_base,
)

ORG_A = "11111111-1111-1111-1111-111111111111"
ORG_B = "22222222-2222-2222-2222-222222222222"
USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
# A second member of ORG_A: same tenant, different person.
USER_A2 = "a2a2a2a2-a2a2-a2a2-a2a2-a2a2a2a2a2a2"


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
        lambda: called.append(1) or [],
    )

    assert artifacts_sources_for_scope(ScopedSession(session, LOCAL_SCOPE)) == []
    assert called == [1]


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
    only one taking no session, so it applies no org filter and no owner filter.
    `/api/v1/search` calls it unconditionally, in every tenancy mode.

    Nothing leaks today, for two reasons and neither of them is a check. It
    walks `_projects_root()`, which is deliberately unkeyed and so empty in org
    mode because org projects live at `<shared_root>/<org_id>/projects`
    (services/artifacts.py carries its own TODO to org-scope it). And
    `_scan_artifact_dirs` looks only for `<project>/.anton/artifacts`, the
    desktop layout, while an org project's LEGACY artifacts sit one level
    deeper under `conversations/<id>/`.

    Since ENG-2056 new org artifacts land at `<project>/.anton/artifacts` —
    the same shape the scan looks for — so for THEM only the first reason (the
    unkeyed, empty root) stands between the scan and another org's tree. That
    is a smaller loss than it sounds: artifacts are project-wide by decision
    now, so within one org there is no cross-MEMBER read left to reopen — the
    remaining stake is cross-ORG, held by the empty root.

    So this points the scan straight at an org project's real tree, defeating
    the first reason, and asserts the second still holds for the legacy layout.
    Whoever org-scopes the root or teaches the scan the conversation layout
    changes that calculus, and this fails when they do.
    """
    from cowork.services import artifacts as artifacts_service
    from cowork.services.artifact_roots import artifacts_sources_for_scan

    project = _project(session, tmp_path, name="scanned", org_id=ORG_A)
    _conversation(session, project, owner=USER_A2)
    monkeypatch.setattr(
        artifacts_service, "_projects_root", lambda: Path(project.path).parent
    )

    assert artifacts_sources_for_scan() == [], (
        "the scan reached an org project's conversation workspaces; /api/v1/search "
        "would now surface a co-member's artifacts"
    )

    # Not vacuous: the same scan, same root, does find a desktop-layout tree, so
    # the empty result above is the org layout being skipped rather than the
    # scan being broken or the root being wrong.
    (Path(project.path) / ".anton" / "artifacts").mkdir(parents=True)
    assert [s.base for s in artifacts_sources_for_scan()] == [
        Path(project.path).resolve() / ".anton" / "artifacts"
    ]
