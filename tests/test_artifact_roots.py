"""Resolving artifact roots — the one place that branches on tenancy mode.

Org mode resolves from the DB through a ScopedSession, so a project belonging to
another organization simply is not found. Desktop keeps the filesystem scan but
still resolves by id, which is what keeps a slug-addressed delete from acting on
whichever project happens to sort first.

Both modes isolate artifacts per conversation on disk (ENG-1933): desktop used
to share one project-wide folder across every conversation, which let a
concurrent sibling conversation's new artifact get misattributed to the wrong
turn's before/after diff. The project-wide folder is kept as a fallback root so
anything written before this change stays reachable.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.project import Project
from cowork.services.artifact_roots import (
    artifacts_sources_for_project,
    artifacts_sources_for_scope,
    conversation_artifacts_base,
)

ORG_A = "11111111-1111-1111-1111-111111111111"
ORG_B = "22222222-2222-2222-2222-222222222222"


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


@pytest.fixture
def org_mode(monkeypatch):
    # `_org_mode` is imported into artifact_roots as a module-level name, so
    # patching it there is what the resolver actually reads.
    monkeypatch.setattr("cowork.services.artifact_roots._org_mode", lambda: True)


def test_sources_for_project_cover_each_conversation(session, tmp_path, org_mode):
    """Org mode: the agent's workspace is a conversation, so a project's artifacts
    are spread across `conversations/<id>/.anton/artifacts` — one root each."""
    row = _project(session, tmp_path, name="proj-a", org_id=ORG_A)
    project_dir = tmp_path / ORG_A / "proj-a"
    for conv in ("conv-1", "conv-2"):
        (project_dir / "conversations" / conv / ".anton" / "artifacts").mkdir(parents=True)
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id="u"))

    sources = artifacts_sources_for_project(scoped, row.id)

    assert [s.base for s in sources] == [
        project_dir / "conversations" / "conv-1" / ".anton" / "artifacts",
        project_dir / "conversations" / "conv-2" / ".anton" / "artifacts",
    ]
    # Every root reports the SAME project: a conversation is where the bytes sit,
    # not something the client addresses by.
    assert {s.project_id for s in sources} == {str(row.id)}
    assert {s.project_name for s in sources} == {"proj-a"}


def test_sources_for_project_is_empty_before_any_conversation_wrote(session, tmp_path, org_mode):
    """The `conversations/` dir only appears once a pod has mounted one. A project
    with no cloud artifacts yet is not an error."""
    row = _project(session, tmp_path, name="fresh", org_id=ORG_A)
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id="u"))

    assert artifacts_sources_for_project(scoped, row.id) == []


def test_source_for_foreign_project_raises(session, tmp_path, org_mode):
    row = _project(session, tmp_path, name="proj-b", org_id=ORG_B)
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id="u"))

    with pytest.raises(ValueError):
        artifacts_sources_for_project(scoped, row.id)


def test_sources_for_scope_covers_only_own_org(session, tmp_path, org_mode):
    mine = _project(session, tmp_path, name="mine", org_id=ORG_A)
    theirs = _project(session, tmp_path, name="theirs", org_id=ORG_B)
    # Both need a conversation on disk to yield a root at all — a project with an
    # empty tree contributes nothing, which would make this pass vacuously.
    for row in (mine, theirs):
        (Path(row.path) / "conversations" / "c1" / ".anton" / "artifacts").mkdir(parents=True)
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id="u"))

    sources = artifacts_sources_for_scope(scoped)
    names = {s.project_name for s in sources}

    assert "mine" in names
    assert "theirs" not in names
    assert str(mine.id) in {s.project_id for s in sources}


def test_source_for_project_works_in_desktop_mode_too(session, tmp_path, monkeypatch):
    """Desktop resolves by id as well — that is what keeps slug-addressed delete
    from acting on whichever project sorts first. Desktop now also isolates
    per-conversation folders, so a project with only a legacy (pre-migration)
    folder yields exactly that one root."""
    monkeypatch.setattr("cowork.services.artifact_roots._org_mode", lambda: False)
    row = _project(session, tmp_path, name="solo", org_id=None)
    (Path(row.path) / ".anton" / "artifacts").mkdir(parents=True)
    scoped = ScopedSession(session, LOCAL_SCOPE)

    sources = artifacts_sources_for_project(scoped, row.id)

    assert [s.base for s in sources] == [tmp_path / "local" / "solo" / ".anton" / "artifacts"]
    assert sources[0].project_id == str(row.id)


def test_source_for_project_includes_desktop_conversation_folders(session, tmp_path, monkeypatch):
    """A desktop project with per-conversation folders (post-fix artifacts)
    lists the legacy folder AND each conversation's own — same shape org mode
    already uses."""
    monkeypatch.setattr("cowork.services.artifact_roots._org_mode", lambda: False)
    row = _project(session, tmp_path, name="mixed", org_id=None)
    (Path(row.path) / ".anton" / "artifacts").mkdir(parents=True)
    (Path(row.path) / "conversations" / "conv-1" / ".anton" / "artifacts").mkdir(parents=True)
    scoped = ScopedSession(session, LOCAL_SCOPE)

    sources = artifacts_sources_for_project(scoped, row.id)

    assert [s.base for s in sources] == [
        Path(row.path) / ".anton" / "artifacts",
        Path(row.path) / "conversations" / "conv-1" / ".anton" / "artifacts",
    ]


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
# `_remote_artifacts_context` resolves through `conversation_artifacts_base`, so
# this is the layout the end-of-turn diff, the cards and autopublish all key off.
# Getting it wrong is silent: the snapshot and the diff are both empty, no card
# is emitted, and the reconciler has nothing to publish — exactly what a missing
# `conversations/` segment produced on staging.

def test_conversation_base_is_scoped_to_the_conversation_in_org_mode(org_mode, tmp_path):
    base = conversation_artifacts_base(str(tmp_path / "proj"), "conv-7")

    assert base == tmp_path / "proj" / "conversations" / "conv-7" / ".anton" / "artifacts"


def test_conversation_base_is_scoped_to_the_conversation_on_desktop_too(monkeypatch, tmp_path):
    """Desktop conversations get their own folder now too (ENG-1933) — a
    concurrent sibling conversation's new artifact must never land in this
    conversation's before/after diff."""
    monkeypatch.setattr("cowork.services.artifact_roots._org_mode", lambda: False)

    base = conversation_artifacts_base(str(tmp_path / "proj"), "conv-7")

    assert base == tmp_path / "proj" / "conversations" / "conv-7" / ".anton" / "artifacts"


def test_conversation_base_agrees_with_the_listed_roots(session, tmp_path, org_mode):
    """The turn writes where the list reads. These are separate code paths and a
    drift between them shows up as artifacts that exist but never appear."""
    row = _project(session, tmp_path, name="agree", org_id=ORG_A)
    turn_base = conversation_artifacts_base(row.path, "conv-9")
    turn_base.mkdir(parents=True)
    scoped = ScopedSession(session, TenantScope(org_mode=True, org_id=ORG_A, user_id="u"))

    assert turn_base in [s.base for s in artifacts_sources_for_project(scoped, row.id)]
