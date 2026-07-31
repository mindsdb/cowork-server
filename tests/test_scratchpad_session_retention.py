"""Scratchpad namespace snapshots must be reclaimed (ENG-1124).

anton persists a pad's namespace to
`<project>/.anton/scratchpad-sessions/<conversation_id>/<pad>.pkl` so variables
survive the pad process being replaced on every turn (anton#283). Nothing else
prunes those: `delete_conversation` handled DB rows and attachment bytes but not
these, and only a whole-project delete reclaimed anything. Left alone they
accumulate one directory per conversation — on desktop in the user's own home
directory — and a namespace can hold injected `DS_*` credentials (ENG-392), so a
stale snapshot is data at rest, not just disk.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import Session

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.models.project import Project
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.services.scratchpad_sessions import (
    conversation_session_dir,
    sessions_root,
    sweep_orphan_sessions,
)


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


def _seed_snapshot(session, conversation_id) -> Path:
    """Create a snapshot dir for a conversation the way anton would."""
    project = session.get(Project, GENERAL_PROJECT_ID)
    folder = conversation_session_dir(project.path, conversation_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "forecast.pkl").write_bytes(b"pretend-namespace")
    return folder


def test_delete_conversation_removes_its_snapshots(session):
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    folder = _seed_snapshot(session, conv.id)
    assert folder.is_dir(), "precondition: snapshot dir exists"

    assert svc.delete_conversation(conv.id) is True
    assert not folder.exists(), "snapshot dir must go with the conversation"


def test_delete_conversation_leaves_other_conversations_snapshots(session):
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    keep = svc.create_conversation("keep", project_id=GENERAL_PROJECT_ID)
    doomed = svc.create_conversation("doomed", project_id=GENERAL_PROJECT_ID)
    keep_dir = _seed_snapshot(session, keep.id)
    doomed_dir = _seed_snapshot(session, doomed.id)

    svc.delete_conversation(doomed.id)

    assert not doomed_dir.exists()
    assert keep_dir.is_dir(), "a live conversation's snapshots must survive"


def test_delete_conversation_is_fine_with_no_snapshots(session):
    """The common case — most conversations never write one."""
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation("no-scratchpad", project_id=GENERAL_PROJECT_ID)
    assert svc.delete_conversation(conv.id) is True


def test_sweep_removes_orphans_and_spares_live_conversations(session):
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    live = svc.create_conversation("live", project_id=GENERAL_PROJECT_ID)
    live_dir = _seed_snapshot(session, live.id)

    # An orphan: a conversation-shaped directory with no matching row. This is the
    # backlog case — conversations deleted before the prune existed.
    project = session.get(Project, GENERAL_PROJECT_ID)
    orphan = sessions_root(project.path) / "00000000-0000-0000-0000-0000deadbeef"
    orphan.mkdir(parents=True, exist_ok=True)
    (orphan / "stale.pkl").write_bytes(b"orphaned")

    removed = sweep_orphan_sessions(session)

    assert not orphan.exists(), "orphan must be swept"
    assert live_dir.is_dir(), "a live conversation's snapshots must survive the sweep"
    assert removed >= 1

    svc.delete_conversation(live.id)


def test_sweep_is_a_noop_when_nothing_to_do(session):
    """Must be cheap and silent on a clean install — it runs on every boot."""
    assert sweep_orphan_sessions(session) >= 0


def test_removal_is_confined_to_the_snapshot_root(session, tmp_path):
    """A traversing conversation id must not direct the delete outside the root."""
    from cowork.services.scratchpad_sessions import remove_conversation_sessions

    project_dir = tmp_path / "project"
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    (outside / "precious.txt").write_text("do not delete")
    sessions_root(project_dir).mkdir(parents=True)

    # `<root>/../../outside` resolves out of the snapshot tree — must be refused.
    assert remove_conversation_sessions(project_dir, "../../outside") is False
    assert (outside / "precious.txt").exists(), "delete must not escape the root"

    # And the root itself is never a valid target.
    assert remove_conversation_sessions(project_dir, ".") is False
    assert sessions_root(project_dir).is_dir()


def test_sweep_never_touches_the_unscoped_bucket(session, tmp_path):
    """anton's `_no-session` bucket is live state, not an orphan.

    cowork-server usually supplies a `session_id`, but not always —
    `services/connectors/probe.py` builds a `ChatSession` with no conversation for
    connector probing, and a probe that execs a cell writes there. A rule of "any
    directory not in the live conversation set" would delete it on every boot.
    """
    project = session.get(Project, GENERAL_PROJECT_ID)
    bucket = sessions_root(project.path) / "_no-session"
    bucket.mkdir(parents=True, exist_ok=True)
    (bucket / "probe.pkl").write_bytes(b"live probe state")

    sweep_orphan_sessions(session)

    assert bucket.is_dir(), "the unscoped bucket must survive the sweep"


def test_sweep_only_removes_conversation_shaped_directories(session):
    """Anything that isn't a conversation id is left alone — deletes must be narrow."""
    project = session.get(Project, GENERAL_PROJECT_ID)
    root = sessions_root(project.path)
    root.mkdir(parents=True, exist_ok=True)
    keep = root / "some-future-sibling"
    keep.mkdir(exist_ok=True)
    orphan = root / "00000000-0000-0000-0000-0000deadbeef"
    orphan.mkdir(exist_ok=True)

    sweep_orphan_sessions(session)

    assert not orphan.exists(), "a real conversation-shaped orphan is still swept"
    assert keep.is_dir(), "an unrecognised directory must be left alone"


def test_uuid_needs_no_sanitising(session):
    """Pins the cross-repo invariant this module's prune depends on.

    anton sanitises the session id into the directory name (`[A-Za-z0-9._-]`); we build
    it from a raw `str(conversation_id)`. Those agree only because a UUID contains
    nothing that sanitisation would touch. If that ever stops holding, the two repos
    disagree on the path and the prune silently stops matching.
    """
    import re
    from uuid import uuid4

    raw = str(uuid4())
    sanitised = re.sub(r"[^A-Za-z0-9._-]", "_", raw).strip("._")
    assert sanitised == raw, "a UUID must survive anton's path sanitiser unchanged"


def test_non_uuid_conversation_id_is_refused_before_any_path_work(session, tmp_path):
    """The id becomes a path segment we delete recursively, so validate at the boundary.

    CodeQL flagged `py/path-injection` (high) here: the id arrives from an HTTP path
    parameter. Containment downstream did catch traversal, but late — and the function
    signature accepts `str`, so a future caller could pass anything. Parsing through
    `UUID` refuses it before a path is ever built.
    """
    from cowork.services.scratchpad_sessions import (
        conversation_session_dir,
        remove_conversation_sessions,
    )

    project_dir = tmp_path / "project"
    sessions_root(project_dir).mkdir(parents=True)

    for bad in ["../../etc", "..", ".", "", "not-a-uuid", "a/b", None]:
        assert conversation_session_dir(project_dir, bad) is None, bad
        assert remove_conversation_sessions(project_dir, bad) is False, bad

    assert sessions_root(project_dir).is_dir(), "nothing under the root was touched"


def test_canonical_uuid_is_accepted_in_any_case(session, tmp_path):
    """A real id still works, and is canonicalised so casing can't fork the directory."""
    from uuid import uuid4

    from cowork.services.scratchpad_sessions import conversation_session_dir

    u = uuid4()
    lower = conversation_session_dir(tmp_path, str(u))
    upper = conversation_session_dir(tmp_path, str(u).upper())
    assert lower is not None and upper is not None
    assert lower == upper, "casing must not produce two directories for one conversation"
    assert lower.name == str(u)


def test_delete_turn_invalidates_the_snapshot(session):
    """Rewinding history must rewind the scratchpad too.

    Without this, truncating the visible history leaves the snapshot at the state the
    *deleted* turns produced, so a resend reloads variables created by a turn the user
    just removed — the history and the agent's state silently disagree.
    """
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation("topic", project_id=GENERAL_PROJECT_ID)
    svc.save_user_message(conv.id, "do a thing")
    svc.save_assistant_turn(conv.id, "did the thing", [])
    folder = _seed_snapshot(session, conv.id)
    assert folder.is_dir(), "precondition"

    svc.delete_turn(conv.id, 0)

    assert not folder.exists(), "snapshot must not outlive the turns that produced it"


def test_sweep_reclaims_a_directory_left_behind_by_a_project_move(session):
    """Liveness is per (project, conversation), not global by conversation id.

    A conversation moved from project A to B keeps a directory under A. Checking
    liveness globally marks it live forever, and conversation delete only cleans the
    current project — so the one under A would be stranded.
    """
    from cowork.services.projects import ProjectService

    projects = ProjectService(ScopedSession(session, LOCAL_SCOPE))
    other = projects.create_project("eng1124-move-test")
    svc = ConversationService(ScopedSession(session, LOCAL_SCOPE))
    conv = svc.create_conversation("mover", project_id=GENERAL_PROJECT_ID)

    stale = _seed_snapshot(session, conv.id)          # written while in General
    conv.project_id = other.id                        # ...then moved
    session.add(conv)
    session.commit()
    fresh = conversation_session_dir(other.path, conv.id)
    fresh.mkdir(parents=True, exist_ok=True)
    (fresh / "forecast.pkl").write_bytes(b"current")

    sweep_orphan_sessions(session)

    assert not stale.exists(), "the directory under the old project must be reclaimed"
    assert fresh.is_dir(), "the directory under the current project must survive"

    svc.delete_conversation(conv.id)
