"""A remote turn's end diff reads the pod's list of the artifacts it touched.

anton names the artifacts a cloud turn's tools touched on ``turn_completed``.
With that list the diff claims and checks only those folders: a sibling turn's
artifact is never this turn's, and the content of every other artifact in the
project is left unread. A pod that sends no list gets today's diff, where
provenance decides what the turn created.
"""
from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.models.queued_agent_repair import QueuedAgentRepair
from cowork.services import task_objects
from cowork.services.task_objects import (
    snapshot_artifact_state,
    tracked_artifacts_from_reply,
    turn_artifact_changes,
)
from test_artifact_resolution_off_the_loop import seeded_artifact  # noqa: F401  (a fixture)
from test_responses_remote_backend import (
    _artifact,
    _conversation_at,
    _FakeBuffer,
    _remote_handler,
    responses_mod,
)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"artifacts_touched": ["a", "b"]}, {"a", "b"}),
        ({"artifacts_touched": []}, set()),
        ({}, None),
        ({"artifacts_touched": "a"}, None),
        ({"artifacts_touched": ["a", 7]}, None),
        ({"artifacts_touched": None}, None),
        ({"artifacts_touched": [f"a{i}" for i in range(1001)]}, None),
    ],
    ids=["list", "empty", "older-pod", "string", "non-string", "null", "too-many"],
)
def test_the_pods_list_is_read_only_when_it_is_a_list_of_names(data, expected):
    assert tracked_artifacts_from_reply(data) == expected


class _ContentReads:
    """Records the files statted inside artifact folders, other than the
    metadata the folder listing checks."""

    def __init__(self, monkeypatch, base: Path):
        self.paths: list[Path] = []
        self.counting = False
        base_text = str(base)
        for name in ("stat", "lstat", "scandir"):
            real = getattr(os, name)

            def recorded(path, *args, _real=real, **kwargs):
                if self.counting and isinstance(path, (str, os.PathLike)):
                    text = os.fspath(path)
                    if text.startswith(base_text) and not text.endswith("metadata.json"):
                        self.paths.append(Path(text))
                return _real(path, *args, **kwargs)

            monkeypatch.setattr(os, name, recorded)

    def folders(self, base: Path) -> set[str]:
        """The artifact folders whose own files were read."""
        return {path.relative_to(base).parts[0] for path in self.paths if len(path.relative_to(base).parts) > 1}


def test_with_the_pods_list_only_its_folders_content_is_read(tmp_path, monkeypatch):
    conversation = uuid4()
    base = tmp_path / ".anton" / "artifacts"
    for index in range(200):
        _artifact(base, f"old-{index}")
    before, before_mtimes = snapshot_artifact_state(base)
    edited = _artifact(base, "old-7", body="<html>edited</html>")
    os.utime(edited / "report.html", ns=(10**19, 10**19))
    _artifact(base, "made", conversation_id=conversation)
    _artifact(base, "siblings", conversation_id=conversation)  # provenance alone would claim it

    reads = _ContentReads(monkeypatch, base)
    reads.counting = True
    changes = turn_artifact_changes(
        conversation_id=conversation, artifacts_base=base, before=before, before_mtimes=before_mtimes,
        tracked_new={"made", "old-7"}, tracked_edits={"made", "old-7"}, attribute_by_provenance=True,
    )
    reads.counting = False

    assert changes.created == ["made"]
    assert changes.touched == {"made", "old-7"}
    # Only the two named folders' files are read. The other 199 and the
    # sibling's get the listing's metadata check and the queued-repair check
    # under `.revisions`, nothing else.
    untouched_reads = [
        path for path in reads.paths
        if len(path.relative_to(base).parts) > 1
        and path.relative_to(base).parts[0] not in {"made", "old-7"}
        and path.relative_to(base).parts[1] != ".revisions"
    ]
    assert untouched_reads == []
    assert {"made", "old-7"} <= reads.folders(base)


def test_without_the_pods_list_every_folder_is_checked_for_edits(tmp_path, monkeypatch):
    """An older pod: today's diff, which reads every folder's content."""
    conversation = uuid4()
    base = tmp_path / ".anton" / "artifacts"
    for index in range(20):
        _artifact(base, f"old-{index}")
    before, before_mtimes = snapshot_artifact_state(base)

    reads = _ContentReads(monkeypatch, base)
    reads.counting = True
    turn_artifact_changes(
        conversation_id=conversation, artifacts_base=base, before=before, before_mtimes=before_mtimes,
        attribute_by_provenance=True,
    )
    reads.counting = False

    assert {f"old-{index}" for index in range(20)} <= reads.folders(base)


async def _remote_turn(monkeypatch, tmp_path, completed: dict, *, queued: tuple[str, ...] = ()) -> list[str]:
    """Run a remote turn whose pod writes `mine` and a sibling's `theirs`,
    both naming this conversation in their provenance, and return the slugs
    it indexed. `queued` are folders recorded as holding a repair queued to
    the conversation."""
    saved: dict = {}
    handler = _remote_handler(monkeypatch, saved)
    project_dir = tmp_path / "proj"
    artifacts_base = project_dir / ".anton" / "artifacts"
    artifacts_base.mkdir(parents=True)
    conversation = _conversation_at(project_dir)
    monkeypatch.setattr(
        responses_mod.ResponsesHandler, "_remote_artifacts_context",
        staticmethod(lambda session, conv_id: (
            conversation, artifacts_base, str(conversation.project_id), conversation.project.name,
        )),
    )
    indexed: dict = {}

    def spy_index(session, **kwargs):
        indexed["slugs"] = kwargs["slugs"]

    monkeypatch.setattr(task_objects, "record_new_artifacts", spy_index)

    async def fake_autopublish(base, scope, *, touched, **kwargs):
        return set(touched)

    monkeypatch.setattr("cowork.services.artifact_autopublish.autopublish_project_artifacts", fake_autopublish)
    conv_id = uuid4()
    _record_queued(conv_id, queued)

    async def fake_replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        _artifact(artifacts_base, "mine", conversation_id=conv_id)
        _artifact(artifacts_base, "theirs", conversation_id=conv_id)
        yield "turn_completed", completed

    monkeypatch.setattr(responses_mod, "stream_remote_replies", fake_replies)
    await handler._produce_remote(
        conv_id=conv_id, input_text="hi", original_content="hi",
        model="anton", harness_id="anton", buffer=_FakeBuffer(),
    )
    return indexed.get("slugs", [])


async def test_a_remote_turn_claims_only_what_the_pod_named(monkeypatch, tmp_path):
    """A sibling turn's folder can carry this conversation's provenance (it
    copied the folder, or anton crashed between writes); the pod's list is
    what this turn's own tools did."""
    assert await _remote_turn(monkeypatch, tmp_path, {"artifacts_touched": ["mine"]}) == ["mine"]


async def test_a_remote_turn_from_an_older_pod_keeps_todays_diff(monkeypatch, tmp_path):
    assert await _remote_turn(monkeypatch, tmp_path, {}) == ["mine", "theirs"]


def _queue_repair(folder: Path, conversation_id) -> str:
    """Queue an agent repair in `folder` for `conversation_id`, as the repair
    route does, and return its id."""
    from cowork.services.artifact_identity import ensure_full_id
    from cowork.services.artifact_revisions import create_agent_repair, current_source

    artifact_id, metadata = ensure_full_id(folder)
    initial = current_source(folder, metadata, artifact_id)
    return create_agent_repair(
        folder, metadata, artifact_id,
        expected_revision_id=initial["revision"]["id"], comment_thread_id="thread-1",
        selector=None, thread=[{"text": "Please check this"}], conversation_id=str(conversation_id),
    )["repair"]["id"]


def test_with_the_queued_repairs_the_diff_lists_nothing_and_still_finishes_a_no_change_repair(tmp_path, monkeypatch):
    from cowork.services.artifact_revisions import agent_repair_detail

    conversation = uuid4()
    base = tmp_path / ".anton" / "artifacts"
    for index in range(200):
        _artifact(base, f"old-{index}")
    repair_id = _queue_repair(base / "old-50", conversation)
    before, before_mtimes = snapshot_artifact_state(base)
    _artifact(base, "made", conversation_id=conversation)

    listings = []
    real_scandir = os.scandir

    def scandir(path=".", *args, **kwargs):
        listings.append(path)
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", scandir)
    reads = _ContentReads(monkeypatch, base)
    reads.counting = True
    changes = turn_artifact_changes(
        conversation_id=conversation, artifacts_base=base, before=before, before_mtimes=before_mtimes,
        tracked_new={"made"}, tracked_edits={"made"}, repair_slugs={"old-50"},
    )
    reads.counting = False

    assert changes.created == ["made"]
    # The repair turn changed nothing in old-50, and its repair still leaves
    # the queued state.
    assert agent_repair_detail(base / "old-50", repair_id)["repair"]["status"] == "no_change"
    assert base not in [Path(p) for p in listings if isinstance(p, (str, os.PathLike))]
    assert reads.folders(base) <= {"made", "old-50"}


def _record_queued(conversation_id, slugs) -> None:
    from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
    from cowork.services.queued_repairs import record_queued_repair

    with Session(get_engine(get_app_settings().database.uri)) as raw:
        for slug in slugs:
            record_queued_repair(ScopedSession(raw, LOCAL_SCOPE), conversation_id=conversation_id, slug=slug)


def _queued_slugs() -> set[str]:
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        return set(raw.exec(select(QueuedAgentRepair.slug)).all())


async def test_a_remote_turn_reads_its_queued_repairs_and_drops_them(monkeypatch, tmp_path):
    seen: dict = {}
    diff = task_objects.turn_artifact_changes

    def spy_diff(**kwargs):
        seen["repair_slugs"] = kwargs["repair_slugs"]
        return diff(**kwargs)

    monkeypatch.setattr(task_objects, "turn_artifact_changes", spy_diff)
    slug = f"repaired-{uuid4().hex[:8]}"

    await _remote_turn(monkeypatch, tmp_path, {"artifacts_touched": ["mine"]}, queued=(slug,))

    assert seen["repair_slugs"] == {slug}
    assert slug not in _queued_slugs()


async def test_an_older_pods_turn_leaves_the_queued_repairs_unread(monkeypatch, tmp_path):
    """Without the pod's list the diff checks every folder anyway, so the
    rows stay for a turn that can use them."""
    seen: dict = {}
    diff = task_objects.turn_artifact_changes

    def spy_diff(**kwargs):
        seen["repair_slugs"] = kwargs["repair_slugs"]
        return diff(**kwargs)

    monkeypatch.setattr(task_objects, "turn_artifact_changes", spy_diff)
    slug = f"repaired-{uuid4().hex[:8]}"

    await _remote_turn(monkeypatch, tmp_path, {}, queued=(slug,))

    assert seen["repair_slugs"] is None
    assert slug in _queued_slugs()


async def test_a_repair_request_records_its_folder_for_the_conversation(seeded_artifact):  # noqa: F811
    import httpx

    from cowork.server import create_app
    from cowork.services.artifact_identity import ensure_full_id
    from cowork.services.artifact_revisions import current_source
    from test_artifact_resolution_off_the_loop import ARTIFACT_ID, WORKSPACE

    artifact_id, metadata = ensure_full_id(seeded_artifact)
    revision = current_source(seeded_artifact, metadata, artifact_id)["revision"]["id"]
    conversation = uuid4()
    transport = httpx.ASGITransport(app=create_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(f"{WORKSPACE}/agent-repairs", json={
            "expectedRevisionId": revision, "commentThreadId": "t1",
            "thread": [{"text": "fix it"}], "conversationId": str(conversation),
        })

    assert response.status_code == 200, response.text
    assert artifact_id == ARTIFACT_ID
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        rows = raw.exec(select(QueuedAgentRepair).where(QueuedAgentRepair.conversation_id == conversation)).all()
    assert [row.slug for row in rows] == [seeded_artifact.name]
