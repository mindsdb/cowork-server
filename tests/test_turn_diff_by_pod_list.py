"""A remote turn's end diff reads the pod's list of the artifacts it touched.

anton names the artifacts a cloud turn's tools touched on ``turn_completed``.
With that list the diff claims and checks only those folders: a sibling turn's
artifact is never this turn's, and the content of every other artifact in the
project is left unread. A pod that sends no list gets today's diff, where
provenance decides what the turn created.
"""
from __future__ import annotations

import errno
import os
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import yaml
from pydantic import BaseModel
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.session import get_engine
from cowork.models.queued_agent_repair import QueuedAgentRepair
from cowork.services import task_objects
from cowork.services.queued_repairs import QueuedRepairs
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


class TurnDiffCases(BaseModel, frozen=True):
    """The tables of turn_diff_by_pod_list.yaml this file reads."""

    hostile_names: list[str]


CASES = TurnDiffCases.model_validate(
    yaml.safe_load((Path(__file__).parent / "fixtures" / "turn_diff_by_pod_list.yaml").read_text(encoding="utf-8"))
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

    def __init__(self, *, monkeypatch: pytest.MonkeyPatch, base: Path):
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

    reads = _ContentReads(monkeypatch=monkeypatch, base=base)
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

    reads = _ContentReads(monkeypatch=monkeypatch, base=base)
    reads.counting = True
    turn_artifact_changes(
        conversation_id=conversation, artifacts_base=base, before=before, before_mtimes=before_mtimes,
        attribute_by_provenance=True,
    )
    reads.counting = False

    assert {f"old-{index}" for index in range(20)} <= reads.folders(base)


# The organization the remote turns below run in.
ORG_SCOPE = TenantScope(org_mode=True, org_id="org-123", user_id="user-456")


async def _remote_turn(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, completed: dict, queued: tuple[str, ...] = (),
    conv_id: UUID | None = None, autopublished: dict | None = None,
) -> list[str]:
    """Run a remote turn whose pod writes `mine` and a sibling's `theirs`,
    both naming this conversation in their provenance, and return the slugs
    it indexed. `queued` are folders recorded as holding a repair queued to
    the conversation. `autopublished` gets the keywords the turn end called
    the publish with."""
    saved: dict = {}
    handler = _remote_handler(monkeypatch, saved)
    monkeypatch.setattr(responses_mod, "scope_from_principal", lambda principal: ORG_SCOPE)
    project_dir = tmp_path / "proj"
    artifacts_base = project_dir / ".anton" / "artifacts"
    artifacts_base.mkdir(parents=True, exist_ok=True)
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
        if autopublished is not None:
            autopublished.update(kwargs, touched=set(touched))
        return set(touched)

    monkeypatch.setattr("cowork.services.artifact_autopublish.autopublish_project_artifacts", fake_autopublish)
    conv_id = conv_id or uuid4()
    _record_queued(conversation_id=conv_id, slugs=queued)

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
    indexed = await _remote_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, completed={"artifacts_touched": ["mine"]})

    assert indexed == ["mine"]


async def test_a_remote_turn_from_an_older_pod_keeps_todays_diff(monkeypatch, tmp_path):
    assert await _remote_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, completed={}) == ["mine", "theirs"]


def _queue_repair(*, folder: Path, conversation_id: UUID) -> str:
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
    repair_id = _queue_repair(folder=base / "old-50", conversation_id=conversation)
    before, before_mtimes = snapshot_artifact_state(base)
    _artifact(base, "made", conversation_id=conversation)

    listings = []
    real_scandir = os.scandir

    def scandir(path=".", *args, **kwargs):
        listings.append(path)
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", scandir)
    reads = _ContentReads(monkeypatch=monkeypatch, base=base)
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


def _record_queued(*, conversation_id: UUID, slugs: Iterable[str], scope: TenantScope = ORG_SCOPE) -> None:
    from cowork.services.queued_repairs import record_queued_repair

    with Session(get_engine(get_app_settings().database.uri)) as raw:
        for slug in slugs:
            record_queued_repair(ScopedSession(raw, scope), conversation_id=conversation_id, slug=slug)


def _queued_slugs(conversation_id: UUID | None = None) -> set[str]:
    query = select(QueuedAgentRepair.slug)
    if conversation_id is not None:
        query = query.where(QueuedAgentRepair.conversation_id == conversation_id)
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        return set(raw.exec(query).all())


async def test_a_remote_turn_reads_its_queued_repairs_and_drops_them(monkeypatch, tmp_path):
    seen: dict = {}
    diff = task_objects.turn_artifact_changes

    def spy_diff(**kwargs):
        seen["repair_slugs"] = kwargs["repair_slugs"]
        return diff(**kwargs)

    monkeypatch.setattr(task_objects, "turn_artifact_changes", spy_diff)
    slug = f"repaired-{uuid4().hex[:8]}"

    await _remote_turn(
        monkeypatch=monkeypatch, tmp_path=tmp_path, completed={"artifacts_touched": ["mine"]}, queued=(slug,),
    )

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

    await _remote_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, completed={}, queued=(slug,))

    assert seen["repair_slugs"] is None
    assert slug in _queued_slugs()


async def _request_repair(*, folder: Path, conversation, revision: str | None = None):
    """POST an agent repair of `folder`, the seeded artifact, for
    `conversation`, against its current revision unless `revision` names
    another."""
    import httpx

    from cowork.server import create_app
    from cowork.services.artifact_identity import ensure_full_id
    from cowork.services.artifact_revisions import current_source
    from test_artifact_resolution_off_the_loop import ARTIFACT_ID, WORKSPACE

    artifact_id, metadata = ensure_full_id(folder)
    assert artifact_id == ARTIFACT_ID
    current = current_source(folder, metadata, artifact_id)["revision"]["id"]
    transport = httpx.ASGITransport(app=create_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(f"{WORKSPACE}/agent-repairs", json={
            "expectedRevisionId": revision or current, "commentThreadId": "t1",
            "thread": [{"text": "fix it"}], "conversationId": str(conversation),
        })


async def test_a_repair_request_records_its_folder_for_the_conversation(seeded_artifact):  # noqa: F811
    conversation = uuid4()

    response = await _request_repair(folder=seeded_artifact, conversation=conversation)

    assert response.status_code == 200, response.text
    assert _queued_slugs(conversation) == {seeded_artifact.name}


async def test_a_refused_repair_request_leaves_no_row(seeded_artifact):  # noqa: F811
    """A request against a stale revision queues nothing, so it keeps no row:
    otherwise any conversation id a caller sends would add one."""
    conversation = uuid4()

    response = await _request_repair(
        folder=seeded_artifact, conversation=conversation, revision="not-the-current-revision",
    )

    assert response.status_code == 409, response.text
    assert _queued_slugs(conversation) == set()


async def test_a_repair_refused_behind_a_queued_one_keeps_the_folders_row(seeded_artifact):  # noqa: F811
    """The folder's row stands for the repair already queued there, which a
    turn of the conversation still has to finish."""
    conversation = uuid4()
    assert (await _request_repair(folder=seeded_artifact, conversation=conversation)).status_code == 200

    response = await _request_repair(folder=seeded_artifact, conversation=conversation)

    assert response.status_code == 422, response.text
    assert _queued_slugs(conversation) == {seeded_artifact.name}


async def test_a_repair_request_that_loses_the_race_to_record_its_folder_keeps_a_row_for_it(monkeypatch):
    """Two requests for one folder both read no row. The first inserts its
    row and commits, so the second's insert collides, and the second replaces
    that row with its own. When the first request's repair is then refused,
    its take-back deletes nothing, so the second request's queued repair
    keeps a row."""
    from cowork.services import queued_repairs

    conversation = uuid4()
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as raw:
        first = queued_repairs.record_queued_repair(
            ScopedSession(raw, ORG_SCOPE), conversation_id=conversation, slug="fixme",
        )
    real_rows = queued_repairs._rows
    reads: list[dict] = []

    def rows_read_before_the_first_commit(session, **kwargs):
        reads.append(kwargs)
        return [] if len(reads) == 1 else real_rows(session, **kwargs)

    monkeypatch.setattr(queued_repairs, "_rows", rows_read_before_the_first_commit)
    with Session(engine) as raw:
        second = queued_repairs.record_queued_repair(
            ScopedSession(raw, ORG_SCOPE), conversation_id=conversation, slug="fixme",
        )
    monkeypatch.setattr(queued_repairs, "_rows", real_rows)

    await queued_repairs.take_back_queued_repair(conversation_id=conversation, row_id=first, scope=ORG_SCOPE)

    assert first is not None
    assert second is None
    assert _queued_slugs(conversation) == {"fixme"}


# ── which rows a turn end drops ──────────────────────────────────────────


def _queued(*, conversation_id: UUID, scope: TenantScope = ORG_SCOPE) -> QueuedRepairs:
    from cowork.services.queued_repairs import read_queued_repairs

    with Session(get_engine(get_app_settings().database.uri)) as raw:
        return read_queued_repairs(ScopedSession(raw, scope), conversation_id=conversation_id)


def test_a_folder_recorded_again_after_the_turn_read_it_keeps_its_row():
    from cowork.services.queued_repairs import forget_queued_repairs

    conversation = uuid4()
    _record_queued(conversation_id=conversation, slugs=["fixme"])
    read_at_turn_start = _queued(conversation_id=conversation)
    _record_queued(conversation_id=conversation, slugs=["fixme"])

    with Session(get_engine(get_app_settings().database.uri)) as raw:
        forget_queued_repairs(
            ScopedSession(raw, ORG_SCOPE), conversation_id=conversation,
            row_ids=read_at_turn_start.ids_for({"fixme"}),
        )

    assert _queued_slugs(conversation) == {"fixme"}


async def test_a_repair_queued_while_the_turn_ends_is_finished_by_the_next_turn(monkeypatch, tmp_path):
    """The person asks for a second repair of a folder once the turn's diff
    has moved the first one out of `queued`, before the turn end drops the
    rows. The second repair's row survives that drop, so the next turn's diff
    checks the folder and finishes the repair that changed nothing."""
    from cowork.services.artifact_revisions import agent_repair_detail
    from cowork.services.queued_repairs import queued_repair_folders

    conv_id = uuid4()
    base = tmp_path / "proj" / ".anton" / "artifacts"
    folder = _artifact(base, "fixme")
    first = _queue_repair(folder=folder, conversation_id=conv_id)
    second: dict[str, str] = {}
    diff = task_objects.turn_artifact_changes

    def diff_then_queue_again(**kwargs):
        changes = diff(**kwargs)
        _record_queued(conversation_id=conv_id, slugs=["fixme"])
        second["id"] = _queue_repair(folder=folder, conversation_id=conv_id)
        return changes

    monkeypatch.setattr(task_objects, "turn_artifact_changes", diff_then_queue_again)
    await _remote_turn(
        monkeypatch=monkeypatch, tmp_path=tmp_path, completed={"artifacts_touched": []}, queued=("fixme",),
        conv_id=conv_id,
    )

    assert agent_repair_detail(folder, first)["repair"]["status"] == "no_change"
    repairs = await queued_repair_folders(conversation_id=conv_id, scope=ORG_SCOPE)
    assert repairs is not None and repairs.slugs == {"fixme"}
    before, before_mtimes = snapshot_artifact_state(base)
    diff(
        conversation_id=conv_id, artifacts_base=base, before=before, before_mtimes=before_mtimes,
        tracked_new=set(), tracked_edits=set(), repair_slugs=repairs.slugs,
    )
    assert agent_repair_detail(folder, second["id"])["repair"]["status"] == "no_change"


async def test_a_turn_end_that_could_not_open_the_artifacts_keeps_the_rows(monkeypatch, tmp_path):
    from cowork.services import artifact_identity

    real_pinned_dir = artifact_identity.pinned_dir

    def unreadable_artifacts(path, *args, **kwargs):
        if Path(path).name == "artifacts":
            raise OSError(errno.EIO, "Input/output error")
        return real_pinned_dir(path, *args, **kwargs)

    monkeypatch.setattr(artifact_identity, "pinned_dir", unreadable_artifacts)
    conv_id = uuid4()

    await _remote_turn(
        monkeypatch=monkeypatch, tmp_path=tmp_path, completed={"artifacts_touched": ["mine"]}, queued=("fixme",),
        conv_id=conv_id,
    )

    assert _queued_slugs(conv_id) == {"fixme"}


async def test_a_channel_turn_end_drops_only_the_rows_of_the_repair_folders_it_read(monkeypatch, tmp_path):
    """The channel producer drops what the browser one does: the rows of the
    repair folders its diff read, and not that of one whose open failed."""
    from cowork.handlers.responses import _RemoteArtifactsDir
    from cowork.services import artifact_identity
    from cowork.turnqueue.remote_turn import _record_turn_artifacts

    base = tmp_path / "proj" / ".anton" / "artifacts"
    _artifact(base, "readable")
    _artifact(base, "flaky")
    before, before_mtimes = snapshot_artifact_state(base)
    conv_id = uuid4()
    _record_queued(conversation_id=conv_id, slugs=["readable", "flaky"])
    real_open_child = artifact_identity.open_pinned_child

    def flaky_open(parent, name, *args, **kwargs):
        if name == "flaky":
            raise OSError(errno.EIO, "Input/output error")
        return real_open_child(parent, name, *args, **kwargs)

    monkeypatch.setattr(artifact_identity, "open_pinned_child", flaky_open)

    await _record_turn_artifacts(
        _RemoteArtifactsDir(base=base, project_id=None, project_name="Project", creator=None),
        conv_id=conv_id, scope=ORG_SCOPE, before_slugs=before, before_mtimes=before_mtimes,
        completed_cleanly=True, touched=set(), repairs=_queued(conversation_id=conv_id),
    )

    assert _queued_slugs(conv_id) == {"flaky"}


def test_the_diff_rules_on_a_repair_folder_only_when_it_could_read_it(tmp_path, monkeypatch):
    """A folder that is gone holds no repair, so its row can go, and so does
    one swapped for a link after its probe, which no reader follows; one
    whose open failed may still hold one, so its row stays."""
    from cowork.services import artifact_identity

    base = tmp_path / ".anton" / "artifacts"
    _artifact(base, "readable")
    _artifact(base, "flaky")
    _artifact(base, "swapped")
    before, before_mtimes = snapshot_artifact_state(base)
    real_open_child = artifact_identity.open_pinned_child

    def flaky_open(parent, name, *args, **kwargs):
        if name == "flaky":
            raise OSError(errno.EIO, "Input/output error")
        if name == "swapped":
            raise OSError(errno.ELOOP, "Too many levels of symbolic links")
        return real_open_child(parent, name, *args, **kwargs)

    monkeypatch.setattr(artifact_identity, "open_pinned_child", flaky_open)

    changes = turn_artifact_changes(
        conversation_id=uuid4(), artifacts_base=base, before=before, before_mtimes=before_mtimes,
        tracked_new=set(), tracked_edits=set(), repair_slugs={"readable", "flaky", "swapped", "gone"},
    )

    assert changes.checked_repairs == {"readable", "swapped", "gone"}


def test_an_artifacts_folder_that_is_gone_rules_on_every_repair_folder(tmp_path):
    """With no artifacts folder, no folder below it holds a repair, so every
    row can go."""
    changes = turn_artifact_changes(
        conversation_id=uuid4(), artifacts_base=tmp_path / ".anton" / "artifacts", before=set(), before_mtimes={},
        tracked_new=set(), tracked_edits=set(), repair_slugs={"fixme"},
    )

    assert changes.checked_repairs == {"fixme"}


def _hold_artifact_lock(*, monkeypatch: pytest.MonkeyPatch, folder: Path) -> None:
    """Hold `folder`'s artifact lock as another writer does (the viewer polls
    a queued repair under it), and shorten the wait for the lock, so a
    capture there gives up after 0.1 s instead of 5 s."""
    from cowork.services import artifact_revisions
    from cowork.services.artifact_lock import artifact_lock

    monkeypatch.setattr(artifact_revisions, "artifact_lock", partial(artifact_lock, timeout=0.1))
    (folder / ".revisions" / ".lock").write_text("another-writer\n", encoding="ascii")


def test_a_repair_folder_whose_queued_repair_the_diff_could_not_finish_is_not_ruled_on(tmp_path, monkeypatch):
    """Another writer holds `held`'s lock, so its capture gives up and its
    repair stays queued: only the next turn end checks that folder, so its
    row has to stay. The repair in `finished` leaves the queued state, so
    that folder's row can go."""
    from cowork.services.artifact_revisions import has_queued_agent_repair

    conversation = uuid4()
    base = tmp_path / ".anton" / "artifacts"
    held = _artifact(base, "held")
    finished = _artifact(base, "finished")
    _queue_repair(folder=held, conversation_id=conversation)
    _queue_repair(folder=finished, conversation_id=conversation)
    before, before_mtimes = snapshot_artifact_state(base)
    _hold_artifact_lock(monkeypatch=monkeypatch, folder=held)

    changes = turn_artifact_changes(
        conversation_id=conversation, artifacts_base=base, before=before, before_mtimes=before_mtimes,
        tracked_new=set(), tracked_edits=set(), repair_slugs={"held", "finished"},
    )

    assert has_queued_agent_repair(held, str(conversation))
    assert not has_queued_agent_repair(finished, str(conversation))
    assert changes.checked_repairs == {"finished"}


@pytest.mark.parametrize("failing", ["listing", "record"])
def test_a_repair_folder_whose_repairs_cannot_be_read_is_not_ruled_on(tmp_path, monkeypatch, failing):
    """The diff can't tell whether the folder still holds a repair queued to
    this conversation, whether listing its repairs fails or reading one of
    them does, so the folder's row stays."""
    from cowork.services.artifact_revisions import _repairs_dir

    conversation = uuid4()
    base = tmp_path / ".anton" / "artifacts"
    folder = _artifact(base, "fixme")
    _queue_repair(folder=folder, conversation_id=conversation)
    before, before_mtimes = snapshot_artifact_state(base)
    repairs = _repairs_dir(folder)
    if failing == "listing":
        for name in ("scandir", "listdir"):
            real = getattr(os, name)

            def unlistable(path=".", *args, _real=real, **kwargs):
                if isinstance(path, (str, os.PathLike)) and os.fspath(path) == os.fspath(repairs):
                    raise OSError(errno.EIO, "Input/output error")
                return _real(path, *args, **kwargs)

            monkeypatch.setattr(os, name, unlistable)
    else:
        real_read_text = Path.read_text

        def unreadable(self, *args, **kwargs):
            if self.parent == repairs:
                raise OSError(errno.EIO, "Input/output error")
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", unreadable)

    changes = turn_artifact_changes(
        conversation_id=conversation, artifacts_base=base, before=before, before_mtimes=before_mtimes,
        tracked_new=set(), tracked_edits=set(), repair_slugs={"fixme"},
    )

    assert changes.checked_repairs == set()


async def test_a_channel_turn_end_keeps_the_row_of_a_repair_it_could_not_finish(monkeypatch, tmp_path):
    from cowork.handlers.responses import _RemoteArtifactsDir
    from cowork.turnqueue.remote_turn import _record_turn_artifacts

    base = tmp_path / "proj" / ".anton" / "artifacts"
    folder = _artifact(base, "fixme")
    conv_id = uuid4()
    _queue_repair(folder=folder, conversation_id=conv_id)
    _record_queued(conversation_id=conv_id, slugs=["fixme"])
    before, before_mtimes = snapshot_artifact_state(base)
    _hold_artifact_lock(monkeypatch=monkeypatch, folder=folder)

    await _record_turn_artifacts(
        _RemoteArtifactsDir(base=base, project_id=None, project_name="Project", creator=None),
        conv_id=conv_id, scope=ORG_SCOPE, before_slugs=before, before_mtimes=before_mtimes,
        completed_cleanly=True, touched=set(), repairs=_queued(conversation_id=conv_id),
    )

    assert _queued_slugs(conv_id) == {"fixme"}


async def test_a_web_turn_end_keeps_the_row_of_a_repair_it_could_not_finish(monkeypatch, tmp_path):
    conv_id = uuid4()
    folder = _artifact(tmp_path / "proj" / ".anton" / "artifacts", "fixme")
    _queue_repair(folder=folder, conversation_id=conv_id)
    _hold_artifact_lock(monkeypatch=monkeypatch, folder=folder)

    await _remote_turn(
        monkeypatch=monkeypatch, tmp_path=tmp_path, completed={"artifacts_touched": []}, queued=("fixme",),
        conv_id=conv_id,
    )

    assert _queued_slugs(conv_id) == {"fixme"}


# ── whose rows a turn reads ──────────────────────────────────────────────

_OTHER_ORG = TenantScope(org_mode=True, org_id="org-other", user_id="someone-else")


def test_another_orgs_row_is_never_read():
    conversation = uuid4()
    _record_queued(conversation_id=conversation, slugs=["planted"], scope=_OTHER_ORG)

    assert _queued(conversation_id=conversation).slugs == set()


def test_a_row_another_org_wrote_does_not_block_this_orgs_row():
    conversation = uuid4()
    _record_queued(conversation_id=conversation, slugs=["shared-name"], scope=_OTHER_ORG)

    _record_queued(conversation_id=conversation, slugs=["shared-name"])

    assert _queued(conversation_id=conversation).slugs == {"shared-name"}
    assert _queued(conversation_id=conversation, scope=_OTHER_ORG).slugs == {"shared-name"}


def test_a_desktop_folder_keeps_one_row_however_often_it_is_recorded():
    conversation = uuid4()
    _record_queued(conversation_id=conversation, slugs=["fixme", "fixme"], scope=LOCAL_SCOPE)

    with Session(get_engine(get_app_settings().database.uri)) as raw:
        rows = raw.exec(select(QueuedAgentRepair).where(QueuedAgentRepair.conversation_id == conversation)).all()
    assert [(row.org_id, row.slug) for row in rows] == [(None, "fixme")]


# ── the pod's names ──────────────────────────────────────────────────────


def test_a_hostile_name_in_the_pods_list_never_leaves_the_artifacts_folder(tmp_path):
    base = tmp_path / "proj" / ".anton" / "artifacts"
    _artifact(base, "old")
    outside = _artifact(tmp_path / "proj" / ".anton", "outside")
    before, before_mtimes = snapshot_artifact_state(base)
    _artifact(base, "good")
    os.symlink(outside, base / "link")
    (base / "linkmeta").mkdir()
    os.symlink(outside / "metadata.json", base / "linkmeta" / "metadata.json")
    names = {"good", str(outside), *CASES.hostile_names}

    changes = turn_artifact_changes(
        conversation_id=uuid4(), artifacts_base=base, before=before, before_mtimes=before_mtimes,
        tracked_new=names, tracked_edits=names, repair_slugs=set(),
    )

    assert changes.created == ["good"]
    assert changes.touched == {"good"}
    assert not (outside / ".revisions").exists()


@pytest.mark.parametrize(
    ("completed", "only_touched"),
    [({"artifacts_touched": ["mine"]}, True), ({}, False)],
    ids=["pod-list", "older-pod"],
)
async def test_with_the_pods_list_the_turn_end_publishes_only_the_turns_folders(
    monkeypatch, tmp_path, completed, only_touched,
):
    autopublished: dict = {}

    await _remote_turn(monkeypatch=monkeypatch, tmp_path=tmp_path, completed=completed, autopublished=autopublished)

    assert autopublished["only_touched"] is only_touched
