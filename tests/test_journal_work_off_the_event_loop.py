"""Turn-end capture and the local comment routes run off the event loop.

Both wait on an artifact's lock, which the agent's code can hold, and parse
journal files of up to tens of megabytes. On the event loop that wait stalls
every request and stream on the server, every other person's included.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import cowork.handlers.responses as responses_mod
import cowork.services.task_objects as task_objects
from cowork.handlers.responses import ResponsesHandler, _RemoteArtifactsDir
from cowork.services import local_artifact_comments

from test_harness_scratchpad_teardown import _completes, _drain, turns  # noqa: F401 (turns: a fixture)
from test_local_draft_comments import _threads, client, deployment, draft  # noqa: F401 (fixtures)
from test_responses_remote_backend import _no_workspace_or_memory_units, _RecBuffer, _remote_handler
from tests._event_loop import on_the_event_loop
from tests._live_apps import PARTITION_A, SUB_A, A, person_scope


def _record_where(monkeypatch, seen: list[bool], owner, name: str) -> None:
    """Record, on each call to ``owner.name``, whether it ran on the loop."""
    real = getattr(owner, name)

    def recorded(*args, **kwargs):
        seen.append(on_the_event_loop())
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, recorded)


async def test_the_in_process_turn_end_capture_runs_off_the_loop(turns, monkeypatch):
    seen: list[bool] = []

    def changes(**_kwargs):
        seen.append(on_the_event_loop())
        return task_objects.ArtifactChanges(created=[], touched=set())

    monkeypatch.setattr(task_objects, "turn_artifact_changes", changes)

    await _drain(turns.run(_completes))

    assert seen == [False]


async def test_the_remote_turn_end_capture_runs_off_the_loop(monkeypatch):
    seen: list[bool] = []

    def changes(**_kwargs):
        seen.append(on_the_event_loop())
        return task_objects.ArtifactChanges(created=[], touched=set())

    context = _RemoteArtifactsDir(base=Path("/project/artifacts"), project_id="project", project_name="Project", creator=None)
    handler = _remote_handler(monkeypatch, {})
    _no_workspace_or_memory_units(monkeypatch)
    monkeypatch.setattr(ResponsesHandler, "_remote_artifacts_context", staticmethod(lambda *_: context))
    monkeypatch.setattr(task_objects, "snapshot_artifact_state", lambda *_: (set(), {}))
    monkeypatch.setattr(task_objects, "turn_artifact_changes", changes)
    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", AsyncMock(return_value=[]))

    async def replies(**_kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", replies)

    await handler._produce_remote(
        conv_id=uuid4(), input_text="run", original_content="run", model="m", harness_id="anton", buffer=_RecBuffer(),
    )

    assert seen == [False]


def test_the_local_comment_routes_run_off_the_loop(client, draft, monkeypatch):
    _project_id, artifact_id, _folder = draft
    seen: list[bool] = []
    _record_where(monkeypatch, seen, local_artifact_comments, "resolve_artifact_folder")
    _record_where(monkeypatch, seen, local_artifact_comments.LocalArtifactComments, "list_payload")

    listed = client.get(_threads(artifact_id), headers=A)

    assert listed.status_code == 200
    assert seen == [False, False]


async def test_the_local_comment_stream_reads_off_the_loop(draft, monkeypatch):
    _project_id, artifact_id, _folder = draft
    seen: list[bool] = []
    _record_where(monkeypatch, seen, local_artifact_comments, "resolve_artifact_folder")
    _record_where(monkeypatch, seen, local_artifact_comments.LocalArtifactComments, "list_payload")
    _record_where(monkeypatch, seen, local_artifact_comments.LocalArtifactComments, "mtime_ns")

    response = await local_artifact_comments.local_comments_stream(
        artifact_id, scope=person_scope(PARTITION_A, SUB_A)
    )
    events = response.body_iterator
    first = await anext(events)
    await events.aclose()

    assert first == ": keepalive\n\n"
    assert len(seen) == 4 and not any(seen)
