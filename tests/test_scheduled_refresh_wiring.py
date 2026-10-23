"""The scheduler's schedule id reaches revision capture on every turn path.

Revision behaviour itself is covered in test_artifact_revisions; these pin
the hops between the handler, the harness and turn_artifact_changes, which a
dropped keyword would silently turn back into an ordinary chat edit.
"""
from __future__ import annotations

from uuid import uuid4

import cowork.services.task_objects as task_objects
from cowork.handlers import responses as responses_mod
from cowork.streaming.buffer import FileStreamBuffer
from test_harness_scratchpad_teardown import _completes, _drain, turns  # noqa: F401  (turns is a fixture)
from tests.test_model_wait_ticker import _handler as _inprocess_handler
from tests.test_responses_remote_backend import (
    _FakeBuffer,
    _artifact,
    _artifacts_context_at,
    _remote_handler,
)


def _spy_artifact_changes(monkeypatch) -> dict:
    seen: dict = {}

    def spy(**kwargs):
        seen["schedule_id"] = kwargs.get("schedule_id")
        return task_objects.ArtifactChanges(created=[], touched=set())

    monkeypatch.setattr(task_objects, "turn_artifact_changes", spy)
    return seen


async def test_inprocess_turn_hands_the_schedule_to_the_harness(monkeypatch, tmp_path):
    seen: dict = {}

    async def _formatter(stream, model, event_sink):
        async for _ in stream:
            pass
        return
        yield

    class Harness:
        formatter = staticmethod(_formatter)

        async def stream_response(self, **kwargs):
            seen.update(kwargs)
            return
            yield

    handler = _inprocess_handler(monkeypatch, {}, Harness())
    handler.schedule_id = "schedule-1"

    await handler._run_turn(
        conv_id=uuid4(), harness_input=[], original_content="refresh", model="anton",
        disabled=None, harness_name="anton", harness_id="anton",
        buffer=FileStreamBuffer(tmp_path / "turn.log"),
    )

    assert seen["schedule_id"] == "schedule-1"


async def test_harness_attributes_its_artifact_edits_to_the_schedule(turns, monkeypatch):
    seen = _spy_artifact_changes(monkeypatch)

    await _drain(turns.run(_completes, schedule_id="schedule-1"))

    assert seen == {"schedule_id": "schedule-1"}


async def test_harness_turn_without_a_schedule_is_an_ordinary_edit(turns, monkeypatch):
    seen = _spy_artifact_changes(monkeypatch)

    await _drain(turns.run(_completes))

    assert seen == {"schedule_id": None}


async def test_remote_turn_attributes_its_artifact_edits_to_the_schedule(monkeypatch, tmp_path):
    handler = _remote_handler(monkeypatch, {})
    handler.schedule_id = "schedule-1"
    artifacts_base = tmp_path / "proj" / ".anton" / "artifacts"
    artifacts_base.mkdir(parents=True)
    _artifacts_context_at(monkeypatch, artifacts_base)
    seen = _spy_artifact_changes(monkeypatch)
    monkeypatch.setattr(task_objects, "record_new_artifacts", lambda session, **kwargs: None)

    async def fake_replies(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        _artifact(artifacts_base, "dash")
        yield "turn_failed", {"error": "boom", "code": "anton_error", "message": "failed"}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", fake_replies)

    await handler._produce_remote(
        conv_id=uuid4(), input_text="refresh", original_content="refresh",
        model="anton", harness_id="anton", buffer=_FakeBuffer(),
    )

    assert seen == {"schedule_id": "schedule-1"}
