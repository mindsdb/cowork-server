"""A Hermes turn that dies on a provider error (e.g. a 404 model-not-found)
must surface as a visible failure, not a silent empty reply.

run_agent's conversation loop catches provider/API errors itself and returns
a result dict with ``failed: True`` instead of raising — unlike Anton, where
the same failure propagates as a real exception out of stream_response and
_run_turn's existing handler turns it into a response.failed error card.
Left unhandled, Hermes's empty final_response flows through
format_hermes_stream's fallback into a plain, wordless response.completed:
the turn looks like it succeeded with nothing to say. HermesHarness.
stream_response now raises HermesTurnFailed when it sees failed=True, so the
turn reaches the same error-card path Anton's failures already use.
"""

import asyncio
from types import SimpleNamespace

import pytest

import cowork.services.task_objects as task_objects
from cowork.harnesses.hermes_harness.harness import HermesHarness, HermesTurnFailed


def _make_conversation():
    return SimpleNamespace(
        id="conv-1",
        project_id="proj-1",
        project=SimpleNamespace(path="/tmp", name="demo"),
        topic="demo turn",
    )


def _patch_common(monkeypatch):
    monkeypatch.setattr(task_objects, "snapshot_artifact_slugs", lambda *_a, **_k: set())
    monkeypatch.setattr(task_objects, "finalize_turn_artifacts", lambda *_a, **_k: [])
    monkeypatch.setattr(task_objects, "snapshot_skill_drafts", lambda *_a, **_k: set())
    monkeypatch.setattr(task_objects, "snapshot_stray_skills", lambda *_a, **_k: set())
    monkeypatch.setattr(task_objects, "finalize_turn_skill_drafts", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "sqlalchemy.orm.object_session", lambda _obj: SimpleNamespace(info={})
    )
    from cowork.services.conversations import ConversationService

    monkeypatch.setattr(ConversationService, "get_ordered_messages", lambda self, _conv_id: [])


def _drain(monkeypatch, run_result):
    _patch_common(monkeypatch)
    monkeypatch.setattr(HermesHarness, "_run", staticmethod(lambda *_a, **_k: run_result))

    async def _collect():
        return [
            event
            async for event in HermesHarness().stream_response(
                conversation=_make_conversation(),
                input=[{"type": "text", "text": "Say exactly: hello world"}],
            )
        ]

    return asyncio.run(_collect())


def test_failed_turn_raises_instead_of_completing_silently(monkeypatch):
    run_result = {
        "failed": True,
        "final_response": None,
        "turn_exit_reason": "Non-retryable client error: model_not_found",
    }
    with pytest.raises(HermesTurnFailed, match="model_not_found"):
        _drain(monkeypatch, run_result)


def test_successful_turn_still_yields_the_result_dict(monkeypatch):
    run_result = {"failed": False, "final_response": "hello world"}
    events = _drain(monkeypatch, run_result)
    assert events[-1] is run_result
