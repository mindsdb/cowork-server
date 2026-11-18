"""A remote turn's relay loop runs the doc tool calls its pod forwards: each as
a task of its own, answered on the pod's stdin queue; the turn end waits for
them a bounded time and claims the docs they wrote."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from uuid import uuid4

import fakeredis.aioredis
import pytest

import cowork.handlers.responses as responses_mod
from cowork.docs import remote as remote_mod
from cowork.docs import tools as doc_tools
from cowork.docs.store import find_doc_folder
from cowork.docs.tools import DOC_TOOL_NAMES, DocToolResult
from cowork.handlers.responses import ResponsesHandler, _RemoteArtifactsDir
from cowork.services import task_objects
from cowork.services.artifact_ownership import provenance_origin
from cowork.turnqueue import host_tools as host_tools_mod
from cowork.turnqueue.redis_client import answer_queue_key
from test_responses_remote_backend import _no_workspace_or_memory_units, _RecBuffer, _remote_handler

CALL = "htool:" + "ef" * 16
PERSISTENT = ("progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"})


@pytest.fixture
def redis(monkeypatch):
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(host_tools_mod, "get_redis", lambda: client)
    return client


def _doc_turn(monkeypatch, tmp_path, saved=None):
    handler = _remote_handler(monkeypatch, saved if saved is not None else {})
    _no_workspace_or_memory_units(monkeypatch)
    base = tmp_path / "proj" / ".anton" / "artifacts"
    base.mkdir(parents=True)
    monkeypatch.setattr(ResponsesHandler, "_remote_artifacts_context", staticmethod(
        lambda session, conv_id: _RemoteArtifactsDir(base=base, project_id=str(uuid4()), project_name="proj",
                                                     creator="user-456")))
    monkeypatch.setattr(task_objects, "record_new_artifacts", lambda session, **kwargs: None)

    async def no_publish(*args, **kwargs):
        return []

    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", no_publish)
    return handler, base


def _call(tool, **tc_input):
    return "turn_step", {"step": "host_tool_call", "call_id": CALL, "tool": tool, "input": tc_input}


async def _answered(redis, corr, timeout=5.0) -> list[dict]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        entries = [json.loads(e) for e in await redis.lrange(answer_queue_key(corr), 0, -1)]
        if any(e["kind"] == "host_tool_result" and e["call_id"] == CALL and e["last"] for e in entries):
            return entries
        assert loop.time() < deadline, "the call was never answered"
        await asyncio.sleep(0.01)


async def _produce(handler, conv_id, *, remote_docs=True, buffer=None):
    await asyncio.wait_for(handler._produce_remote(
        conv_id=conv_id, input_text="Write a plan", original_content="Write a plan", model="anton",
        harness_id="anton", buffer=buffer or _RecBuffer(), remote_docs=remote_docs,
    ), timeout=10)


async def test_a_pods_create_doc_call_writes_the_doc_answers_it_and_the_turn_end_claims_it(
    monkeypatch, tmp_path, redis,
):
    handler, base = _doc_turn(monkeypatch, tmp_path)
    conv_id = uuid4()
    sent: dict = {}
    claimed: dict = {}
    real_changes = task_objects.turn_artifact_changes

    def spy(**kwargs):
        claimed.update(kwargs)
        return real_changes(**kwargs)

    monkeypatch.setattr(task_objects, "turn_artifact_changes", spy)

    async def pod(**kwargs):
        sent.update(kwargs)
        yield PERSISTENT
        yield _call("create_doc", title="Plan", markdown="Lead.")
        sent["entries"] = await _answered(redis, kwargs["correlation_id"])
        yield "turn_delta", {"text": "Made it."}
        yield "turn_completed", {"artifacts_touched": []}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod)
    await _produce(handler, conv_id)

    assert [spec["name"] for spec in sent.get("host_tools") or []] == list(DOC_TOOL_NAMES)
    progress = [e for e in sent["entries"] if e["kind"] == "host_tool_progress"]
    (result,) = [e for e in sent["entries"] if e["kind"] == "host_tool_result"]
    assert [p["text"] for p in progress] == ["Creating doc “Plan”"]
    assert result["ok"] is True and "Created doc “Plan”" in result["text"]
    folder = find_doc_folder(base, "Plan")
    assert provenance_origin(folder) == conv_id
    assert folder.name in claimed["tracked_new"] and folder.name in claimed["tracked_edits"]


@pytest.mark.parametrize("remote_docs", [False, True])
async def test_the_pod_is_offered_the_doc_tools_only_with_remote_docs(monkeypatch, tmp_path, redis, remote_docs):
    handler, _base = _doc_turn(monkeypatch, tmp_path)
    sent: dict = {}

    async def pod(**kwargs):
        sent.update(kwargs)
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod)
    await _produce(handler, uuid4(), remote_docs=remote_docs)
    assert (sent.get("host_tools") is not None) is remote_docs


async def test_a_failed_artifacts_read_is_retried_so_the_pod_still_gets_the_doc_tools(monkeypatch, tmp_path, redis):
    handler, base = _doc_turn(monkeypatch, tmp_path)
    reads: list[int] = []

    def flaky(session, conv_id):
        reads.append(1)
        if len(reads) == 1:
            return None  # the seed's read failed
        return _RemoteArtifactsDir(base=base, project_id=str(uuid4()), project_name="proj", creator="user-456")

    monkeypatch.setattr(ResponsesHandler, "_remote_artifacts_context", staticmethod(flaky))
    sent: dict = {}

    async def pod(**kwargs):
        sent.update(kwargs)
        yield PERSISTENT
        yield _call("create_doc", title="Plan", markdown="Lead.")
        await _answered(redis, kwargs["correlation_id"])
        yield "turn_completed", {"artifacts_touched": []}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod)
    await _produce(handler, uuid4())
    assert len(reads) == 2
    assert [spec["name"] for spec in sent.get("host_tools") or []] == list(DOC_TOOL_NAMES)
    assert find_doc_folder(base, "Plan").is_dir()


async def test_no_artifacts_folder_means_no_doc_tools_and_a_warning(monkeypatch, tmp_path, redis, caplog):
    handler, _base = _doc_turn(monkeypatch, tmp_path)
    monkeypatch.setattr(ResponsesHandler, "_remote_artifacts_context", staticmethod(lambda session, conv_id: None))
    sent: dict = {}

    async def pod(**kwargs):
        sent.update(kwargs)
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod)
    with caplog.at_level(logging.WARNING, logger=responses_mod.logger.name):
        await _produce(handler, uuid4())
    assert sent.get("host_tools") is None
    assert any("found no artifacts folder" in r.getMessage() for r in caplog.records)


async def test_a_call_without_remote_docs_is_ignored(monkeypatch, tmp_path):
    handler, base = _doc_turn(monkeypatch, tmp_path)
    monkeypatch.setattr(host_tools_mod, "get_redis", lambda: pytest.fail("a turn without doc tools used Redis"))

    async def pod(**kwargs):
        yield PERSISTENT
        yield _call("create_doc", title="Plan", markdown="Lead.")
        yield "turn_completed", {"artifacts_touched": []}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod)
    await _produce(handler, uuid4(), remote_docs=False)
    assert list(base.iterdir()) == []


async def test_a_running_call_does_not_hold_up_the_pods_other_replies(monkeypatch, tmp_path, redis):
    saved: dict = {}
    handler, _base = _doc_turn(monkeypatch, tmp_path, saved)
    release = asyncio.Event()

    async def blocked(ctx, name, tc_input, emit_step, **kwargs):
        await release.wait()
        return DocToolResult("late but fine")

    monkeypatch.setattr(remote_mod, "run_doc_tool", blocked)

    async def pod(**kwargs):
        yield PERSISTENT
        yield _call("read_doc", doc="Plan")
        # Asked for while the call still waits: the loop did not await it.
        yield "turn_delta", {"text": "Still here."}
        release.set()
        await _answered(redis, kwargs["correlation_id"])
        yield "turn_completed", {"artifacts_touched": []}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod)
    await _produce(handler, uuid4())
    assert saved["assistant"] == "Still here."


async def test_the_turn_end_waits_for_a_late_call_names_its_doc_and_sends_nothing_after(
    monkeypatch, tmp_path, redis, caplog,
):
    handler, base = _doc_turn(monkeypatch, tmp_path)
    conv_id = uuid4()
    monkeypatch.setattr(host_tools_mod, "TURN_END_WAIT_S", 0.2)
    release = threading.Event()

    def held_card(*_args, **_kwargs):
        release.wait(5)
        return None

    monkeypatch.setattr(doc_tools, "card_for_folder", held_card)
    corr: dict = {}

    async def pod(**kwargs):
        corr["id"] = kwargs["correlation_id"]
        yield PERSISTENT
        yield _call("create_doc", title="Plan", markdown="Lead.")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while not any(provenance_origin(f) == conv_id for f in base.iterdir()):
            assert loop.time() < deadline, "the doc was never written"
            await asyncio.sleep(0.01)
        yield "turn_completed", {"artifacts_touched": []}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod)
    try:
        with caplog.at_level(logging.WARNING, logger=responses_mod.logger.name):
            await _produce(handler, conv_id)
    finally:
        release.set()
    slug = find_doc_folder(base, "Plan").name
    assert any(getattr(r, "artifact_slug", None) == slug and "still running" in r.getMessage()
               for r in caplog.records)
    await asyncio.sleep(0.2)
    kinds = [json.loads(e)["kind"] for e in await redis.lrange(answer_queue_key(corr["id"]), 0, -1)]
    assert "host_tool_result" not in kinds
