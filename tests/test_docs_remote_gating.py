"""One remote_docs decision per request: it alone decides whether a remote
turn's pod input carries the doc blocks and whether the pod is offered the doc
tools, so the two never disagree."""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

import cowork.handlers.responses as responses_mod
from cowork.docs.prompt import DOC_OUTPUT_NOTE, DOC_VIEW_TAG, doc_context_block
from cowork.handlers.responses import ResponsesHandler, _RemoteArtifactsDir, _turn_has_doc
from cowork.schemas.responses import ResponsesRequest
from cowork.services import product_permissions

DOC_CONTEXT = {"artifactId": "a" * 32, "title": "Plan", "mode": "edit"}
OPEN = f"<{DOC_VIEW_TAG}>"


def _handler(*, interactive=True, schedule_id=None):
    handler = object.__new__(ResponsesHandler)
    handler.interactive = interactive
    handler.schedule_id = schedule_id
    handler.scope = SimpleNamespace(org_mode=True, org_id="org-1", user_id="user-1")
    return handler


@pytest.mark.parametrize("case,expected", [
    ("all", True),
    ("flag_off", False),
    ("no_docs", False),
    ("not_interactive", False),
    ("ask_user_off", False),
    ("scheduled", False),
    ("no_manage", False),
    ("manage_unknown", False),
    ("in_process", False),
])
async def test_remote_docs_needs_every_condition(monkeypatch, case, expected):
    monkeypatch.setenv("COWORK_TURN_BACKEND", "inprocess" if case == "in_process" else "remote")
    monkeypatch.setattr(responses_mod, "get_app_settings", lambda: SimpleNamespace(
        remote_docs=case != "flag_off", ask_user_enabled=case != "ask_user_off"))

    async def may_manage(scope, permission):
        assert permission == "artifact.manage"
        if case == "manage_unknown":
            raise RuntimeError("auth unreachable")
        return case != "no_manage"

    monkeypatch.setattr(product_permissions, "has_product_permission", may_manage)
    handler = _handler(interactive=case != "not_interactive", schedule_id="s1" if case == "scheduled" else None)
    request = ResponsesRequest(input="hi", stream=True, docs=case != "no_docs")
    assert await handler._remote_docs(request) is expected


def test_a_remote_turn_has_the_doc_only_with_remote_docs(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    request = ResponsesRequest(input="hi", stream=True, doc_context=DOC_CONTEXT)
    assert _turn_has_doc(request) is False
    assert _turn_has_doc(request, remote_docs=True) is True
    monkeypatch.setenv("COWORK_TURN_BACKEND", "inprocess")
    assert _turn_has_doc(request) is True


@pytest.mark.parametrize("remote_docs", [False, True])
def test_the_pods_input_carries_the_doc_blocks_only_with_remote_docs(monkeypatch, remote_docs):
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    request = ResponsesRequest(input="Tighten the intro", stream=True, doc_context=DOC_CONTEXT, output="doc")
    texts = [b["text"] for b in _handler()._build_harness_input(request, session=None, remote_docs=remote_docs)]
    assert texts[-1] == "Tighten the intro"
    assert any(t.startswith(OPEN) for t in texts) is remote_docs
    assert (DOC_OUTPUT_NOTE in texts) is remote_docs


def test_the_remote_producer_gets_the_typed_question_apart_from_the_doc_blocks(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")
    called: dict = {}
    handler = _handler()
    handler._produce_remote = lambda **kwargs: called.update(kwargs) or "remote"
    handler._select_producer(
        conv_id=uuid4(),
        harness_input=[{"type": "text", "text": doc_context_block(DOC_CONTEXT)},
                       {"type": "text", "text": "Tighten the intro"}],
        original_content="Tighten the intro", model="anton", disabled=None, harness_name="anton",
        harness_id="anton", buffer=object(), remote_docs=True,
    )
    assert called["input_text"].startswith(OPEN)
    assert called["question_text"] == "Tighten the intro"
    assert called["remote_docs"] is True


async def test_the_revision_label_and_the_turns_question_never_start_with_the_doc_block(monkeypatch, tmp_path):
    from cowork.harnesses.anton_harness.stream_formatter import TurnContext
    from cowork.services import revision_summaries, task_objects
    from test_responses_remote_backend import _no_workspace_or_memory_units, _RecBuffer, _remote_handler

    handler = _remote_handler(monkeypatch, {})
    _no_workspace_or_memory_units(monkeypatch)
    base = tmp_path / "proj" / ".anton" / "artifacts"
    base.mkdir(parents=True)
    monkeypatch.setattr(ResponsesHandler, "_remote_artifacts_context", staticmethod(
        lambda session, conv_id: _RemoteArtifactsDir(base=base, project_id=str(uuid4()), project_name="proj",
                                                     creator=None)))
    monkeypatch.setattr(task_objects, "record_new_artifacts", lambda session, **kwargs: None)

    async def no_publish(*args, **kwargs):
        return []

    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", no_publish)
    questions: list[str] = []
    requests: list[str | None] = []

    def context(**kwargs):
        questions.append(kwargs["question"])
        return TurnContext(**kwargs)

    monkeypatch.setattr(responses_mod, "TurnContext", context)
    real = revision_summaries.summaries_from_messages

    def spy(messages, *, reply=None, request=None):
        requests.append(request)
        return real(messages, reply=reply, request=request)

    monkeypatch.setattr(revision_summaries, "summaries_from_messages", spy)

    async def pod(**kwargs):
        yield "progress", {"phase": "workspace_authorized", "workspace_mode": "persistent"}
        yield "turn_delta", {"text": "Done."}
        yield "turn_completed", {"artifacts_touched": []}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod)
    await handler._produce_remote(
        conv_id=uuid4(), input_text=f"{doc_context_block(DOC_CONTEXT)} Tighten the intro",
        question_text="Tighten the intro", original_content="Tighten the intro", model="anton",
        harness_id="anton", buffer=_RecBuffer(),
    )
    assert questions == ["Tighten the intro"]
    assert requests == ["Tighten the intro"]
