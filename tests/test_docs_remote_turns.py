"""A remote turn sent with a doc open, end to end through POST /api/v1/responses/:
the request's one remote_docs decision sets both the pod's input and the tools
it is offered."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import httpx
import pytest

import cowork.handlers.responses as responses_mod
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE
from cowork.db.units import unit_session
from cowork.docs.prompt import DOC_OUTPUT_NOTE, DOC_SKILL_LABEL, DOC_VIEW_TAG
from cowork.handlers.response_routing import DELEGATED_AGENTIC, RouteDecision
from cowork.harnesses.anton_harness import harness as harness_mod
from cowork.server import create_app
from cowork.services import product_permissions
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT_ID
from cowork.streaming import registry

DOC_CONTEXT = {"artifactId": "a" * 32, "title": "Plan", "mode": "edit"}


@pytest.fixture(autouse=True)
def _remote_turns(monkeypatch):
    monkeypatch.setenv("COWORK_TURN_BACKEND", "remote")

    async def delegate(**_kwargs):
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    async def no_vault(_scope):
        return None

    async def no_minted_key(self):
        return None, None

    monkeypatch.setattr(responses_mod, "decide_route", delegate)
    monkeypatch.setattr(responses_mod, "register_vault_secrets", no_vault)
    monkeypatch.setattr(responses_mod.ResponsesHandler, "_router_binding", no_minted_key)
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: harness_mod.AntonHarness())
    yield
    registry.reset()
    get_app_settings.cache_clear()


async def _remote_turn(monkeypatch, *, flag: bool, may_manage: bool, **body) -> dict:
    monkeypatch.setenv("COWORK_REMOTE_DOCS", "true" if flag else "false")
    get_app_settings.cache_clear()

    async def manage(scope, permission):
        # handle() also requires product.execute through the same function.
        return may_manage if permission == "artifact.manage" else True

    monkeypatch.setattr(product_permissions, "has_product_permission", manage)
    sent: dict = {}

    async def pod(**kwargs):
        sent.update(kwargs)
        yield "turn_delta", {"text": "ok"}
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", pod)
    with unit_session(scope=LOCAL_SCOPE) as session:
        service = ConversationService(session)
        conversation_id = service.create_conversation("docs remote", project_id=GENERAL_PROJECT_ID).id
        project_path = Path(service.get_conversation(conversation_id).project.path)
    try:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            answered = await asyncio.wait_for(client.post("/api/v1/responses/", json={
                "input": "Tighten the intro", "stream": True, "conversation": str(conversation_id), **body,
            }), timeout=10)
    finally:
        shutil.rmtree(project_path / "conversations" / str(conversation_id), ignore_errors=True)
    assert answered.status_code == 200, answered.text
    return sent


@pytest.mark.parametrize("flag,may_manage,expected", [(True, True, True), (True, False, False), (False, True, False)])
async def test_the_pods_input_carries_the_open_doc_only_when_the_turn_gets_the_doc_tools(
    monkeypatch, flag, may_manage, expected,
):
    sent = await _remote_turn(monkeypatch, flag=flag, may_manage=may_manage, doc_context=DOC_CONTEXT, output="doc")
    assert (f"<{DOC_VIEW_TAG}>" in sent["input_text"]) is expected
    assert (DOC_OUTPUT_NOTE in sent["input_text"]) is expected
    assert sent["input_text"].endswith("Tighten the intro")


async def test_the_pods_input_keeps_the_doc_blocks_and_the_typed_text_in_separate_paragraphs(monkeypatch):
    """The in-process harness hands these as separate blocks; joined with a
    space, the typed message would read as the end of the Output: Doc note."""
    from cowork.docs.prompt import doc_context_block

    sent = await _remote_turn(monkeypatch, flag=True, may_manage=True, doc_context=DOC_CONTEXT, output="doc")
    assert sent["input_text"] == "\n\n".join([doc_context_block(DOC_CONTEXT), DOC_OUTPUT_NOTE, "Tighten the intro"])


@pytest.mark.parametrize("flag", [True, False])
async def test_without_doc_blocks_the_pods_input_is_the_typed_text_unchanged(monkeypatch, flag):
    sent = await _remote_turn(monkeypatch, flag=flag, may_manage=True)
    assert sent["input_text"] == "Tighten the intro"


@pytest.mark.parametrize("flag,may_manage,expected", [(True, True, True), (True, False, False), (False, True, False)])
async def test_the_pod_is_offered_the_doc_tools_exactly_when_its_input_carries_the_doc(
    monkeypatch, flag, may_manage, expected,
):
    from cowork.docs.tools import DOC_TOOL_NAMES

    sent = await _remote_turn(monkeypatch, flag=flag, may_manage=may_manage, doc_context=DOC_CONTEXT)
    assert (f"<{DOC_VIEW_TAG}>" in sent["input_text"]) is expected
    if expected:
        assert [spec["name"] for spec in sent.get("host_tools") or []] == list(DOC_TOOL_NAMES)
    else:
        assert sent.get("host_tools") is None
    # The skill that unlocks them in the pod travels with them, and only with them.
    assert (DOC_SKILL_LABEL in (sent.get("skills") or {})) is expected


async def test_a_pod_left_without_the_doc_tools_gets_the_typed_text_alone(monkeypatch):
    """No artifacts folder to run the doc tools on: the doc blocks would tell the
    pod to call tools it is not offered, so they are dropped as for any turn without docs."""
    monkeypatch.setattr(responses_mod.ResponsesHandler, "_remote_artifacts_context",
                        staticmethod(lambda session, conv_id: None))
    sent = await _remote_turn(monkeypatch, flag=True, may_manage=True, doc_context=DOC_CONTEXT, output="doc")
    assert sent["input_text"] == "Tighten the intro"
    assert sent.get("host_tools") is None
