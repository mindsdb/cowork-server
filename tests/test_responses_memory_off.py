"""A question with "memory": "off" reads no learned memory and learns nothing,
while every other question keeps memory.

The handler runs it on a copy of the turn's settings snapshot with memory
off, and the harness turns that into two empty tiers in the turn's scratch
folder and no background passes (test_harness_turn_memory pins that half).
A deployment where memory off still reads stored memory, the desktop, refuses
the question instead of answering it from memory.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import delete
from sqlmodel import Session

import cowork.common.settings.user_settings as user_settings
import cowork.handlers.responses as responses_mod
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_engine
from cowork.handlers.response_routing import DELEGATED_AGENTIC, RouteDecision
from cowork.harnesses.memory.policy import memory_policy
from cowork.models.conversation import Conversation
from cowork.models.project import Project
from cowork.models.setting import Setting
from cowork.principal import personal_partition_id
from cowork.schemas.responses import ResponsesRequest
from cowork.server import create_app
from cowork.streaming import registry

from _fakes import PausedHarness

TOKEN = "memory-off-test-token"
DEPLOYMENT = "5f0c7d2e-8b1a-4c3e-9d4f-2a6b8c0e1f3a"
SUB = "33333333-3333-4333-8333-333333333333"
PERSON = {"Authorization": f"Bearer {TOKEN}", "X-User-Id": SUB, "X-Organization-Id": DEPLOYMENT}


@dataclass(frozen=True)
class TurnMemory:
    """What a turn's harness sees: memory_enabled in the settings it builds
    the turn from, and whether memory_policy then turns memory off."""

    enabled: bool
    off: bool


class _SeesMemory(PausedHarness):
    """Records each turn's TurnMemory, read as AntonHarness._build_chat_session
    reads it, and answers at once."""

    def __init__(self) -> None:
        super().__init__()
        self.release.set()
        self.seen: list[TurnMemory] = []

    def stream_response(self, **kwargs):
        enabled = user_settings.get_user_settings().memory_enabled
        self.seen.append(TurnMemory(enabled=enabled, off=memory_policy(memory_enabled=enabled).off))
        return super().stream_response(**kwargs)


@pytest.fixture(autouse=True)
def _forget_turns():
    yield
    registry.reset()


@pytest.fixture
def trusted(personal_deployment, monkeypatch, tmp_path):
    """create_app's environment on a trusted-identity stack, as
    test_trusted_identity_local sets it up. The person's rows are removed
    afterwards."""
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    monkeypatch.setenv("COWORK_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", "audit")
    get_app_settings.cache_clear()
    yield
    partition = personal_partition_id(DEPLOYMENT, SUB)
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        raw.exec(delete(Conversation).where(Conversation.org_id == partition))
        raw.exec(delete(Project).where(Project.org_id == partition))
        raw.exec(delete(Setting).where(Setting.org_id == partition))
        raw.commit()


async def _ask(monkeypatch, harness: _SeesMemory, *, headers: dict[str, str], **body) -> httpx.Response:
    async def decide(**_kwargs):
        return RouteDecision(route=DELEGATED_AGENTIC, reason="test")

    monkeypatch.setattr(responses_mod, "decide_route", decide)
    monkeypatch.setattr(responses_mod, "get_harness", lambda name: harness)
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await asyncio.wait_for(
            client.post(
                "/api/v1/responses/",
                headers=headers,
                json={"input": "hi", "stream": True, "conversation": str(uuid4()), **body},
            ),
            timeout=10,
        )


async def test_a_question_with_memory_off_runs_its_turn_with_memory_off(trusted, monkeypatch):
    harness = _SeesMemory()

    answered = await _ask(monkeypatch, harness, headers=PERSON, memory="off")

    assert answered.status_code == 200 and "response.completed" in answered.text, answered.text
    assert harness.seen == [TurnMemory(enabled=False, off=True)]


async def test_questions_without_memory_off_keep_memory(trusted, monkeypatch):
    """The copy is the one turn's: the next question, and one that says "on",
    read the deployment's own setting again."""
    harness = _SeesMemory()

    for body in ({"memory": "off"}, {}, {"memory": "on"}):
        answered = await _ask(monkeypatch, harness, headers=PERSON, **body)
        assert answered.status_code == 200, answered.text

    assert harness.seen == [
        TurnMemory(enabled=False, off=True),
        TurnMemory(enabled=True, off=False),
        TurnMemory(enabled=True, off=False),
    ]


async def test_no_question_turns_memory_on_where_the_operator_turned_it_off(trusted, monkeypatch):
    """With MEMORY_ENABLED false, a question without the key and one that
    says "on" run with memory off too: the key only ever turns memory off."""
    monkeypatch.setenv("MEMORY_ENABLED", "false")
    harness = _SeesMemory()

    for body in ({}, {"memory": "on"}, {"memory": "off"}):
        answered = await _ask(monkeypatch, harness, headers=PERSON, **body)
        assert answered.status_code == 200, answered.text

    assert harness.seen == [TurnMemory(enabled=False, off=True)] * 3


async def test_the_desktop_refuses_a_question_with_memory_off(monkeypatch):
    """Without trusted identity headers, memory off still reads the base
    tiers, so the question is refused before anything runs: no conversation
    is saved for it."""
    harness = _SeesMemory()
    conversation = uuid4()

    answered = await _ask(monkeypatch, harness, headers={}, memory="off", conversation=str(conversation))

    assert answered.status_code == 501, answered.text
    assert answered.json()["detail"] == (
        "This deployment can't keep one question from reading memory. "
        'Retry the request without "memory": "off".'
    )
    assert harness.started == 0
    with Session(get_engine(get_app_settings().database.uri)) as raw:
        assert raw.get(Conversation, conversation) is None


async def test_the_desktop_answers_a_question_with_memory_on(monkeypatch):
    """Only "off" is refused there: "on" keeps the desktop's own memory."""
    harness = _SeesMemory()

    answered = await _ask(monkeypatch, harness, headers={}, memory="on")

    assert answered.status_code == 200 and "response.completed" in answered.text, answered.text
    assert harness.seen == [TurnMemory(enabled=True, off=False)]


@pytest.mark.parametrize("value", ["OFF", "none", "", False, None])
def test_memory_takes_only_on_or_off(value):
    assert ResponsesRequest(input="hi").memory == "on"
    with pytest.raises(ValidationError):
        ResponsesRequest(input="hi", memory=value)
