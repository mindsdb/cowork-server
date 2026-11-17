"""A cloud turn's job says whether this server applies the memory changes its
pod sends (memory_writable).

The server drops every memory save and removal a cloud turn sends while
memory is disabled, its mode is "off", or the conversation is a
model-comparison side (ResponsesHandler._change_turn_memory). The job carries
the same answer, read by the same check (turn_memory_writable), so the pod's
memory tools say memory is read-only instead of reporting a change nobody
applies. tests/turnqueue/test_producer.py pins that the value reaches the
job's params.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

import cowork.handlers.responses as responses_mod
import cowork.services.shared_resources as shared_resources
import cowork.turnqueue.remote_turn as remote_turn_mod
from cowork.common.settings.user_settings import UserSettings
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.handlers.responses import ResponsesHandler
from cowork.services.projects import COMPARISON_SANDBOX_PREFIX
from cowork.turnqueue.remote_turn import remote_turn_events
from tests.test_remote_turn import _drain, _fake_handler
from tests.test_responses_remote_backend import _FakeBuffer, _no_workspace_or_memory_units, _remote_handler

# The real read, kept before the shared test helpers replace it.
_REMOTE_MEMORY_WRITABLE = ResponsesHandler._remote_memory_writable

# (memory_enabled, memory_mode, project name, whether the server applies the
# pod's memory changes)
_CASES = {
    "normal": (True, "autopilot", "reports", True),
    "memory-disabled": (False, "autopilot", "reports", False),
    "mode-off": (True, "off", "reports", False),
    "comparison-side": (True, "autopilot", f"{COMPARISON_SANDBOX_PREFIX}{'a' * 32}", False),
}


@pytest.fixture(params=list(_CASES))
def memory_case(request, monkeypatch):
    """The turn's settings and its conversation's project for one case, as
    both the job's read and the save read them: the settings through
    get_user_settings, the project through the conversation."""
    enabled, mode, name, writable = _CASES[request.param]
    project = SimpleNamespace(id=uuid4(), path="/projects/one", name=name)
    settings = UserSettings(memory_enabled=enabled, memory_mode=mode)
    monkeypatch.setattr(responses_mod, "get_user_settings", lambda scope=None: settings.model_copy())
    return SimpleNamespace(project=project, writable=writable)


def _conversation_service(base: type, project) -> type:
    """``base`` with every conversation read answering a conversation in
    ``project``."""

    class InProject(base):
        def __init__(self, session):
            pass

        def get_conversation(self, conv_id):
            return SimpleNamespace(project=project, project_id=project.id)

    return InProject


def test_the_job_and_the_save_read_one_check(monkeypatch, memory_case):
    """The value the job carries is the save's own decision: memory_writable
    is False exactly when _change_turn_memory drops what the pod sends."""
    project = memory_case.project
    applied: list = []

    class FakeSession:
        scope = LOCAL_SCOPE

        def get(self, model, ident):
            return project

        def refresh(self, row):
            pass

    class FakeAccess:
        def __init__(self, session, principal):
            pass

        @contextmanager
        def coordination_lock(self, kind, key):
            yield

    def fake_apply(scope, path, entries, **kwargs):
        applied.append(entries)
        return len(entries)

    monkeypatch.setattr(responses_mod, "ConversationService", _conversation_service(object, project))
    monkeypatch.setattr(shared_resources, "SharedResourceAccess", FakeAccess)
    session, conv_id, entries = FakeSession(), uuid4(), [{"text": "Use metric units"}]

    job_says = ResponsesHandler._remote_memory_writable(session, conv_id)
    saved = ResponsesHandler._change_turn_memory(session, conv_id, entries, object(), apply=fake_apply)

    assert job_says is memory_case.writable
    assert (saved is not None) is job_says
    assert applied == ([entries] if job_says else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("interactive", [True, False], ids=["web", "scheduled"])
async def test_a_browser_or_scheduled_turns_job_says_whether_memory_is_writable(
    monkeypatch, memory_case, interactive,
):
    """Web and scheduled turns both build their job in _produce_remote; a
    scheduled run is a handler that takes no answers."""
    handler = _remote_handler(monkeypatch, {})
    handler.interactive = interactive
    _no_workspace_or_memory_units(monkeypatch)
    monkeypatch.setattr(ResponsesHandler, "_remote_artifacts_context", staticmethod(lambda session, conv_id: None))
    monkeypatch.setattr(ResponsesHandler, "_remote_memory_writable", staticmethod(_REMOTE_MEMORY_WRITABLE))
    monkeypatch.setattr(
        responses_mod, "ConversationService",
        _conversation_service(responses_mod.ConversationService, memory_case.project),
    )
    sent: list = []

    async def fake_replies(**kwargs):
        sent.append(kwargs.get("memory_writable"))
        yield "turn_completed", {}

    monkeypatch.setattr(responses_mod, "stream_remote_replies", fake_replies)

    await handler._produce_remote(
        conv_id=uuid4(), input_text="remember that", original_content="remember that",
        model="anton", harness_id="anton", buffer=_FakeBuffer(),
    )

    assert sent == [memory_case.writable]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope",
    [LOCAL_SCOPE, TenantScope(org_mode=True, org_id="org-123")],
    ids=["local", "organization"],
)
async def test_a_channel_turns_job_says_whether_memory_is_writable(monkeypatch, memory_case, scope):
    """Outside org tenancy a channel turn's memory changes pass the same check
    as a browser turn's. An organization's channel turn has no member, and
    none of its memory changes is applied, so its job always says read-only."""
    _fake_handler(monkeypatch)
    monkeypatch.setattr(ResponsesHandler, "_remote_memory_writable", staticmethod(_REMOTE_MEMORY_WRITABLE))
    monkeypatch.setattr(responses_mod, "ConversationService", _conversation_service(object, memory_case.project))
    sent: list = []

    async def fake_replies(**kwargs):
        sent.append(kwargs.get("memory_writable"))
        yield "turn_completed", {}

    monkeypatch.setattr(remote_turn_mod, "stream_remote_replies", fake_replies)

    _events, failure = await _drain(remote_turn_events(
        scope=scope, conv_id=uuid4(), org_id=scope.org_id, user_id=None,
        input_text="remember that", model="anton", turn_rows=[],
    ))

    assert failure is None
    assert sent == [memory_case.writable and not scope.org_mode]


def test_a_failed_read_sends_memory_writable(monkeypatch, caplog):
    """A read that fails tells the pod memory is writable, and the failure is
    logged. The save checks each change again when it arrives
    (_change_turn_memory), so a pod told read-only would refuse a save that
    check applies."""

    class Unreadable:
        def __init__(self, session):
            pass

        def get_conversation(self, conv_id):
            raise RuntimeError("database down")

    monkeypatch.setattr(responses_mod, "ConversationService", Unreadable)
    # A migration test run earlier in the session (Alembic's fileConfig)
    # leaves the logger disabled, and caplog would then capture nothing.
    monkeypatch.setattr(responses_mod.logger, "disabled", False)

    assert ResponsesHandler._remote_memory_writable(SimpleNamespace(scope=LOCAL_SCOPE), uuid4()) is True
    assert [
        (r.levelname, r.getMessage()) for r in caplog.records if r.name == responses_mod.logger.name
    ] == [
        ("ERROR", "[responses] failed to read whether memory is writable; the pod gets it writable"),
    ]
