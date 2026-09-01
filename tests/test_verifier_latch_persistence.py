"""Anton's verifier latch has to survive the per-message session rebuild.

The latch is ChatSession state and this server builds a fresh ChatSession for
every message, so anton's no-verdict counter restarted at zero each time. One
message contributes at most one failure and the threshold is two, so the latch
could never engage: a verifier failing the same way every time re-diagnosed on
every message instead of once per conversation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from sqlmodel import Session

from cowork.build_info import verifier_latch_kwarg
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.db.session import get_engine
from cowork.services.conversations import ConversationService
from cowork.services.projects import ProjectService

_LATCH = {
    "model": "mindshub_air",
    "no_verdict_failures": 1,
    "latched": False,
    "reason": "",
    "skips": 0,
}


@dataclass
class _CurrentConfig:
    """Mirrors anton's real declaration, which is keyword-only: a guard that
    only recognised a positional field would report the feature as unsupported
    and silently make it a no-op."""

    initial_verifier_latch: dict | None = field(default=None, kw_only=True)


@dataclass
class _OlderAntonConfig:
    """An anton build predating the field."""

    initial_history: list | None = None


@pytest.fixture
def session():
    engine = get_engine(get_app_settings().database.uri)
    with Session(engine) as s:
        yield s


@pytest.fixture
def svc(session):
    return ConversationService(ScopedSession(session, LOCAL_SCOPE))


@pytest.fixture
def conversation(session, svc):
    project = ProjectService(ScopedSession(session, LOCAL_SCOPE)).create_project(
        "latch-persistence"
    )
    return svc.create_conversation("a conversation", project_id=project.id)


def test_the_stored_latch_is_passed_when_anton_declares_the_field():
    assert verifier_latch_kwarg(_CurrentConfig, _LATCH) == {
        "initial_verifier_latch": _LATCH
    }


def test_an_anton_without_the_field_warns_once_and_is_a_no_op(monkeypatch):
    """Not a TypeError on every turn, which is what an unknown keyword to a
    plain dataclass would cause. Announced once, because the pin is bumped by
    hand and a silent no-op here restores the bug this change fixes.

    The module's logger is stubbed rather than captured: another test in this
    suite calls `setup_logging`, whose `basicConfig(force=True)` makes any
    capture-based assertion order-dependent.
    """
    import cowork.build_info as build_info

    warnings: list[str] = []

    class _Stub:
        @staticmethod
        def warning(message, *args, **kwargs):
            warnings.append(message)

    monkeypatch.setattr(build_info, "logger", _Stub)
    monkeypatch.setattr(build_info, "_latch_unsupported_warned", False)

    assert build_info.verifier_latch_kwarg(_OlderAntonConfig, _LATCH) == {}
    assert build_info.verifier_latch_kwarg(_OlderAntonConfig, _LATCH) == {}

    assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"
    assert "initial_verifier_latch" in warnings[0]


def test_a_non_dataclass_is_a_no_op():
    assert verifier_latch_kwarg(object, _LATCH) == {}


def test_no_stored_latch_still_passes_the_field(session):
    """None is a legitimate value, not an absence: it is what a fresh
    conversation has, and anton reads it as "nothing carried"."""
    assert verifier_latch_kwarg(_CurrentConfig, None) == {
        "initial_verifier_latch": None
    }


def test_a_fresh_conversation_carries_no_latch(conversation):
    assert conversation.verifier_latch is None


def test_the_latch_round_trips_through_the_column(session, svc, conversation):
    svc.update_verifier_latch(conversation.id, _LATCH)
    session.expire_all()

    assert svc.get_conversation(conversation.id).verifier_latch == _LATCH


def test_a_successful_verdict_clears_the_stored_latch(session, svc, conversation):
    """None means clear, not "leave what is there": a stale latch would keep
    verification off for the rest of the conversation."""
    svc.update_verifier_latch(conversation.id, _LATCH)
    session.expire_all()
    svc.update_verifier_latch(conversation.id, None)
    session.expire_all()

    assert svc.get_conversation(conversation.id).verifier_latch is None


def test_an_unowned_conversation_is_not_written(svc):
    """Same owner scoping as every other write on this service."""
    from uuid import uuid4

    svc.update_verifier_latch(uuid4(), _LATCH)  # must not raise
