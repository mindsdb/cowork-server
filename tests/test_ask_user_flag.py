"""The kill switch.

Frontend and server versions are independent by design — the cloud frontend
is built from a cowork git branch while the server is a pinned wheel — and an
unknown SSE event is dropped silently on both ends. Withholding the elicitor
un-registers ask_user, so a version-skewed pair degrades to plain-text
questions instead of an agent that appears to hang.
"""

from __future__ import annotations

import pytest


def test_flag_defaults_to_off(monkeypatch):
    monkeypatch.delenv("COWORK_ASK_USER_ENABLED", raising=False)
    from cowork.common.settings.app_settings import AppSettings

    assert AppSettings().ask_user_enabled is False


def test_flag_reads_the_env_var(monkeypatch):
    monkeypatch.setenv("COWORK_ASK_USER_ENABLED", "true")
    from cowork.common.settings.app_settings import AppSettings

    assert AppSettings().ask_user_enabled is True


def test_harness_injects_no_elicitor_when_the_flag_is_off(monkeypatch):
    from cowork.harnesses.anton_harness.harness import build_elicitor

    monkeypatch.setattr(
        "cowork.harnesses.anton_harness.harness.get_app_settings",
        lambda: type("S", (), {"ask_user_enabled": False})(),
    )
    assert build_elicitor("conv-1") is None


def test_harness_injects_a_choice_elicitor_when_the_flag_is_on(monkeypatch):
    from cowork.harnesses.anton_harness.harness import build_elicitor

    monkeypatch.setattr(
        "cowork.harnesses.anton_harness.harness.get_app_settings",
        lambda: type("S", (), {"ask_user_enabled": True})(),
    )
    elicitor = build_elicitor("conv-1")
    assert elicitor is not None
    assert elicitor.supported_kinds == ("choice",)
    assert elicitor.timeout_s == 300


def _mock_llm():
    """Minimal LLM stand-in. ChatSession.__init__ reads
    coding_provider.export_connection_info() and
    planning_provider.native_web_tools() synchronously, so those two must be
    MagicMocks rather than AsyncMocks."""
    from unittest.mock import AsyncMock, MagicMock

    from anton.core.llm.provider import ProviderConnectionInfo

    mock = AsyncMock()
    mock.coding_provider = MagicMock()
    mock.coding_provider.export_connection_info = MagicMock(
        return_value=ProviderConnectionInfo(provider="anthropic", api_key="test")
    )
    mock.coding_model = "claude-sonnet-4-6"
    mock.planning_provider = MagicMock()
    mock.planning_provider.native_web_tools = MagicMock(return_value=set())
    return mock


def test_a_session_without_an_elicitor_has_no_ask_user_tool():
    """The end-to-end consequence of the flag being off."""
    from anton.core.session import ChatSession, ChatSessionConfig

    session = ChatSession(ChatSessionConfig(llm_client=_mock_llm(), elicitor=None))
    session._build_tools()
    names = {t["name"] for t in session.tool_registry.dump()}
    assert "ask_user" not in names
    assert "select_path" in names


def test_a_session_with_a_choice_elicitor_has_the_ask_user_tool():
    """The end-to-end consequence of the flag being on: the other direction
    of the same proof, so the switch is shown to actually switch the
    feature rather than merely returning None from a helper."""
    from anton.core.session import ChatSession, ChatSessionConfig
    from cowork.harnesses.anton_harness.elicitor import CoworkElicitor
    from cowork.streaming.answers import AnswerBroker

    elicitor = CoworkElicitor("conv-1", AnswerBroker(), timeout_s=300)
    session = ChatSession(ChatSessionConfig(llm_client=_mock_llm(), elicitor=elicitor))
    session._build_tools()
    names = {t["name"] for t in session.tool_registry.dump()}
    assert "ask_user" in names
    assert "select_path" in names
