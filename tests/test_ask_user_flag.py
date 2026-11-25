"""The kill switch.

Frontend and server versions are independent by design — the cloud frontend
is built from a cowork git branch while the server is a pinned wheel — and an
unknown SSE event is dropped silently on both ends. Withholding the elicitor
un-registers ask_user, so a version-skewed pair degrades to plain-text
questions instead of an agent that appears to hang.

Since ENG-1984 the flag defaults to ON (the renderer card shipped with
cowork #537); COWORK_ASK_USER_ENABLED=false remains as the way back off.
"""

from __future__ import annotations

import pytest


def _questions(monkeypatch, *, enabled: bool) -> None:
    """Set COWORK_ASK_USER_ENABLED as the harness reads it."""
    monkeypatch.setattr(
        "cowork.harnesses.anton_harness.harness.get_app_settings",
        lambda: type("S", (), {"ask_user_enabled": enabled})(),
    )


def _write_approval(monkeypatch, *, on: bool) -> None:
    """Set ANTON_WRITE_APPROVAL as the harness's settings read it."""
    from cowork.harnesses.anton_harness import harness

    monkeypatch.setattr(harness.settings, "write_approval", on)


def test_flag_defaults_to_on(monkeypatch):
    monkeypatch.delenv("COWORK_ASK_USER_ENABLED", raising=False)
    from cowork.common.settings.app_settings import AppSettings

    # _env_file=None on purpose: Settings.model_config points env_file at
    # _env_file_chain() (~/.anton/.env, <COWORK_HOME>/.env, ./.env), and
    # delenv does not neutralise a file. Without this, "defaults to on" would
    # silently weaken to "on on this machine".
    assert AppSettings(_env_file=None).ask_user_enabled is True


def test_flag_reads_the_env_var(monkeypatch):
    # The kill-switch direction: the default is on, so the env var matters
    # only as the way to turn the feature OFF (ENG-1984).
    monkeypatch.setenv("COWORK_ASK_USER_ENABLED", "false")
    from cowork.common.settings.app_settings import AppSettings

    assert AppSettings().ask_user_enabled is False


@pytest.mark.parametrize("interactive", [True, False], ids=["interactive", "scheduled"])
def test_harness_injects_no_elicitor_when_the_flag_is_off(monkeypatch, interactive):
    """Questions off is the kill switch for every turn, a scheduled run's
    approval cards included: the server refuses to start with
    ANTON_WRITE_APPROVAL on and questions off, so no write waits on one."""
    from cowork.harnesses.anton_harness.harness import build_elicitor

    _questions(monkeypatch, enabled=False)
    _write_approval(monkeypatch, on=True)
    assert build_elicitor("conv-1", interactive=interactive) is None


def test_harness_injects_a_choice_elicitor_when_the_flag_is_on(monkeypatch):
    from cowork.harnesses.anton_harness.harness import build_elicitor

    _questions(monkeypatch, enabled=True)
    elicitor = build_elicitor("conv-1")
    assert elicitor is not None
    assert elicitor.supported_kinds == ("choice",)
    assert elicitor.timeout_s == 300


def test_a_scheduled_run_gets_an_elicitor_that_takes_only_approval_cards(monkeypatch):
    """A scheduled run keeps the write-approval card, which a person who opens
    the run can answer. It takes no "choice", so a question or a browser
    hand-over has nobody waiting to answer it and comes back at once instead
    of waiting out its timeout before the agent moves on."""
    from cowork.harnesses.anton_harness.harness import build_elicitor

    _questions(monkeypatch, enabled=True)
    _write_approval(monkeypatch, on=True)
    elicitor = build_elicitor("conv-1", interactive=False)
    assert elicitor is not None
    assert elicitor.supported_kinds == ("approval",)
    assert "choice" not in elicitor.supported_kinds
    assert elicitor.timeout_s == 300


def test_a_scheduled_run_without_write_approval_gets_no_elicitor(monkeypatch):
    """With ANTON_WRITE_APPROVAL off there is no write gate to remember an
    unanswered card, so an approval card a scheduled run raises, such as a
    live app's launch or a browser step, would wait out its timeout with
    nobody watching. The run gets no elicitor, and every card comes back at
    once."""
    from cowork.harnesses.anton_harness.harness import build_elicitor

    _questions(monkeypatch, enabled=True)
    _write_approval(monkeypatch, on=False)
    assert build_elicitor("conv-1", interactive=False) is None
    assert build_elicitor("conv-1").supported_kinds == ("choice",)


def test_the_approval_kind_is_antons():
    """The literal stands in for anton's constant, which the pinned anton may
    predate. Where the installed anton has the constant, the two match."""
    from anton.core.interaction import elicit

    from cowork.harnesses.anton_harness.elicitor import APPROVAL_KIND

    assert APPROVAL_KIND == getattr(elicit, "APPROVAL_KIND", "approval")


async def test_a_scheduled_run_refuses_a_tools_confirmation_card(monkeypatch):
    """A tool's own confirmation card (publishing credentials, changing a
    scheduled task) is a plain choice question, not an approval card, so a
    scheduled run refuses it at once and publishes no card. It asks through
    anton's elicit() exactly as the tools' confirmation cards do."""
    from anton.core.interaction.elicit import AskOption, AskRequest, elicit

    from cowork.harnesses.anton_harness.harness import build_elicitor

    _questions(monkeypatch, enabled=True)
    _write_approval(monkeypatch, on=True)
    published = []

    class _Session:
        elicitor = build_elicitor("conv-1", interactive=False)
        emitter = object()
        question_count = 0

        async def emit(self, event):
            published.append(event)

    request = AskRequest(
        prompt="Publish this app?",
        options=(
            AskOption(value="publish", label="Publish", detail="Publish the app"),
            AskOption(value="cancel", label="Cancel", detail="Publish nothing"),
        ),
        allow_custom=False,
    )
    answer = await elicit(_Session(), "publish:1", request, counted=False)
    assert answer.status == "unavailable"
    assert published == []


def test_the_harness_builds_its_elicitor_for_the_turn_it_serves():
    import inspect

    from cowork.harnesses.anton_harness.harness import AntonHarness

    source = inspect.getsource(AntonHarness._build_turn_session)
    assert "build_elicitor(str(conversation.id), interactive=interactive)" in source


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


def test_a_scheduled_runs_session_has_no_ask_user_tool(monkeypatch):
    """The end-to-end consequence for a scheduled run: the session built from
    the harness's elicitor for an unattended turn offers the model no
    question, as a session with no elicitor doesn't."""
    from anton.core.session import ChatSession, ChatSessionConfig
    from cowork.harnesses.anton_harness.harness import build_elicitor

    _questions(monkeypatch, enabled=True)
    _write_approval(monkeypatch, on=True)
    elicitor = build_elicitor("conv-1", interactive=False)
    session = ChatSession(ChatSessionConfig(llm_client=_mock_llm(), elicitor=elicitor))
    session._build_tools()
    names = {t["name"] for t in session.tool_registry.dump()}
    assert session.elicitor is elicitor
    assert "ask_user" not in names
    assert "select_path" in names


def test_a_console_would_defeat_the_kill_switch():
    """anton builds its own CLIElicitor when a console is present, even with
    elicitor=None — so the kill switch depends on cowork never passing one.
    This documents the hazard the next test guards against.
    """
    from rich.console import Console

    from anton.core.session import ChatSession, ChatSessionConfig

    session = ChatSession(
        ChatSessionConfig(llm_client=_mock_llm(), elicitor=None, console=Console(quiet=True))
    )
    session._build_tools()
    names = {t["name"] for t in session.tool_registry.dump()}
    assert "ask_user" in names


def test_harness_never_passes_a_console_to_chat_session_config():
    """A source-level assertion, justified here because the property under
    test is the *absence* of a `console=` keyword argument from the
    harness's ChatSessionConfig(...) call. Absent code has no runtime
    behaviour to exercise — there is no call to make or return value to
    inspect that would prove a kwarg was never passed — so reading the
    source of the one function that builds the config is the only way to
    observe the invariant test_a_console_would_defeat_the_kill_switch just
    showed matters. If this ever fires, either cowork now must build its own
    CLIElicitor with the tool's "choice" support disabled, or a bare
    `console=None` was added (a no-op for anton, but establishes a place for
    a future `console=self._console` to slip in unnoticed).
    """
    import inspect

    from cowork.harnesses.anton_harness.harness import AntonHarness

    source = inspect.getsource(AntonHarness._build_turn_session)
    assert "console=" not in source, (
        "AntonHarness._build_chat_session must not pass console= to "
        "ChatSessionConfig: anton falls back to its own CLIElicitor "
        "(which supports 'choice') whenever elicitor is None and a console "
        "is present, silently reopening the ask_user kill switch."
    )
