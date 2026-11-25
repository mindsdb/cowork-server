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


def _anton_approval_kind(monkeypatch) -> None:
    """Give the installed anton the approval capability, whether or not it
    has it. On an anton without it, a scheduled run gets no elicitor."""
    from anton.core.interaction import elicit

    monkeypatch.setattr(elicit, "APPROVAL_KIND", "approval", raising=False)


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
    _anton_approval_kind(monkeypatch)
    assert build_elicitor("conv-1", interactive=interactive, write_approval=True) is None


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
    _anton_approval_kind(monkeypatch)
    elicitor = build_elicitor("conv-1", interactive=False, write_approval=True)
    assert elicitor is not None
    assert elicitor.supported_kinds == ("approval",)
    assert "choice" not in elicitor.supported_kinds
    assert elicitor.timeout_s == 300


def test_a_scheduled_run_without_write_approval_gets_no_elicitor(monkeypatch):
    """With write approval off there is no write gate to remember an
    unanswered card, so an approval card a scheduled run raises, such as a
    browser step, would wait out its timeout with nobody watching. The run
    gets no elicitor, and every card comes back at once. The turn's own
    write_approval decides: this server's import-time settings reading it as
    on doesn't give the run an elicitor."""
    from cowork.harnesses.anton_harness import harness
    from cowork.harnesses.anton_harness.harness import build_elicitor

    _questions(monkeypatch, enabled=True)
    _anton_approval_kind(monkeypatch)
    monkeypatch.setattr(harness.settings, "write_approval", True)
    assert build_elicitor("conv-1", interactive=False, write_approval=False) is None
    assert build_elicitor("conv-1").supported_kinds == ("choice",)


def test_a_scheduled_run_takes_the_approval_kind_anton_names(monkeypatch):
    """The scheduled elicitor lists anton's own approval capability, read by
    name from anton's elicit module: a value anton names differently is the
    value the elicitor lists."""
    from anton.core.interaction import elicit

    from cowork.harnesses.anton_harness.harness import build_elicitor

    _questions(monkeypatch, enabled=True)
    monkeypatch.setattr(elicit, "APPROVAL_KIND", "approval-sentinel", raising=False)
    elicitor = build_elicitor("conv-1", interactive=False, write_approval=True)
    assert elicitor is not None
    assert elicitor.supported_kinds == ("approval-sentinel",)


def test_a_scheduled_run_on_an_anton_without_the_approval_kind_gets_no_elicitor(monkeypatch):
    """An anton that predates the approval capability reads any elicitor as
    a channel: its artifact brief takes an approvals-only elicitor as someone
    who can confirm the brief, so a changed repeat brief would stop at its
    confirmation instead of being written. A scheduled run there gets no
    elicitor even with write approval on, and an interactive turn keeps its
    "choice" elicitor."""
    from anton.core.interaction import elicit

    from cowork.harnesses.anton_harness.harness import build_elicitor

    _questions(monkeypatch, enabled=True)
    monkeypatch.delattr(elicit, "APPROVAL_KIND", raising=False)
    assert build_elicitor("conv-1", interactive=False, write_approval=True) is None
    assert build_elicitor("conv-1", write_approval=True).supported_kinds == ("choice",)


async def test_a_scheduled_run_refuses_a_tools_confirmation_card(monkeypatch):
    """A tool's own confirmation card (publishing credentials, changing a
    scheduled task) is a plain choice question, not an approval card, so a
    scheduled run refuses it at once and publishes no card. It asks through
    anton's elicit() exactly as the tools' confirmation cards do."""
    from anton.core.interaction.elicit import AskOption, AskRequest, elicit

    from cowork.harnesses.anton_harness.harness import build_elicitor

    _questions(monkeypatch, enabled=True)
    _anton_approval_kind(monkeypatch)
    published = []
    scheduled = build_elicitor("conv-1", interactive=False, write_approval=True)
    assert scheduled is not None

    class _Session:
        elicitor = scheduled
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
    """The elicitor follows the turn's own interactive flag and the
    write_approval of the AntonSettings the turn's write gate reads."""
    import inspect

    from cowork.harnesses.anton_harness.harness import AntonHarness

    source = " ".join(inspect.getsource(AntonHarness._build_turn_session).split())
    assert (
        "build_elicitor( str(conversation.id), interactive=interactive, "
        "write_approval=anton_settings.write_approval is True, )"
    ) in source


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
    _anton_approval_kind(monkeypatch)
    elicitor = build_elicitor("conv-1", interactive=False, write_approval=True)
    assert elicitor is not None
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
