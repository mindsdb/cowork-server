"""ConversationExecutionIsolation — the core invariant of the coworker
architecture: a CLI coworker's native session (e.g. Claude Code's resume
token) is scoped to (conversation, coworker). Chat history is a shared UI
artifact; native execution sessions never bleed across coworkers or
conversations.

Encodes the scenario:

    Conversation A: claude-code, claude-code, anton, claude-code
    Conversation B: anton, anton
    Conversation C: <a second CLI coworker>, <same>

Assertions:
  - Claude resumes only when Claude produced a prior turn (A: yes; B: n/a).
  - A CLI coworker never resumes on another coworker's turns.
  - A conversation with no turns by this coworker starts fresh.
  - The resume decision reads only THIS coworker's messages — no coworker
    reads another's resume metadata.
"""
from types import SimpleNamespace

from cowork.harnesses.cli_agents.base import BaseCliHarness
from cowork.harnesses.cli_agents.config import CliConfig


def _msg(role: str, harness: str | None = None):
    # The resume decision reads only .role and .harness (see
    # BaseCliHarness.should_resume), so a lightweight stand-in is a
    # faithful test double for a Message row.
    return SimpleNamespace(role=role, harness=harness)


class _FakeCliA(BaseCliHarness):
    id = "cli-a"
    label = "CLI A"
    config = CliConfig(executable="cli-a", supports_resume=True)


class _FakeCliB(BaseCliHarness):
    id = "cli-b"
    label = "CLI B"
    config = CliConfig(executable="cli-b", supports_resume=True)


class _NoResumeCli(BaseCliHarness):
    id = "cli-noresume"
    label = "No-Resume CLI"
    config = CliConfig(executable="cli-nr", supports_resume=False)


def test_resumes_only_own_prior_turns():
    cli_a = _FakeCliA()
    # Conversation A: mixed coworkers, cli-a produced turns 1 and 4.
    conv_a = [
        _msg("user"), _msg("assistant", "cli-a"),
        _msg("user"), _msg("assistant", "anton"),
        _msg("user"), _msg("assistant", "cli-a"),
    ]
    assert cli_a.should_resume(conv_a) is True


def test_never_resumes_on_another_coworkers_turns():
    cli_a = _FakeCliA()
    # Only Anton and cli-b turns present — cli-a has no session here.
    conv = [
        _msg("user"), _msg("assistant", "anton"),
        _msg("user"), _msg("assistant", "cli-b"),
    ]
    assert cli_a.should_resume(conv) is False


def test_fresh_conversation_starts_fresh():
    assert _FakeCliA().should_resume([]) is False
    # A conversation with only a pending user message (no assistant yet).
    assert _FakeCliA().should_resume([_msg("user")]) is False


def test_isolation_between_coworkers_in_same_conversation():
    # Conversation A from the scenario, evaluated by each coworker.
    conv_a = [
        _msg("user"), _msg("assistant", "claude-code"),
        _msg("user"), _msg("assistant", "claude-code"),
        _msg("user"), _msg("assistant", "anton"),
        _msg("user"), _msg("assistant", "claude-code"),
    ]

    class _ClaudeLike(BaseCliHarness):
        id = "claude-code"
        label = "Claude Code"
        config = CliConfig(executable="claude", supports_resume=True)

    # Claude resumes (it has turns). A different CLI in the same
    # conversation does NOT — it never produced a turn here, so it never
    # reads/claims Claude's resume metadata.
    assert _ClaudeLike().should_resume(conv_a) is True
    assert _FakeCliB().should_resume(conv_a) is False


def test_conversation_b_independent():
    # Conversation B: pure Anton — no CLI coworker resumes anything.
    conv_b = [_msg("user"), _msg("assistant", "anton"), _msg("user"), _msg("assistant", "anton")]
    assert _FakeCliA().should_resume(conv_b) is False
    assert _FakeCliB().should_resume(conv_b) is False


def test_supports_resume_false_never_resumes():
    # A coworker whose CLI can't resume always starts fresh, even with
    # its own prior turns present.
    conv = [_msg("user"), _msg("assistant", "cli-noresume")]
    assert _NoResumeCli().should_resume(conv) is False


def test_antigravity_never_resumes_real_registration():
    # The real AntigravityHarness: headless resume hangs/times out on
    # this CLI (confirmed by direct testing, not a harness assumption),
    # so supports_resume=False is load-bearing — verify the actual
    # registered class, not just a fake, holds the invariant even with
    # its own prior turns present.
    from cowork.harnesses.cli_agents.antigravity import AntigravityHarness

    conv = [_msg("user"), _msg("assistant", "antigravity"), _msg("user"), _msg("assistant", "antigravity")]
    assert AntigravityHarness().should_resume(conv) is False
