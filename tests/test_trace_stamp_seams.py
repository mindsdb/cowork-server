"""Every turn path must stamp the build on its trace (ENG-1279).

This exists because of a mistake, not a hypothesis. Wiring the build stamp
into ``ResponsesHandler`` felt like it covered everything — the API, scheduled
runs and the desktop all go through it. It didn't: ``channels/runtime``
calls ``harness.stream_response`` directly, so every Telegram / Slack /
WhatsApp / Discord turn was silently unattributable to a build. The miss came
from grepping for the wrong symbol (``identity_trace_metadata`` has one
caller; ``stream_response`` has two).

So rather than trusting the next person to grep the right thing, this walks
the AST of every module under ``cowork/`` and fails if a
``…stream_response(...)`` call site doesn't pass ``trace_metadata``. A new
entry point — a webhook, an agent-to-agent hop, another scheduler — trips the
build instead of quietly producing unattributable traces.

Deliberately structural (a call-site rule), not behavioural: a behaviour test
can only cover the paths someone remembered to write a test for, which is the
exact failure being guarded against.
"""

from __future__ import annotations

import ast
from pathlib import Path

COWORK_ROOT = Path(__file__).resolve().parent.parent / "cowork"

# The harness implementations themselves *define* stream_response and forward
# the kwarg they were handed; the rule is about CALLERS that originate a turn.
_EXEMPT_DIRS = {"harnesses"}


def _call_sites() -> list[tuple[Path, ast.Call]]:
    sites: list[tuple[Path, ast.Call]] = []
    for path in sorted(COWORK_ROOT.rglob("*.py")):
        if _EXEMPT_DIRS & set(path.relative_to(COWORK_ROOT).parts[:-1]):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "stream_response":
                sites.append((path, node))
    return sites


def test_every_turn_originator_stamps_the_build():
    offenders = []
    for path, call in _call_sites():
        kwargs = {kw.arg for kw in call.keywords if kw.arg}
        if "trace_metadata" not in kwargs:
            offenders.append(f"{path.relative_to(COWORK_ROOT.parent)}:{call.lineno}")

    assert not offenders, (
        "These stream_response call sites start a turn without a build stamp, so their "
        "traces can't be attributed to a release (ENG-1279). Pass "
        "trace_metadata=build_trace_metadata(...) — see cowork/handlers/responses.py "
        f"and cowork/channels/runtime.py: {offenders}"
    )


def test_the_guard_actually_sees_the_known_call_sites():
    """Guard against the guard passing vacuously.

    An AST rule that matches nothing is worse than no rule — it reports green
    forever. Pin the two seams we know exist, so a refactor that renames or
    relocates them fails here rather than silently disabling the check above.
    """
    found = {str(path.relative_to(COWORK_ROOT.parent)) for path, _ in _call_sites()}
    assert "cowork/handlers/responses.py" in found
    assert "cowork/channels/runtime.py" in found
