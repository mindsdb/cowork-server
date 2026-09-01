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


# ---------------------------------------------------------------------------
# The same rule, for the surface (ENG-1459).
#
# Written because the first version of that change WAS silently untested: with
# the call site in `anton_harness` deleted entirely, all 1,448 tests still
# passed. The helper had unit tests and the resolver had unit tests, and
# nothing connected them to a turn — the identical shape of miss this file was
# created for, one layer down.
#
# `stream_response` is the wrong symbol to key on here: the surface rides on
# `ChatSessionConfig`, which is constructed by turn ORIGINATORS, and one of
# them (the connector probe) lives outside the harnesses dir this file's other
# rule exempts.
# ---------------------------------------------------------------------------


def _is_config_call(node: ast.AST) -> bool:
    """A ``ChatSessionConfig(...)`` construction, however it was imported.

    Matches the attribute form (``mod.ChatSessionConfig(...)``) as well as the
    bare name. A name-only rule would let a future originator that imports the
    module rather than the symbol slip past — and a guard that quietly matches
    nothing is worse than no guard, because it reports green. This also makes
    the finder agree with :func:`_declares_surface`, which already tolerates
    both shapes for ``surface_kwarg`` (#357 review).

    Extracted from the loop so the matching rule is itself testable; the
    previous inline form was a nested ternary that read as a precedence puzzle.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
    return name == "ChatSessionConfig"


def _config_sites() -> list[tuple[Path, ast.Call]]:
    sites: list[tuple[Path, ast.Call]] = []
    for path in sorted(COWORK_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if _is_config_call(node):
                sites.append((path, node))
    return sites


def _declares_surface(call: ast.Call) -> bool:
    if any(kw.arg == "surface" for kw in call.keywords):
        return True
    # `**surface_kwarg(ChatSessionConfig)` — a **-unpack has arg=None.
    for kw in call.keywords:
        if kw.arg is None and isinstance(kw.value, ast.Call):
            func = kw.value.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name == "surface_kwarg":
                return True
    return False


def test_every_turn_originator_declares_its_surface():
    offenders = [
        f"{path.relative_to(COWORK_ROOT.parent)}:{call.lineno}"
        for path, call in _config_sites()
        if not _declares_surface(call)
    ]
    assert not offenders, (
        "These ChatSessionConfig call sites originate a turn without declaring a "
        "surface, so their traces can't be attributed to web vs desktop "
        "(ENG-1459). Pass **surface_kwarg(ChatSessionConfig) — see "
        f"cowork/harnesses/anton_harness/harness.py: {offenders}"
    )


def test_the_surface_guard_actually_sees_the_known_call_sites():
    """Guard against the guard passing vacuously — an AST rule that matches
    nothing reports green forever."""
    found = {str(path.relative_to(COWORK_ROOT.parent)) for path, _ in _config_sites()}
    assert "cowork/harnesses/anton_harness/harness.py" in found
    assert "cowork/services/connectors/probe.py" in found


def test_the_matcher_catches_an_attribute_qualified_construction():
    """Positive control on the MATCHER, not the predicate.

    The guard exists to notice a turn originator that forgot to declare its
    surface. If the matcher only saw bare-name calls, an originator written as
    `session.ChatSessionConfig(...)` would never be examined at all — the guard
    would pass while the site went unchecked, which is the exact
    silently-vacuous failure this file was created to prevent (#357 review).
    """
    bare = ast.parse("ChatSessionConfig(llm_client=x)").body[0].value
    qualified = ast.parse("session.ChatSessionConfig(llm_client=x)").body[0].value
    deep = ast.parse("anton.core.session.ChatSessionConfig(llm_client=x)").body[0].value
    other = ast.parse("SomethingElse(llm_client=x)").body[0].value
    not_a_call = ast.parse("ChatSessionConfig").body[0].value

    assert _is_config_call(bare)
    assert _is_config_call(qualified)
    assert _is_config_call(deep)
    assert not _is_config_call(other)
    assert not _is_config_call(not_a_call)


# ---------------------------------------------------------------------------
# The same rule, for the harness (ENG-1941).
#
# The connector probe got the ENG-1459 `surface` plumbing and never the older
# `harness` field, so every probe trace reached Langfuse with a surface and an
# empty harness — the only anton-originated traffic that did, and therefore
# indistinguishable from direct-API callers in any harness filter. The surface
# rule above was sitting one line away and could not see it, because it keys on
# a different kwarg. Same seam, same guard, second field.
# ---------------------------------------------------------------------------


def _declares_harness(call: ast.Call) -> bool:
    """A literal ``harness=`` keyword — deliberately NOT the ``**helper()``
    shape that :func:`_declares_surface` also accepts.

    The asymmetry is intentional and load-bearing. ``surface`` is **absent**
    from the pinned anton (``uv.lock`` → ``466520b``, v2.26.8.20.3), so passing
    it unconditionally would raise ``TypeError`` on every turn — hence the
    defensive ``**surface_kwarg(ChatSessionConfig)``. ``harness`` *is* declared
    at that same commit, so it can be passed as a plain keyword, and requiring
    the literal keeps this guard exact.

    If a future change ever needs harness passed defensively too, this
    predicate will reject that shape and must be widened the way
    ``_declares_surface`` was — matching on the **inner helper name**, not
    merely on a ``**``-unpack. Matching any unpack would make the guard
    satisfiable by any ``**kwargs`` that happens not to contain a harness,
    which is the vacuous-guard failure this file exists to prevent
    (#386 review).
    """
    return any(kw.arg == "harness" for kw in call.keywords)


def test_every_turn_originator_declares_its_harness():
    offenders = [
        f"{path.relative_to(COWORK_ROOT.parent)}:{call.lineno}"
        for path, call in _config_sites()
        if not _declares_harness(call)
    ]
    assert not offenders, (
        "These ChatSessionConfig call sites originate a turn without declaring a "
        "harness, so their traces carry an empty `harness` and read as direct-API "
        "traffic in Langfuse (ENG-1941). Pass harness=... — see "
        f"cowork/harnesses/anton_harness/harness.py: {offenders}"
    )


def test_the_probe_reports_the_same_harness_as_the_anton_harness():
    """The probe declares its harness as a literal (the connectors service
    does not import the harness package). Pin it to `AntonHarness.id`: if
    either side is renamed alone, probe traces silently become a second
    population under a name nothing filters on — the exact failure ENG-1941
    fixed, reintroduced by a refactor."""
    from cowork.harnesses.anton_harness.harness import AntonHarness
    from cowork.services.connectors.probe import PROBE_HARNESS

    assert PROBE_HARNESS == AntonHarness.id


def test_the_harness_predicate_actually_rejects_a_bare_call():
    """Positive control on `_declares_harness` — a predicate weakened to
    `return True` must fail here, not report green forever."""
    accepted = ast.parse("ChatSessionConfig(harness='anton')").body[0].value
    bare = ast.parse("ChatSessionConfig(llm_client=x)").body[0].value

    assert _declares_harness(accepted)
    assert not _declares_harness(bare), "the predicate accepts a call site with no harness"


def test_the_surface_predicate_actually_rejects_a_bare_call():
    """Positive control on `_declares_surface` itself.

    Without this, weakening the predicate to `return True` disables the rule
    above and every test still passes — the guard would report green forever.
    """
    accepted = ast.parse("ChatSessionConfig(surface='web')").body[0].value
    unpacked = ast.parse("ChatSessionConfig(**surface_kwarg(ChatSessionConfig))").body[0].value
    bare = ast.parse("ChatSessionConfig(llm_client=x)").body[0].value

    assert _declares_surface(accepted)
    assert _declares_surface(unpacked)
    assert not _declares_surface(bare), "the predicate accepts a call site with no surface"


# ---------------------------------------------------------------------------
# The same rule, for anton's verifier latch (ENG-2193).
#
# Fourth field, same seam, and written for the reason recorded above the surface
# rule: delete the call site and every unit test still passes, because helper
# tests and service tests never touch a turn.
#
# The latch is ChatSession state and this server rebuilds that object per
# message, so a dropped call site silently restores the bug it fixed — anton's
# no-verdict counter restarting at zero every message, never reaching two, and
# every message paying a full-history hand-back.
# ---------------------------------------------------------------------------

# The connector probe originates a turn but has no conversation row, so there is
# nowhere to carry a latch to. Exempt by path so the omission is a decision on
# the record rather than a gap the rule cannot see.
_LATCH_EXEMPT = {"cowork/services/connectors/probe.py"}


def _declares_verifier_latch(call: ast.Call) -> bool:
    if any(kw.arg == "initial_verifier_latch" for kw in call.keywords):
        return True
    # `**verifier_latch_kwarg(ChatSessionConfig, ...)` — a **-unpack has arg=None.
    for kw in call.keywords:
        if kw.arg is None and isinstance(kw.value, ast.Call):
            func = kw.value.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name == "verifier_latch_kwarg":
                return True
    return False


def test_every_conversation_turn_carries_the_verifier_latch():
    offenders = [
        f"{path.relative_to(COWORK_ROOT.parent)}:{call.lineno}"
        for path, call in _config_sites()
        if str(path.relative_to(COWORK_ROOT.parent)) not in _LATCH_EXEMPT
        and not _declares_verifier_latch(call)
    ]
    assert not offenders, (
        "These ChatSessionConfig call sites build a per-message session without "
        "carrying anton's verifier latch, so a deterministic verifier failure "
        "re-diagnoses on every message. Pass "
        "**verifier_latch_kwarg(ChatSessionConfig, conversation.verifier_latch) — "
        f"see cowork/harnesses/anton_harness/harness.py: {offenders}"
    )


def test_the_latch_exemption_does_not_swallow_the_harness():
    """The exemption is for the probe alone. If it ever covered the harness, the
    rule above would pass while the only site that matters went unchecked."""
    assert "cowork/harnesses/anton_harness/harness.py" not in _LATCH_EXEMPT
    checked = {
        str(path.relative_to(COWORK_ROOT.parent))
        for path, _ in _config_sites()
        if str(path.relative_to(COWORK_ROOT.parent)) not in _LATCH_EXEMPT
    }
    assert "cowork/harnesses/anton_harness/harness.py" in checked


def test_the_latch_predicate_actually_rejects_a_bare_call():
    """Positive control: a predicate weakened to `return True` must fail here."""
    accepted = ast.parse("ChatSessionConfig(initial_verifier_latch=x)").body[0].value
    unpacked = ast.parse(
        "ChatSessionConfig(**verifier_latch_kwarg(ChatSessionConfig, stored))"
    ).body[0].value
    bare = ast.parse("ChatSessionConfig(llm_client=x)").body[0].value

    assert _declares_verifier_latch(accepted)
    assert _declares_verifier_latch(unpacked)
    assert not _declares_verifier_latch(bare), (
        "the predicate accepts a call site that carries no latch"
    )
