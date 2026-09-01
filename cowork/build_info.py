"""Which build is running, for trace attribution (ENG-1279).

Delivery in this stack is four hops with different latencies per channel (a
git-channel desktop picks up anton in minutes, a PyPI-channel desktop only
gets it when a cowork-server wheel ships, a hosted install waits for a
snapshot bake), so "which build produced this trace?" cannot be answered from
release dates. These values ride on every turn's Langfuse trace metadata, and
the router lifts ``anton_version`` onto the trace's native ``version`` field —
the only form the Langfuse metrics API can group by.

Everything here is a process-lifetime constant: a version change means the
package on disk was replaced, and the sidecar is restarted as part of that, so
the cached values can't go stale within one process.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution, version

logger = logging.getLogger(__name__)

# Trace-metadata keys. ``anton_version`` is the reserved key the MindsHub
# router lifts onto the trace's groupable ``version`` field; the other two stay
# metadata-only (filterable, not groupable — the Langfuse SDK exposes per-trace
# ``version`` but not per-trace ``release``).
KEY_ANTON_VERSION = "anton_version"
KEY_SERVER_VERSION = "cowork_server_version"
KEY_INSTALL_CHANNEL = "install_channel"

# Distribution names, not import names: anton was renamed anton -> anton-agent.
_SERVER_DIST = "cowork-server"
_ANTON_DIST = "anton-agent"

# The closed set of channel values. A release measurement groups/filters on
# these, so an override outside this set is ignored (with a warning) rather
# than minted into a new population.
VALID_CHANNELS = frozenset({"hosted", "git", "pypi", "local", "unknown"})


@lru_cache(maxsize=None)
def _dist_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - metadata machinery should never break a turn
        logger.warning("build_info: could not read version for %s", name, exc_info=True)
        return None


def surface() -> str | None:
    """Which surface this deployment serves: ``web`` / ``desktop`` / other.

    ENG-1459. Web and desktop are one codebase and one server, so nothing on a
    trace distinguished them — and "are web users behaving differently?" is the
    question the SaaS launch needs answered.

    Resolution mirrors :func:`install_channel`: an explicit ``COWORK_SURFACE``
    wins, because a deployer declaring what it serves knows more than any
    inference from inside the process. Failing that, **org tenancy is the
    signal** — it is set only on the multi-tenant cloud deployment, so it means
    web; anything else is a desktop sidecar.

    .. warning::

       **``web`` is currently unreachable through the in-process turn path, so
       this only delivers ``desktop`` in practice.** The web deployment sets
       ``COWORK_TURN_BACKEND=remote`` (``deployment/cowork-server/values.yaml``),
       and :meth:`AntonHarness.stream_response` refuses to run in-process when
       ``tenancy_mode == "org"`` — the *same* condition under which this function
       returns ``web``. So whenever the answer is ``web``, the only consumer of
       it has already raised, and the turn is executing in a scratchpad-controller
       pod via ``anton.cloud_turn`` instead.

       Reaching the pod means carrying the surface over the Redis job contract:
       cowork-server puts it in ``ScratchpadJobPayload.params``,
       scratchpad-controller forwards it in ``anton_turn._build_request`` (an
       explicit allowlist), and anton adds it to ``TurnRequestV1`` and passes it
       into the pod's ``ChatSessionConfig``. Four repos, tracked on ENG-1459.

       Not urgent as of 2026-08-19: the web surface has no real traffic yet
       (scratchpad-controller has no GitHub Release, and prod deploys only on
       one), so nothing is being mis-measured today — the desktop half is
       correct and the web half is absent rather than wrong.

    Deliberately NOT cached, unlike ``install_channel``: the channel is a fact
    about how the process was installed, while this reads settings that tests
    and a reload can legitimately change.

    Returns None when an explicit override is unrecognised — anton drops an
    unknown value anyway, and an absent surface is honestly unknown where a
    guessed one would silently join the population it is being compared with.

    Two populations inference gets wrong on purpose, both expected to declare
    themselves via the override: the hub snapshot instances being deprecated
    (local tenancy, but not desktops — they would otherwise inflate the very
    baseline web is measured against) and the enterprise container.
    """
    try:
        # anton owns the canonical vocabulary. Guarded because cowork-server
        # pins anton to a branch: a lock predating anton's half of ENG-1459 has
        # no such name, and telemetry must not raise on the way past.
        try:
            from anton.core.llm.tracing import VALID_SURFACES
        except ImportError:
            VALID_SURFACES = frozenset({"desktop", "web", "cli"})

        from cowork.common.settings.app_settings import get_app_settings

        settings = get_app_settings()
        override = (settings.surface_override or "").strip().lower()
        if override:
            if override in VALID_SURFACES:
                return override
            logger.warning(
                "build_info: ignoring COWORK_SURFACE=%r (expected one of %s)",
                override,
                sorted(VALID_SURFACES),
            )
            return None
        return "web" if settings.tenancy_mode == "org" else "desktop"
    except Exception:  # pragma: no cover - defensive: never fail a turn over telemetry
        logger.warning("build_info: could not resolve surface", exc_info=True)
        return None


_latch_unsupported_warned = False


def _warn_latch_unsupported_once() -> None:
    global _latch_unsupported_warned
    if _latch_unsupported_warned:
        return
    _latch_unsupported_warned = True
    logger.warning(
        "the installed anton has no initial_verifier_latch field, so the "
        "completion-verifier latch will not survive between messages and a "
        "failing verifier will re-diagnose on every one; bump the anton pin"
    )


def verifier_latch_kwarg(config_cls, stored) -> dict[str, object]:
    """``{"initial_verifier_latch": stored}`` for ``ChatSessionConfig``, or ``{}``.

    Lives beside ``surface_kwarg`` for the reason given in its docstring: more
    than one path originates a turn, and each needs the same guard. anton's
    verifier latch is ChatSession state, and this server rebuilds that object
    per message, so without carrying it the no-verdict counter restarts at zero
    every message, never reaches its threshold of two, and every message pays a
    full-history hand-back instead of one per conversation.

    Guarded like ``surface_kwarg``: anton is pinned by rev here and by a version
    floor on the desktop wheel, so the installed copy can predate this field,
    and an unexpected keyword to a plain dataclass would raise on EVERY turn.

    Unlike ``surface_kwarg`` the miss is announced once per process. An absent
    ``surface`` degrades a trace; an absent latch silently restores a
    customer-visible bug, and the pin is bumped by hand after anton merges, so
    the gap must not be invisible in the environment that has it.
    """
    import dataclasses

    try:
        if not any(f.name == "initial_verifier_latch" for f in dataclasses.fields(config_cls)):
            _warn_latch_unsupported_once()
            return {}
        return {"initial_verifier_latch": stored}
    except Exception:  # pragma: no cover - defensive: never fail a turn over this
        logger.warning("could not pass the stored verifier latch", exc_info=True)
        return {}


def surface_kwarg(config_cls) -> dict[str, str]:
    """``{"surface": ...}`` for ``ChatSessionConfig``, or ``{}`` — never raises.

    Lives here rather than in one harness because every path that ORIGINATES a
    turn needs it, and there is more than one: the anton harness serves the UI
    and the channel bots, and the connector probe runs its own turn (the
    datasource-connection path, which is one of the things ENG-1459 wants
    measured per surface). ``tests/test_trace_stamp_seams.py`` enforces that
    every such call site passes it.

    cowork-server pins anton to a *branch*, so the installed copy can predate
    anton's half of ENG-1459. Passing an unexpected keyword would then raise on
    **every turn**, which a telemetry field must never be able to do — so the
    kwarg is only produced when the installed dataclass actually declares it.

    Returns ``{}`` when the surface is unresolvable, so an unknown surface stays
    absent rather than being sent as a guess.
    """
    import dataclasses

    try:
        if not any(f.name == "surface" for f in dataclasses.fields(config_cls)):
            return {}
        resolved = surface()
        return {"surface": resolved} if resolved else {}
    except Exception:  # pragma: no cover - defensive: never fail a turn over telemetry
        logger.warning("could not resolve the trace surface", exc_info=True)
        return {}


@lru_cache(maxsize=None)
def install_channel() -> str:
    """How this server was installed: hosted / git / pypi / local / unknown.

    Two users on the same version via different channels are not equivalent
    for delivery questions — the PyPI channel pins anton inside the wheel
    while the git channel tracks a branch — so the channel is recorded
    alongside the versions rather than inferred from them.

    An explicit ``COWORK_INSTALL_CHANNEL`` wins over everything: a deployer
    declaring the channel knows more than any inference from inside the
    process. The hub snapshot instances need this — they run tenancy ``local``
    (org mode would change auth semantics) with cowork-server installed from
    PyPI *inside* the docker image, so inference would file a snapshot-bake
    deployment under ``pypi``, merging the slowest delivery channel into the
    wheel population. Their ``docker run`` passes
    ``COWORK_INSTALL_CHANNEL=hosted`` (anton_services snapshots/cowork).

    ``hosted`` is otherwise derived from org tenancy — a multi-tenant cloud
    deployment's provenance is a property of its image, not of how pip
    fetched the package inside it. Failing both, the answer comes from pip's
    ``direct_url.json``, the same signal the desktop updater switches on
    (``parseVcsInfo`` in cowork's ``update-logic.ts``): a VCS record means a
    git install; an editable / directory install is a developer checkout; no
    record at all means the package came from an index. ``unknown`` is
    returned rather than a guess when the metadata is missing or unreadable,
    so an absent channel is explicit in the data instead of being silently
    bucketed with PyPI.
    """
    # Imported lazily: app settings pull in the settings stack, and this module
    # is imported from the request path.
    from cowork.common.settings.app_settings import get_app_settings

    try:
        settings = get_app_settings()
        override = (settings.install_channel_override or "").strip().lower()
        if override:
            if override in VALID_CHANNELS:
                return override
            logger.warning(
                "build_info: ignoring COWORK_INSTALL_CHANNEL=%r (expected one of %s)",
                override,
                sorted(VALID_CHANNELS),
            )
        if settings.tenancy_mode == "org":
            return "hosted"
    except Exception:  # pragma: no cover - defensive: never fail a turn over telemetry
        logger.warning("build_info: could not read deployment settings", exc_info=True)

    try:
        raw = distribution(_SERVER_DIST).read_text("direct_url.json")
    except PackageNotFoundError:
        return "unknown"
    except Exception:  # pragma: no cover - unreadable dist-info
        logger.warning("build_info: could not read direct_url.json", exc_info=True)
        return "unknown"

    # No direct_url.json at all → installed from an index (PyPI).
    if raw is None:
        return "pypi"

    try:
        parsed = json.loads(raw)
    except ValueError:
        return "unknown"
    if not isinstance(parsed, dict):
        return "unknown"

    if (parsed.get("vcs_info") or {}).get("commit_id"):
        return "git"
    if parsed.get("dir_info") is not None or str(parsed.get("url", "")).startswith("file://"):
        return "local"
    return "unknown"


def build_trace_metadata(base: dict[str, str] | None = None) -> dict[str, str]:
    """Merge the running build's identity into ``base`` trace metadata.

    Server-derived values win over anything the client sent, so a caller
    can't misattribute a turn to another build. ``anton_version`` is the one
    exception in spirit: anton overwrites it again when it builds its outbound
    headers, because only anton knows which anton is loaded. Reporting it here
    anyway is what makes the version visible on installs whose anton predates
    that change — the exact delivery lag this ticket exists for.
    """
    merged = dict(base or {})
    # Belt for the hot path: this runs on every turn (API, channel bot,
    # scheduled run) purely for observability. Nothing inside is expected to
    # raise — the pieces have their own handlers — but an unattributable turn
    # is a far better failure than a failed turn, so a surprise here degrades
    # to "no build stamp" instead of 500-ing a user's message.
    try:
        server_version = _dist_version(_SERVER_DIST)
        anton_version = _dist_version(_ANTON_DIST)
        if server_version:
            merged[KEY_SERVER_VERSION] = server_version
        if anton_version:
            merged[KEY_ANTON_VERSION] = anton_version
        merged[KEY_INSTALL_CHANNEL] = install_channel()
    except Exception:  # pragma: no cover - exercised by the degradation test
        logger.warning("build_info: could not stamp the build on this turn", exc_info=True)
    return merged
