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


def supported_kwargs(config_cls, **candidates) -> dict:
    """Only those `candidates` the installed dataclass actually declares.

    Same hazard as `surface_kwarg`: this server pins anton to a rev, so a
    field it knows about can be absent from the installed copy, and passing an
    unexpected keyword raises on **every turn**. A dropped field degrades that
    feature; raising takes the whole turn down.
    """
    import dataclasses

    try:
        declared = {f.name for f in dataclasses.fields(config_cls)}
    except Exception:  # pragma: no cover - defensive: never fail a turn over this
        logger.warning("could not inspect %s for optional kwargs", config_cls, exc_info=True)
        return {}
    dropped = [k for k in candidates if k not in declared]
    if dropped:
        logger.debug("installed anton has no %s; not passing it", ", ".join(dropped))
    return {k: v for k, v in candidates.items() if k in declared}


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


def _uuid_or_none(value) -> str | None:
    """Canonical lowercase UUID, or None. The only shape an account id may take
    on its way to analytics: anything else (an email being the likely mistake)
    would reach PostHog as a distinct_id."""
    from uuid import UUID

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return str(UUID(value.strip()))
    except ValueError:
        return None


def account_ids(user_id, org_id) -> dict[str, str]:
    """``{"user_id", "organization_id"}`` for whichever of the two is a UUID."""
    out: dict[str, str] = {}
    user = _uuid_or_none(user_id)
    org = _uuid_or_none(org_id)
    if user:
        out["user_id"] = user
    if org:
        out["organization_id"] = org
    return out


def desktop_account() -> dict[str, str]:
    """The signed-in desktop user's account ids, from the held MindsHub JWT.

    For analytics attribution only (ENG-2121): anton keys ``turn_completed`` on
    ``user_id``, so a completed turn joins the same PostHog person as sign-up
    and payment. The desktop sidecar has no principal; the only identity it
    holds is the credential the desktop app hands over (``runtime_credential``),
    and when that is a Keycloak JWT its ``sub`` is the distinct_id the desktop
    renderer already keys on, read from the same claims
    (``activate_organization`` for the org, as the renderer does).

    **Decoded, not verified.** Same as the renderer: this decides nothing about
    access, and only the service the token is forwarded to can verify it. A
    forged token here could only misattribute that machine's own turns, which
    anyone can already do by posting to PostHog directly.

    Only the two ids leave this function, never the email or name the token
    also carries. Returns ``{}`` for anything else: no credential, a
    user-supplied ``mdb_`` API key (no identity in it), a malformed token, or
    org mode (the holder returns None there). Never raises.
    """
    import base64

    try:
        from cowork.common.settings import runtime_credential

        token = runtime_credential.get_minds_credential() or ""
        parts = token.split(".")
        if len(parts) != 3 or not parts[1]:
            return {}
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
        if not isinstance(payload, dict):
            return {}
        org = payload.get("activate_organization")
        org_id = org.get("id") if isinstance(org, dict) else None
        ids = account_ids(payload.get("sub"), org_id)
        # An org without a user identifies nobody; keep the pair or the user.
        return ids if "user_id" in ids else {}
    except Exception:
        logger.debug("build_info: could not read the desktop account", exc_info=True)
        return {}


def account_kwargs(config_cls) -> dict[str, str]:
    """``{"user_id", "organization_id"}`` for ``ChatSessionConfig``, or ``{}``.

    Same version-skew guard as ``surface_kwarg``: the pinned anton may predate
    ENG-2121, and an unexpected keyword would raise on every turn. Absent
    rather than empty when there is no account, so the event falls back to the
    install fingerprint exactly as before. Never raises.
    """
    try:
        return supported_kwargs(config_cls, **desktop_account())
    except Exception:  # pragma: no cover - defensive: never fail a turn over telemetry
        logger.warning("could not resolve the turn's account", exc_info=True)
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
