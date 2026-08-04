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


@lru_cache(maxsize=None)
def _dist_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - metadata machinery should never break a turn
        logger.warning("build_info: could not read version for %s", name, exc_info=True)
        return None


@lru_cache(maxsize=None)
def install_channel() -> str:
    """How this server was installed: hosted / git / pypi / local / unknown.

    Two users on the same version via different channels are not equivalent
    for delivery questions — the PyPI channel pins anton inside the wheel
    while the git channel tracks a branch — so the channel is recorded
    alongside the versions rather than inferred from them.

    ``hosted`` wins over the install source because a hosted deployment's
    provenance is a property of the snapshot image, not of the user's machine.
    Otherwise the answer comes from pip's ``direct_url.json``, the same signal
    the desktop updater switches on (``parseVcsInfo`` in cowork's
    ``update-logic.ts``): a VCS record means a git install; an editable /
    directory install is a developer checkout; no record at all means the
    package came from an index. ``unknown`` is returned rather than a guess
    when the metadata is missing or unreadable, so an absent channel is
    explicit in the data instead of being silently bucketed with PyPI.
    """
    # Imported lazily: app settings pull in the settings stack, and this module
    # is imported from the request path.
    from cowork.common.settings.app_settings import get_app_settings

    try:
        if get_app_settings().tenancy_mode == "org":
            return "hosted"
    except Exception:  # pragma: no cover - defensive: never fail a turn over telemetry
        logger.warning("build_info: could not read tenancy mode", exc_info=True)

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
    server_version = _dist_version(_SERVER_DIST)
    anton_version = _dist_version(_ANTON_DIST)
    if server_version:
        merged[KEY_SERVER_VERSION] = server_version
    if anton_version:
        merged[KEY_ANTON_VERSION] = anton_version
    merged[KEY_INSTALL_CHANNEL] = install_channel()
    return merged
