"""Autopublish reconciliation: deciding what to publish, and publishing it.

No new state is introduced. The source of truth for "already published" is the
artifact's own `.published.json`, which is why retry is free: a publish that
failed, timed out, or died with the replica left no record, so the next turn sees
the artifact as new again and republishes it.

cowork-server has no metrics backend (no prometheus, no statsd, no counter
facade), so the "metrics" this feature needs are structured log lines with the
stable `artifact_autopublish` prefix, ready to be wired to a collector.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cowork.services.artifact_locks import LOCKS_DIRNAME, acquire, release
from cowork.services.artifact_publish_key import PublishKey
from cowork.services.publish import publish_artifact

logger = logging.getLogger(__name__)

# Owner-side publish access for every auto-published artifact.
#
# `org_allowed` is NOT decoration. `anton.publish_access.resolve_access` treats
# "restricted with neither emails nor an org" as a caller mistake and degrades it
# to `public`, so `{"mode": "restricted", "emails": []}` would make every
# auto-published artifact world-readable. With the flag set, access is granted by
# two independent conditions in auth — owner-by-FK and org membership — which also
# removes the dependency on the never-live-verified "empty emails still grants the
# owner" behavior.
AUTOPUBLISH_ACCESS: dict = {"mode": "restricted", "emails": [], "org_allowed": True}

# Below this much remaining budget a publish is not started at all: entering
# `wait_for` with a sliver of time only guarantees a timeout, an abandoned upload
# thread and a held lock.
_MIN_START_BUDGET_S = 5.0

# Reasons a candidate is skipped. Kept distinct because a single None would make
# the skip log unusable: "nothing to do" and "cannot be published at all" demand
# different follow-up.
REASON_NO_CONTENT = "no_content"
REASON_NOT_PUBLISHABLE = "not_publishable"
REASON_UNPUBLISHED = "unpublished"
REASON_UNCHANGED = "unchanged"


@dataclass(frozen=True)
class PublishDecision:
    action: Literal["new", "changed"] | None
    reason: str | None = None
    is_fullstack: bool = False
    content_mtime: int = 0


def needs_publish(folder: Path, artifacts_base: Path) -> PublishDecision:
    """Whether this artifact should be (re)published, and why not when it shouldn't.

    Order of the checks matters: the cheap ones come first, and the md5 bundle
    rebuild happens only after the mtime gate has already said something moved.
    """
    from cowork.services.artifacts import _user_files, content_mtime, load_published_map
    from cowork.services.publish import (
        PUBLISHABLE_STATIC_SUFFIXES,
        _resolve_publish_target,
        compute_publish_md5,
    )

    if not _user_files(folder):
        return PublishDecision(None, REASON_NO_CONTENT)

    try:
        publish_target, published_dir, published_key, is_fullstack = _resolve_publish_target(
            folder, container_dirs=[artifacts_base]
        )
    except Exception:
        logger.warning("Could not resolve publish target for %s", folder, exc_info=True)
        return PublishDecision(None, REASON_NOT_PUBLISHABLE)

    if is_fullstack:
        # /upload is strict here: literally backend.py plus static/index.html.
        if not (folder / "backend.py").is_file() or not (folder / "static" / "index.html").is_file():
            return PublishDecision(None, REASON_NOT_PUBLISHABLE, True)
    elif publish_target.suffix.lower() not in PUBLISHABLE_STATIC_SUFFIXES:
        # Suffix, not the on-disk name: _zip_html renames a single file to
        # index.html inside the archive, so an artifact whose primary is
        # `report.html` publishes normally.
        return PublishDecision(None, REASON_NOT_PUBLISHABLE)

    current_mtime = content_mtime(folder)
    entry = load_published_map(published_dir).get(published_key)

    if not isinstance(entry, dict) or not entry.get("report_id"):
        return PublishDecision("new", None, is_fullstack, current_mtime)
    if entry.get("published") is False:
        # Explicitly unpublished — never resurrect it on our own.
        return PublishDecision(None, REASON_UNPUBLISHED, is_fullstack, current_mtime)

    published_mtime = entry.get("published_mtime")
    if isinstance(published_mtime, (int, float)) and current_mtime <= published_mtime:
        return PublishDecision(None, REASON_UNCHANGED, is_fullstack, current_mtime)

    digest = compute_publish_md5(folder, artifacts_base=artifacts_base)
    if digest is None or digest == entry.get("last_md5"):
        # None means "can't tell" — never republish on a guess.
        return PublishDecision(None, REASON_UNCHANGED, is_fullstack, current_mtime)
    return PublishDecision("changed", None, is_fullstack, current_mtime)


# ── reconciliation ────────────────────────────────────────────────────────


def _is_enabled(scope) -> bool:
    from cowork.common.settings.user_settings import get_user_settings

    return bool(getattr(get_user_settings(scope), "artifact_autopublish_enabled", False))


def _publish_url(scope) -> str:
    from cowork.common.settings.user_settings import get_user_settings
    from cowork.services.publish import _resolve_publish_endpoint

    publish_url, _unused_key = _resolve_publish_endpoint(get_user_settings(scope))
    return publish_url


def _record(result: str, **fields: object) -> None:
    """The metric. cowork-server has no metrics backend, so this is a structured
    log line with a stable prefix — greppable now, collectable later.

    Strictly `key=value` pairs and no free-form tail, so the line parses the same
    way whatever the caller passes.

    WARNING, not INFO, and not because any of these results is a fault —
    `result=published` is the happy path. Every deployment that runs this feature
    runs at LOG_LEVEL=WARNING (deployment/cowork-server/values-{staging,prod}.yaml),
    so an INFO line is invisible on exactly the deployments it exists for. That
    made a live diagnosis impossible: the reconciler also swallows publish
    failures by design, so with the metric filtered out too, an artifact that
    never published looked identical to one that was never eligible — no signal
    of any kind, anywhere.

    Volume is bounded: at most one line per artifact considered, per turn, and
    only in org mode.
    """
    tail = " ".join(f"{k}={v}" for k, v in fields.items() if v not in (None, ""))
    logger.warning("artifact_autopublish result=%s %s", result, tail)


def _candidate_slugs(artifacts_base: Path) -> list[str]:
    try:
        children = sorted(Path(artifacts_base).iterdir())
    except OSError:
        return []
    return [
        child.name
        for child in children
        if child.is_dir()
        and child.name != LOCKS_DIRNAME
        and (child / "metadata.json").is_file()
    ]


def _plan(artifacts_base: Path, slugs: list[str]) -> list[tuple[str, PublishDecision]]:
    """Decide and order one phase's work: static before fullstack, newest first.

    Static goes first because publishing a fullstack artifact triggers a dependency
    install on the backend and regularly hits the gateway timeout
    (ENG-1547/ENG-1580); with the opposite order one slow fullstack would eat the
    whole budget and the static artifacts would never get a link.

    Skips are counted, not logged per artifact: `needs_publish` runs for every
    candidate on every turn, so a per-artifact line would bury `published` and
    `failed` under dozens of `unchanged` on any project with history.
    """
    planned: list[tuple[str, PublishDecision]] = []
    skipped: dict[str, int] = {}
    for slug in slugs:
        decision = needs_publish(Path(artifacts_base) / slug, Path(artifacts_base))
        if decision.action is None:
            reason = decision.reason or "unknown"
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        planned.append((slug, decision))
    if skipped:
        _record("skipped", **skipped)
    planned.sort(key=lambda item: (item[1].is_fullstack, -item[1].content_mtime))
    return planned


async def _publish_one(
    artifacts_base: Path, slug: str, api_key: str, publish_url: str, timeout_s: float, scope
) -> bool:
    """Publish one artifact. True when it landed. Never raises."""
    folder = Path(artifacts_base) / slug
    try:
        await asyncio.wait_for(
            asyncio.to_thread(
                publish_artifact,
                folder,
                artifacts_base=Path(artifacts_base),
                api_key=api_key,
                publish_url=publish_url,
                access=dict(AUTOPUBLISH_ACCESS),
                # Required, not optional, on this path: the publisher reads
                # datasource secrets from the org-keyed connector vault, and
                # `vault_for_scope(None)` raises on an org deployment rather
                # than falling back to the shared namespace root. We only ever
                # get here with an org scope in hand (see the caller's guard).
                scope=scope,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        # to_thread is not cancellable: the upload is still running and may yet
        # write .published.json. So the lock is deliberately NOT released and no
        # retry is attempted — the next turn reads the record and sees the real
        # state. The TTL is what eventually frees the slug.
        _record("timeout", slug=slug, timeout_s=f"{timeout_s:.1f}")
        return False
    except Exception as exc:
        # A synchronous failure (a 502 a second in, ValueError from the target
        # resolver, PublisherUnavailable) leaves NO running thread, so holding the
        # lock for its whole TTL would block self-heal for minutes and break the
        # "next turn retries" promise.
        release(artifacts_base, slug)
        _record("failed", slug=slug, error=type(exc).__name__)
        logger.warning("artifact autopublish failed for %s", slug, exc_info=True)
        return False
    release(artifacts_base, slug)
    _record("published", slug=slug)
    return True


async def autopublish_project_artifacts(
    artifacts_base: Path,
    scope,
    *,
    touched: set[str],
    limit: int = 5,
    budget_s: float = 60.0,
    touched_budget_s: float = 30.0,
    timeout_s: float = 60.0,
) -> set[str]:
    """Publish every artifact in this project that needs it. Returns the slugs
    actually (re)published, so the caller can build their cards.

    Two phases: what this turn wrote (`touched`) first under its own budget, then
    the rest of the project as self-heal on whatever budget is left. The split
    matters because the user is waiting for the artifact they just got, while a
    backlog of older artifacts can wait for the next turn.

    Never raises — a lost publication is recoverable on the next turn, a lost reply
    is not. The one exception is CancelledError, which propagates.

    Every settings read goes through `scope` explicitly rather than the ambient
    `use_settings_scope` binding. The remote-turn producer that drives this on an
    org deployment (handlers/responses.py `_produce_remote`) is a DETACHED task
    with no ambient scope bound, and an unscoped `get_user_settings()` silently
    resolves LOCAL_SCOPE — which would read the global row for the org-scoped
    enable flag and the wrong provider for the publish URL.
    """
    # Scope guards first: the enable flag is an org setting, so reading it is
    # only meaningful once we know we have an org scope to read it for.
    if scope is None or not getattr(scope, "org_mode", False):
        return set()
    if not scope.org_id or not scope.user_id:
        _record("skipped", reason="no_scope_identity")
        return set()
    if not _is_enabled(scope):
        return set()

    base = Path(artifacts_base)
    all_slugs = _candidate_slugs(base)
    phase_one = [s for s in all_slugs if s in touched]
    phase_two = [s for s in all_slugs if s not in touched]

    publish_url = _publish_url(scope)
    key = PublishKey(scope.user_id, scope.org_id, min_ttl_s=timeout_s + 60.0)
    started = time.monotonic()
    published: set[str] = set()
    lock_ttl = timeout_s * 3

    try:
        for phase_slugs, phase_deadline in (
            (phase_one, started + touched_budget_s),
            (phase_two, started + budget_s),
        ):
            if not phase_slugs:
                continue
            # Budget check BEFORE planning: `_plan` runs `needs_publish` for every
            # candidate — a recursive rglob each, plus a bundle re-zip whenever the
            # mtime gate fires. On a project with dozens of artifacts that is real
            # work, and doing it only to log `deferred` is waste.
            if (started + budget_s) - time.monotonic() < _MIN_START_BUDGET_S:
                _record("deferred", reason="budget", slugs=len(phase_slugs))
                continue
            for slug, _decision in _plan(base, phase_slugs):
                if len(published) >= limit:
                    _record("deferred", slug=slug, reason="limit")
                    continue
                remaining = min(phase_deadline, started + budget_s) - time.monotonic()
                if remaining < _MIN_START_BUDGET_S:
                    _record("deferred", slug=slug, reason="budget")
                    continue
                if not acquire(base, slug, ttl_s=lock_ttl):
                    _record("lock_busy", slug=slug)
                    continue
                api_key = await key.get()
                if not api_key:
                    release(base, slug)
                    _record("no_key", slug=slug)
                    return published
                if await _publish_one(base, slug, api_key, publish_url, min(timeout_s, remaining), scope):
                    published.add(slug)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("artifact autopublish reconciliation failed", exc_info=True)
    finally:
        await key.revoke()
    return published
