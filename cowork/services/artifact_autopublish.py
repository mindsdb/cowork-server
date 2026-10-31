"""Autopublish reconciliation: deciding what to publish, and publishing it.

No new state is introduced. The source of truth for "already published" is the
artifact's own `.published.json`, which is why retry is free: a publish that
failed, timed out, or died with the replica left no record, so the next
reconciliation sees the artifact as new again and republishes it. That is the
next turn end, or, after a remote turn whose pod named its artifacts, the
project's scheduled self-heal (request_self_heal).

cowork-server has no metrics backend (no prometheus, no statsd, no counter
facade), so the "metrics" this feature needs are structured log lines with the
stable `artifact_autopublish` prefix, ready to be wired to a collector.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from cowork.common.logger import log_context
from cowork.db.units import run_efs, run_efs_cleanup, run_on_unit_threads
from cowork.services.artifact_locks import LOCKS_DIRNAME, acquire, release
from cowork.services.artifact_publish_key import PublishKey
from cowork.services.product_permissions import ProductPermissionDenied, ProductPermissionUnavailable
from cowork.services.publish import PublishOwnedByOtherUser, PublishRejected, PublishTooLarge, publish_artifact

if TYPE_CHECKING:
    from cowork.db.scoped import TenantScope
    from cowork.services.artifact_ownership import OwnerResolution

logger = logging.getLogger(__name__)

# Owner-side publish access for a FIRST auto-publish. A re-publish preserves
# whatever the owner has since chosen — see `_access_for`.
#
# `owner_only` is NOT decoration. `anton.publish_access.resolve_access` treats
# "restricted with neither emails, nor an org, nor owner_only" as a caller
# mistake and degrades it to `public`, so `{"mode": "restricted", "emails": []}`
# would make every auto-published artifact world-readable. The flag is what
# makes "only the owner" expressible instead of collapsing into that hole.
#
# This used to be `org_allowed: True` — every auto-published artifact readable by
# the whole organization. An artifact is now private until its owner shares it
# (ENG-2316), which is the behaviour the Share control in the viewer exists to
# change.
AUTOPUBLISH_ACCESS: dict = {"mode": "restricted", "emails": [], "owner_only": True}

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
    # The stored `.published.json` entry this decision was made against, so a
    # re-publish can reconstruct the access the owner chose rather than
    # reapplying the first-publish default. None for a first publish.
    published_entry: dict | None = None


def _access_for(decision: PublishDecision) -> dict:
    """The access to publish this artifact with.

    A first publish gets the owner-only default. A re-publish MUST preserve what
    the owner has since chosen — reapplying the default here would silently
    revoke a share every time the agent touched the artifact, which is the whole
    reason the Share control would otherwise be untrustworthy (ENG-2316).

    `access_from_owner_side` is the canonical reconstruction, and it carries
    `owner_only` for exactly this reason: without that key an owner-only entry
    rebuilds as an empty selection and degrades to public.
    """
    from anton.publish_access import access_from_owner_side

    if not decision.published_entry:
        return dict(AUTOPUBLISH_ACCESS)
    return access_from_owner_side(decision.published_entry)


def needs_publish(folder: Path, artifacts_base: Path) -> PublishDecision:
    """Whether this artifact should be (re)published, and why not when it shouldn't.

    Order of the checks matters: the cheap ones come first, and the md5 bundle
    rebuild happens only after the mtime gate has already said something moved.
    """
    from cowork.services.artifacts import _user_files, content_mtime, load_published_map
    from cowork.services.publish import (
        _resolve_publish_target,
        compute_publish_md5,
        is_publishable_static,
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
    elif not is_publishable_static(publish_target, artifacts_base):
        # Suffix, not the on-disk name: _zip_html renames a single file to
        # index.html inside the archive, so an artifact whose primary is
        # `report.html` publishes normally. A non-content primary (README,
        # generation input) is refused by `publish_artifact`, so skip it here.
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
    # `entry` rides along so the re-publish keeps the owner's access choice.
    return PublishDecision("changed", None, is_fullstack, current_mtime, entry)


# ── reconciliation ────────────────────────────────────────────────────────


def _is_enabled(scope) -> bool:
    from cowork.common.settings.user_settings import get_user_settings

    return bool(getattr(get_user_settings(scope), "artifact_autopublish_enabled", False))


def _publish_url(scope) -> str:
    from cowork.common.settings.user_settings import get_user_settings
    from cowork.services.publish import _resolve_publish_endpoint

    publish_url, _unused_key = _resolve_publish_endpoint(get_user_settings(scope))
    return publish_url


def _active_workspace_id(scope) -> str | None:
    """The caller's picked MindsHub workspace, or None for the organization's Default."""
    from cowork.common.settings.user_settings import get_user_settings

    return getattr(get_user_settings(scope), "hub_workspace_id", "") or None


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


def _artifact_owners(
    session, *, artifacts_base: Path, project_id: str, slugs: list[str]
) -> dict[str, OwnerResolution] | None:
    """The owner of each slug under the project's root at ``artifacts_base``,
    or None when no root of the project sits there. Read-only, one query; run
    as a unit (cowork.db.units)."""
    from cowork.services.artifact_ownership import (
        artifact_root_for_base,
        resolve_artifact_owners,
    )

    source = artifact_root_for_base(session, project_id, artifacts_base)
    if source is None:
        return None
    return resolve_artifact_owners(session, source, slugs)


@dataclass(frozen=True)
class OwnedSlugs:
    """What the owner filter kept, and how many slugs it dropped and why."""

    owned: list[str]
    # Slugs another member owns.
    not_owner: int
    # Slugs with no recorded owner, or every slug when the lookup failed.
    owner_unknown: int


async def _owned_slugs(
    *, artifacts_base: Path, scope, project_id: str, slugs: list[str]
) -> OwnedSlugs:
    """Filter ``slugs`` down to the ones this scope's user owns.

    The project root is shared by every member (ENG-2056), so `_candidate_slugs`
    lists every member's artifacts, not just the caller's. Without this filter
    `autopublish_project_artifacts` plans a publish for artifacts it cannot
    legally publish, and each one fails downstream in `publish_authorization_key`
    with `ArtifactOwnerUnknown` or "Only the artifact owner can publish" — on
    every turn of every member, forever.

    The lookup runs as one database unit in a worker thread, so a wait for a
    connection never stalls the event loop. Fails closed: if the source cannot
    be found or anything raises, a full pool included, nothing is treated as
    owned rather than risking a publish attempt on someone else's artifact.
    """
    from cowork.db.units import run_db

    try:
        resolutions = await run_db(
            partial(
                _artifact_owners,
                artifacts_base=artifacts_base, project_id=project_id, slugs=slugs,
            ),
            scope=scope,
        )
    except Exception:
        logger.warning("artifact_autopublish owner filter failed", exc_info=True)
        return OwnedSlugs(owned=[], not_owner=0, owner_unknown=len(slugs))
    if resolutions is None:
        logger.warning(
            "artifact_autopublish owner filter: root not found for project=%s base=%s",
            project_id, artifacts_base,
        )
        return OwnedSlugs(owned=[], not_owner=0, owner_unknown=len(slugs))

    owned: list[str] = []
    not_owner = 0
    owner_unknown = 0
    for slug in slugs:
        resolution = resolutions[slug]
        if resolution.unknown:
            owner_unknown += 1
        elif str(resolution.owner_user_id) == str(scope.user_id):
            owned.append(slug)
        else:
            not_owner += 1
    return OwnedSlugs(owned=owned, not_owner=not_owner, owner_unknown=owner_unknown)


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


async def _release(artifacts_base: Path, slug: str, *, scope) -> None:
    """Free a slug's publish lock. It waits for a file store slot rather than
    failing, and a cancel that lands while it waits or runs waits for it: a
    lock left behind makes every later turn skip the slug as `lock_busy` until
    its TTL runs out."""
    await run_efs_cleanup(partial(release, artifacts_base, slug), scope=scope)


async def _acquire(artifacts_base: Path, slug: str, *, ttl_s: float, scope) -> bool:
    """Take a slug's publish lock. True when taken. It waits for a file store
    slot rather than failing.

    A cancel that lands while it waits for a slot leaves at once, and the
    lock is never taken. One that lands once the take runs waits for it, as
    run_efs does, and then frees a lock it took: the caller never sees that
    True, so nothing else would free the lock before its TTL."""
    taken = False

    def take() -> bool:
        nonlocal taken
        taken = acquire(artifacts_base, slug, ttl_s=ttl_s)
        return taken

    try:
        return await run_efs(take, scope=scope, deadline=False)
    except asyncio.CancelledError:
        if taken:
            await _release(artifacts_base, slug, scope=scope)
        raise


async def _publish_one(
    artifacts_base: Path,
    slug: str,
    api_key: str,
    publish_url: str,
    timeout_s: float,
    scope,
    access: dict | None = None,
    project_id: str | None = None,
) -> bool:
    """Publish one artifact. True when it landed. Never raises.

    `access` defaults to the first-publish owner-only access; callers pass
    `_access_for(decision)` so a re-publish keeps the owner's own choice.
    """
    folder = Path(artifacts_base) / slug
    try:
        await asyncio.wait_for(
            asyncio.to_thread(
                publish_artifact,
                folder,
                artifacts_base=Path(artifacts_base),
                api_key=api_key,
                publish_url=publish_url,
                access=dict(access) if access else dict(AUTOPUBLISH_ACCESS),
                # Required, not optional, on this path: the publisher reads
                # datasource secrets from the org-keyed connector vault, and
                # `vault_for_scope(None)` raises on an org deployment rather
                # than falling back to the shared namespace root. We only ever
                # get here with an org scope in hand (see the caller's guard).
                scope=scope,
                project_id=project_id,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        # to_thread is not cancellable: an upload already running goes on and
        # may yet write .published.json. So the lock is deliberately NOT
        # released and no retry is attempted: the next turn reads the record
        # and sees the real state. The TTL is what eventually frees the slug.
        # The lock is kept whether the upload is running or still queued for
        # its thread, since nothing here tells the two apart. A cancel (a
        # Stop, a shutdown) that lands once the upload is handed to its thread
        # keeps the lock the same way: it propagates past every handler here.
        _record("timeout", slug=slug, timeout_s=f"{timeout_s:.1f}")
        return False
    except PublishOwnedByOtherUser:
        # The upload service holds this report_id for another account, so a
        # retry cannot succeed and a fresh report_id would publish a copy. A
        # skip like `not_owner`, not a failure; no thread is left running.
        await _release(artifacts_base, slug, scope=scope)
        _record("skipped", slug=slug, reason="owned_by_other_user")
        return False
    except PublishTooLarge:
        # Every retry of the same bundle is refused again, and the 413 is
        # already logged by the classifier: a skip, not a failure.
        await _release(artifacts_base, slug, scope=scope)
        _record("skipped", slug=slug, reason="too_large")
        return False
    except Exception as exc:
        # A synchronous failure (a 502 a second in, ValueError from the target
        # resolver, PublisherUnavailable) leaves NO running thread, so holding the
        # lock for its whole TTL would block self-heal for minutes and break the
        # "next turn retries" promise.
        await _release(artifacts_base, slug, scope=scope)
        _record("failed", slug=slug, error=type(exc).__name__)
        # publish_artifact has already logged a PublishRejected's traceback.
        logger.warning(
            "artifact autopublish failed for %s", slug, exc_info=not isinstance(exc, PublishRejected),
        )
        return False
    await _release(artifacts_base, slug, scope=scope)
    _record("published", slug=slug)
    return True


async def autopublish_project_artifacts(
    artifacts_base: Path,
    scope,
    *,
    touched: set[str],
    project_id: str | None = None,
    only_touched: bool = False,
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

    With `only_touched`, the `touched` folders are opened by name
    (artifact_folders_named), the project's folder is not listed, and there is
    no second phase: the self-heal of the rest is requested from the
    scheduler's poll loop instead (request_self_heal).

    Never raises — a lost publication is recoverable on the next turn, a lost reply
    is not. The one exception is CancelledError, which propagates.

    Every settings read goes through `scope` explicitly rather than the ambient
    `use_settings_scope` binding, because an unscoped `get_user_settings()`
    outside a bound scope silently resolves LOCAL_SCOPE, which would read the
    global row for the org-scoped enable flag and the wrong provider for the
    publish URL. Inside a turn (handlers/responses.py `_produce_remote` on an
    org deployment), the turn's settings snapshot answers those reads
    (use_turn_settings), so none of them opens a connection.

    `project_id` is required in org mode: publishing resolves the artifact's
    owner by it (ENG-2961).
    """
    # Scope guards first: the enable flag is an org setting, so reading it is
    # only meaningful once we know we have an org scope to read it for.
    if scope is None or not getattr(scope, "org_mode", False):
        return set()
    if not scope.org_id or not scope.user_id:
        _record("skipped", reason="no_scope_identity")
        return set()
    if not await run_on_unit_threads(partial(_is_enabled, scope)):
        return set()
    if not project_id:
        # Fail closed once, instead of raising in `_publish_one` for every slug.
        _record("skipped", reason="no_project_id")
        return set()

    base = Path(artifacts_base)
    # Turn bookkeeping: each file store step waits for a slot rather than fail.
    if only_touched:
        from cowork.services.artifact_identity import artifact_folders_named

        named = await run_efs(
            partial(artifact_folders_named, base=base, slugs=set(touched)), scope=scope, deadline=False,
        )
        all_slugs = sorted(named.found)
        request_self_heal(SelfHeal(artifacts_base=base, scope=scope, project_id=project_id))
    else:
        all_slugs = await run_efs(partial(_candidate_slugs, base), scope=scope, deadline=False)
    owners = await _owned_slugs(
        artifacts_base=base, scope=scope, project_id=project_id, slugs=all_slugs,
    )
    all_slugs = owners.owned
    if owners.not_owner or owners.owner_unknown:
        _record("skipped", not_owner=owners.not_owner or None, owner_unknown=owners.owner_unknown or None)
    phase_one = [s for s in all_slugs if s in touched]
    phase_two = [s for s in all_slugs if s not in touched]

    # Settings reads, each through a session of its own: off the loop.
    publish_url = await run_on_unit_threads(partial(_publish_url, scope))
    workspace_id = await run_on_unit_threads(partial(_active_workspace_id, scope))
    key = PublishKey(
        scope.user_id, scope.org_id, min_ttl_s=timeout_s + 60.0, workspace_id=workspace_id
    )
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
            plans = await run_efs(partial(_plan, base, phase_slugs), scope=scope, deadline=False)
            for slug, decision in plans:
                if len(published) >= limit:
                    _record("deferred", slug=slug, reason="limit")
                    continue
                remaining = min(phase_deadline, started + budget_s) - time.monotonic()
                if remaining < _MIN_START_BUDGET_S:
                    _record("deferred", slug=slug, reason="budget")
                    continue
                if not await _acquire(base, slug, ttl_s=lock_ttl, scope=scope):
                    _record("lock_busy", slug=slug)
                    continue
                try:
                    api_key = await key.get()
                except (ProductPermissionDenied, ProductPermissionUnavailable):
                    await _release(base, slug, scope=scope)
                    _record("no_key", slug=slug)
                    return published
                except BaseException:
                    # A cancel, or a failure such as a settings load that
                    # raises. No publish has started, so nothing else would
                    # free the lock before its TTL. Past this point, the only
                    # await before the publish thread starts is a release,
                    # which a cancel waits for.
                    await _release(base, slug, scope=scope)
                    raise
                if not api_key:
                    await _release(base, slug, scope=scope)
                    _record("no_key", slug=slug)
                    return published
                if await _publish_one(
                    base, slug, api_key, publish_url, min(timeout_s, remaining), scope,
                    _access_for(decision), project_id=project_id,
                ):
                    published.add(slug)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("artifact autopublish reconciliation failed", exc_info=True)
    finally:
        await key.revoke()
    return published


# ── scheduled self-heal ───────────────────────────────────────────────────

# How long a member's project waits between two of its scheduled self-heals.
# A remote turn whose pod named its artifacts publishes only those
# (`only_touched`), so this bounds how long the rest of the project waits for
# a retry, such as an artifact whose publish failed.
SELF_HEAL_INTERVAL_S = 15 * 60.0


@dataclass(frozen=True)
class SelfHeal:
    """One member's project for a scheduled self-heal: every artifact below
    `artifacts_base` that the scope's user owns."""

    artifacts_base: Path
    scope: TenantScope
    project_id: str


# Asked for since each one last started, waiting for the poll loop.
_self_heals_requested: set[SelfHeal] = set()
# When each one last started, on time.monotonic().
_self_heals_started_at: dict[SelfHeal, float] = {}
# The task of each one started, so a second never runs beside it and a
# shutdown can cancel it (cancel_self_heals).
_self_heal_tasks: dict[SelfHeal, asyncio.Task] = {}


def request_self_heal(heal: SelfHeal) -> None:
    """Ask for a self-heal of `heal`'s project: the reconciler with nothing
    touched, so it lists the project's artifacts and publishes what needs it.
    The scheduler's poll loop starts it (start_due_self_heals) once
    SELF_HEAL_INTERVAL_S has passed since its last one; asking again before
    then changes nothing."""
    _self_heals_requested.add(heal)


async def _run_self_heal(heal: SelfHeal) -> None:
    try:
        await autopublish_project_artifacts(
            heal.artifacts_base, heal.scope, touched=set(), project_id=heal.project_id,
        )
    except Exception:
        # The reconciler logs its own failures; this covers a settings read
        # it makes before them, such as one a full pool refuses.
        logger.warning(
            "artifact autopublish self-heal failed", exc_info=True,
            extra=log_context(project_id=heal.project_id),
        )


def start_due_self_heals() -> list[asyncio.Task]:
    """Start, each as a task of its own, every requested self-heal whose
    project's last one started at least SELF_HEAL_INTERVAL_S ago and has
    finished. The scheduler's poll loop calls this once per poll
    (cowork/scheduler.py). Returns the tasks it started."""
    now = time.monotonic()
    started: list[asyncio.Task] = []
    for heal in list(_self_heals_requested):
        last = _self_heals_started_at.get(heal)
        if last is not None and now - last < SELF_HEAL_INTERVAL_S:
            continue
        running = _self_heal_tasks.get(heal)
        if running is not None and not running.done():
            continue
        _self_heals_requested.discard(heal)
        _self_heals_started_at[heal] = now
        task = asyncio.create_task(_run_self_heal(heal))
        _self_heal_tasks[heal] = task
        started.append(task)
    # A project with no self-heal running and none for an interval holds
    # nothing back, so its entries go.
    for heal, last in list(_self_heals_started_at.items()):
        task = _self_heal_tasks.get(heal)
        if now - last >= SELF_HEAL_INTERVAL_S and (task is None or task.done()):
            del _self_heals_started_at[heal]
            _self_heal_tasks.pop(heal, None)
    return started


def cancel_self_heals() -> list[asyncio.Task]:
    """Cancel the self-heals still running, for a shutdown, and return them
    for the caller to wait on. Their projects' next requests start them again
    on a later boot."""
    running = [task for task in _self_heal_tasks.values() if not task.done()]
    for task in running:
        task.cancel()
    return running
