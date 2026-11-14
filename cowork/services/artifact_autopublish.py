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
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple
from uuid import UUID

from pydantic import ValidationError

from cowork.common.logger import log_context
from cowork.db.units import run_efs, run_efs_cleanup, run_on_unit_threads
from cowork.models.artifact_owner_choice import AccessMode
from cowork.schemas.artifact_access import ArtifactAccessRequest
from cowork.services.artifact_locks import LOCKS_DIRNAME, acquire, release
from cowork.services.artifact_publish_key import PublishKey
from cowork.services.product_permissions import ProductPermissionDenied, ProductPermissionUnavailable
from cowork.services.publish import (
    PublishOwnedByOtherUser,
    PublishRejected,
    PublishTooLarge,
    kept_access,
    publish_artifact,
)

if TYPE_CHECKING:
    from cowork.db.scoped import TenantScope
    from cowork.services.artifact_identity import ListedIds
    from cowork.services.artifact_ownership import OwnerResolution
    from cowork.services.artifact_sort_times import SortTimeRow, SortTimesRead

logger = logging.getLogger(__name__)

# Owner-side publish access for a FIRST auto-publish. A re-publish preserves
# whatever the owner has since chosen (`_access_for`).
#
# `owner_only` is NOT decoration. `anton.publish_access.resolve_access` treats
# "restricted with neither emails, nor an org, nor owner_only" as a caller
# mistake and degrades it to `public`, so a restricted access with no emails
# alone would make every auto-published artifact world-readable. The flag is
# what makes "only the owner" expressible instead of collapsing into that hole.
#
# An auto-published artifact is private to its owner until they share it,
# which is what the Share control in the viewer changes.
AUTOPUBLISH_ACCESS = ArtifactAccessRequest(mode=AccessMode.RESTRICTED, allowed_emails=[], owner_only=True)

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


def _access_for(decision: PublishDecision) -> ArtifactAccessRequest:
    """The access to publish this artifact with.

    A first publish gets the owner-only default. A re-publish MUST preserve what
    the owner has since chosen: reapplying the default here would silently
    revoke a share every time the agent touched the artifact, and the Share
    control could not be trusted.

    `kept_access` is the canonical reconstruction, and it carries `owner_only`
    for exactly this reason: without that key an owner-only entry rebuilds as
    an empty selection and degrades to public. This path is MindsHub mode
    only, so a restricted artifact's audience comes from MindsHub's
    rule, which publish_artifact reads; a failed read fails this publish.
    """
    if not decision.published_entry:
        return AUTOPUBLISH_ACCESS
    return kept_access(decision.published_entry, org_mode=True)


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
        # Explicitly unpublished: never resurrect it on our own. The pod can
        # flip this flag back; publish_artifact refuses that re-publish while
        # the owner's recorded choice says unpublished
        # (cowork.services.artifact_owner_choice).
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


def _candidate_slugs(*, artifacts_base: Path) -> list[str]:
    """The artifact folders the reconciler's full pass considers, sorted
    (artifact_folders_below, which never lists the project's lock folder).
    One the listing could not read waits for a later pass."""
    from cowork.services.artifact_identity import artifact_folders_below

    return sorted(artifact_folders_below(base=Path(artifacts_base)).found)


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
    access: ArtifactAccessRequest | None = None,
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
                access=access or AUTOPUBLISH_ACCESS,
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
        # publish_artifact already logged every PublishRejected: an upload
        # failure with its traceback, and an AccessModeChanged refusal as a
        # warning with its reason, which needs none.
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
    no second phase: the turn end asks for the project's scheduled self-heal
    instead, as its record starts (request_turn_end_self_heal), which
    publishes the rest.

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
    from cowork.services.projects import path_in_comparison_sandbox

    if path_in_comparison_sandbox(artifacts_base):
        # publish_artifact would refuse each one anyway; skipping here keeps a
        # comparison turn from minting a publish key and logging failures.
        _record("skipped", reason="comparison_side")
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
        # The project's lock folder is never an artifact, as in a full
        # listing (artifact_folders_below), even when agent code put a
        # metadata.json there.
        all_slugs = sorted(named.found - {LOCKS_DIRNAME})
    else:
        all_slugs = await run_efs(partial(_candidate_slugs, artifacts_base=base), scope=scope, deadline=False)
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
                try:
                    access = _access_for(decision)
                except ValidationError:
                    # The pod can write the entry, so its access fields can
                    # hold any type: this artifact's re-publish fails, and the
                    # rest of the pass goes on.
                    _record("failed", slug=slug, error="ValidationError")
                    logger.warning(
                        "artifact autopublish could not read the kept access",
                        extra={"artifact_slug": slug},
                        exc_info=True,
                    )
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
                    access, project_id=project_id,
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
# A remote turn whose pod named its artifacts opens and publishes only those
# (`only_touched`), so this bounds how long the rest of the project waits for
# a retry, such as an artifact whose publish failed, and how long a copy no
# turn end claims waits for its id to be marked conflicted. A full listing
# that could not read a folder asks for the next location check alone, with
# no publish (request_self_heal_for_unread_folders). Each replica keeps its
# own requests and runs its own poll (scheduler._scheduler_loop calls
# start_due_self_heals on every replica), and that request stays on the
# replica that listed. So a folder that stays unreadable costs one listing of
# the project per interval on each replica that has listed it, for each
# member and project, and for each project's channel turns, until the folder
# reads again or the replica restarts.
SELF_HEAL_INTERVAL_S = 15 * 60.0


@dataclass(frozen=True)
class SelfHeal:
    """One member's project for a scheduled self-heal: the location of every
    artifact below `artifacts_base`, then, when the publish was asked for
    too (request_self_heal), the publish of those the scope's user owns. A
    channel turn's scope names no user, so its self-heal checks the
    locations only."""

    artifacts_base: Path
    scope: TenantScope
    project_id: str


# Asked for since each one last started, waiting for the poll loop.
_self_heals_requested: set[SelfHeal] = set()
# Those of them whose publish was asked for too (request_self_heal). One
# asked for only by request_location_check checks the locations alone.
_self_heal_publishes_requested: set[SelfHeal] = set()
# When each one last started, on time.monotonic().
_self_heals_started_at: dict[SelfHeal, float] = {}
# The task of each one started, so a second never runs beside it and a
# shutdown can cancel it (cancel_self_heals).
_self_heal_tasks: dict[SelfHeal, asyncio.Task] = {}
# The location checks a check run inside keep_location_checks asks for, kept
# apart from _self_heals_requested, or None outside one. A ContextVar, so
# only the requests of the task that entered the block, and of any task it
# starts there, land in it.
_location_checks_kept: ContextVar[set[SelfHeal] | None] = ContextVar("_location_checks_kept", default=None)


def request_self_heal(heal: SelfHeal) -> None:
    """Ask for a self-heal of `heal`'s project: a check of every artifact's
    location (_record_listed_locations), then the reconciler with nothing
    touched, so it lists the project's artifacts and publishes what needs it
    behind its own gates (autopublish_project_artifacts). The scheduler's
    poll loop starts it (start_due_self_heals) once SELF_HEAL_INTERVAL_S has
    passed since its last one; asking again before then changes nothing."""
    _self_heals_requested.add(heal)
    _self_heal_publishes_requested.add(heal)


def request_location_check(heal: SelfHeal) -> None:
    """Ask for the location half of `heal`'s self-heal alone
    (_record_listed_locations), for a project whose locations a listing has
    to check again. The poll starts it as it starts a self-heal
    (start_due_self_heals), and it publishes nothing unless the publish is
    asked for too (request_self_heal) before it starts. Inside
    keep_location_checks the request goes to that block's own set instead,
    and the poll never sees it."""
    kept = _location_checks_kept.get()
    if kept is not None:
        kept.add(heal)
        return
    _self_heals_requested.add(heal)


@contextmanager
def keep_location_checks() -> Iterator[set[SelfHeal]]:
    """Collect the location checks this task asks for inside the block
    (request_location_check) in the set it yields, instead of asking this
    replica's poll for them (start_due_self_heals). The heal sweep
    (artifact_heal_sweep) runs each project's check inside it: a check that
    asks for another has fallen short, and the sweep, not the poll, lists
    that project again, one project at a time. The set lives in a
    ContextVar, so a turn end or a self-heal in another task that asks for
    the same project's self-heal during the block still reaches the poll."""
    kept: set[SelfHeal] = set()
    token = _location_checks_kept.set(kept)
    try:
        yield kept
    finally:
        _location_checks_kept.reset(token)


def request_turn_end_self_heal(*, artifacts_base: Path, scope: TenantScope | None, project_id: str | None) -> None:
    """Ask for the self-heal of the project a remote turn end worked in, when
    the turn end does not check the locations of the project's artifacts
    itself. A turn end whose pod named the artifacts its tools touched opens
    only the turn's own folders and lists nothing, so the self-heal's
    location check is what finds a copy no turn end claims, such as one a
    shell `cp -r` makes. Each producer asks for it as its turn end's record
    starts, before the diff (handlers.responses `record_artifacts`,
    turnqueue.remote_turn `remote_turn_events`), so a turn end whose diff
    fails, or that a Stop cuts, still asks. A turn end whose pod sent no
    list lists the project and checks every id itself, so it asks only when
    its diff or the record of what the diff listed fails. Asked for wherever
    artifacts have location rows (artifact_locations.keeps_locations),
    whether or not the organization's autopublish is on, and for a channel
    turn too; the publish half keeps the reconciler's own gates, the enable
    flag among them, and runs only for a scope that names a user
    (_run_self_heal). Adds an entry to an in-memory set, and makes no file
    store call."""
    heal = _project_self_heal(artifacts_base=artifacts_base, scope=scope, project_id=project_id)
    if heal is not None:
        request_self_heal(heal)


def request_project_location_check(*, artifacts_base: Path, scope: TenantScope | None, project_id: str | None) -> None:
    """Ask for the location check alone (request_location_check) of the
    project whose shared root is `artifacts_base`, built as a turn end
    builds its project's self-heal: for a writer that could not record a
    row of that root, such as a project files PUT or a task move, and for a
    list whose rows named folders that are gone or build no card. The
    check's listing holds the root, writes the time of every folder it
    reads, drops the rows of folders gone from it or holding no id, and
    keeps the rows of folders it could not read (_record_listing).
    Nothing is asked where artifacts have no location rows or no project is
    named (_project_self_heal). Adds an entry to an in-memory set, and makes
    no file store call."""
    heal = _project_self_heal(artifacts_base=artifacts_base, scope=scope, project_id=project_id)
    if heal is not None:
        request_location_check(heal)


def request_self_heal_for_unread_folders(
    *, listing: ListedIds, artifacts_base: Path, scope: TenantScope | None, project_id: str | None,
) -> None:
    """After a full listing of a project's shared root, the self-heal's or a
    remote turn end's with no pod list, that could not read a folder or the
    root itself: log one WARNING, and ask for the project's location check
    alone (request_location_check), so the poll lists the project again
    once SELF_HEAL_INTERVAL_S has passed since its last self-heal
    (start_due_self_heals), or the heal sweep does for its own check
    (keep_location_checks), and publishes nothing for it. A copy no turn end
    claims, in a folder the listing could not read, is then marked at the
    first listing that reads it. Asked for even when the listing found no
    id, and wherever artifacts have location rows, as
    request_turn_end_self_heal is. The WARNING's `reason` is
    `artifacts_folder_unreadable` when the listing could not read the root
    itself, so it read no folder below it, and otherwise
    `folders_unreadable`, with the `count` of folders it could not read.
    Makes no file store call."""
    if not (listing.unreadable or listing.base_unreadable):
        return
    heal = _project_self_heal(artifacts_base=artifacts_base, scope=scope, project_id=project_id)
    if heal is None:
        return
    unread = (
        {"reason": "artifacts_folder_unreadable"}
        if listing.base_unreadable
        else {"reason": "folders_unreadable", "count": len(listing.unreadable)}
    )
    logger.warning(
        "a full listing of a project's artifacts could not read every folder, so the project is listed again",
        extra={"event": "artifact_location_listing_unreadable", **log_context(project_id=project_id), **unread},
    )
    request_location_check(heal)


def request_self_heal_for_untimed_folders(
    *, folders: Collection[str], artifacts_base: Path, scope: TenantScope | None, project_id: str | None,
) -> None:
    """After a read of the sort times of folders in a project's shared root,
    a turn end's or the self-heal's (artifact_sort_times.sort_times_below),
    that could not read some of them, such as with EIO: log one WARNING, with
    the `count` of those folders, and ask for the project's location check
    alone (request_location_check). Its listing reads their times again and
    writes them (_record_listing), and until then their rows keep the time
    they had. Asked for wherever artifacts have location rows, as
    request_self_heal_for_unread_folders is. Makes no file store call."""
    if not folders:
        return
    heal = _project_self_heal(artifacts_base=artifacts_base, scope=scope, project_id=project_id)
    if heal is None:
        return
    logger.warning(
        "a read of artifact folders' sort times could not read every folder, so the project is listed again",
        extra={"event": "artifact_sort_times_unreadable", **log_context(project_id=project_id), "count": len(folders)},
    )
    request_location_check(heal)


def _project_self_heal(*, artifacts_base: Path, scope: TenantScope | None, project_id: str | None) -> SelfHeal | None:
    """The self-heal of the project's shared root at `artifacts_base` for
    `scope`, or None where its artifacts have no location rows
    (artifact_locations.keeps_locations) or no project is named."""
    from cowork.services.artifact_locations import keeps_locations

    if scope is None or not keeps_locations(scope) or not project_id:
        return None
    return SelfHeal(artifacts_base=Path(artifacts_base), scope=scope, project_id=project_id)


def _listed_artifact_ids(*, artifacts_base: Path) -> ListedIds:
    """The id each artifact folder directly below `artifacts_base` holds, and
    the folders whose id could not be read, from the same listing the
    reconciler's full pass makes (artifact_folders_below)."""
    from cowork.services.artifact_identity import artifact_folders_below, listed_folder_ids

    base = Path(artifacts_base)
    return listed_folder_ids(base=base, listing=artifact_folders_below(base=base))


class ListedIdsAndTimes(NamedTuple):
    """What the self-heal's full listing of a project's shared root read
    (_listed_ids_and_times)."""

    #: The id each folder holds, and the folders it could not read
    #: (_listed_artifact_ids).
    listing: ListedIds
    #: The time each folder holding an id sorts by in the artifacts list,
    #: and the folders whose time it could not read
    #: (artifact_sort_times.sort_times_below).
    times: SortTimesRead


def _listed_ids_and_times(*, artifacts_base: Path) -> ListedIdsAndTimes:
    """The listing _listed_artifact_ids makes, and the time each folder
    holding an id sorts by in the artifacts list (artifact_sort_times)."""
    from cowork.services.artifact_sort_times import sort_times_below

    listing = _listed_artifact_ids(artifacts_base=artifacts_base)
    return ListedIdsAndTimes(listing=listing, times=sort_times_below(base=Path(artifacts_base), slugs=listing.ids))


def _still_the_project_root(session, *, project_id: UUID, artifacts_base: Path) -> bool:
    """Whether `artifacts_base` is still `project_id`'s shared root, read
    from the project's row. A self-heal names the root its project had when
    it was asked for, and the project's path follows its name: a rename
    moves the project's folder (projects.ProjectService._stage_project_rename),
    so the old path is gone, or holds another project's folders once a
    project takes the old name. False too for a project the scope no longer
    sees, such as one deleted since. Makes no file store call."""
    from cowork.models.project import Project
    from cowork.services.artifact_roots import project_artifacts_base

    project = session.get(Project, project_id)
    return project is not None and project_artifacts_base(project.path) == Path(artifacts_base)


def _rows_before_the_listing(
    session, *, project_id: UUID, artifacts_base: Path
) -> dict[str, SortTimeRow] | None:
    """The sort-time rows of `project_id`'s shared root
    (artifact_sort_times.sort_time_rows), which the self-heal reads before it
    lists the root, or None when `artifacts_base` is no longer that root
    (_still_the_project_root), so the self-heal lists nothing."""
    from cowork.services.artifact_sort_times import sort_time_rows

    if not _still_the_project_root(session, project_id=project_id, artifacts_base=artifacts_base):
        return None
    return sort_time_rows(session, project_id=project_id)


class ListingNotRecorded(Exception):
    """One or more of _record_listing's writes failed. Each logged its own
    WARNING and was rolled back, and the others ran."""


def _record_listing(
    session,
    *,
    project_id: UUID,
    artifacts_base: Path,
    listing: ListedIds,
    times: SortTimesRead,
    droppable: Mapping[str, SortTimeRow],
) -> None:
    """Record what a full listing of `project_id`'s shared root at
    `artifacts_base` found: that the root holds a folder, when it does, so
    the artifacts list opens it (held_artifact_roots); each listed folder's
    sort time from `times`: a newer one always, and an older one when the
    row is still the one `droppable` holds, the rows read before the
    listing began (_rows_before_the_listing); the end of each row in
    `droppable` whose folder the listing found gone or holding no id
    (reconcile_sort_times); and each listed id's location
    (artifact_locations.record_listed_folders).

    It records nothing when `artifacts_base` is no longer the project's
    shared root (_still_the_project_root): a rename that committed while the
    listing ran moved the project's folders away from the path it listed,
    which may hold another project's folders by now. A folder whose id or
    time the listing could not read keeps its row, and so does every row
    when the listing could not read the root or found it gone, as it does
    when a rename moves the folder before that rename commits.

    Each of the three writes runs on its own, and one that fails is rolled
    back and logged at WARNING, so it never skips the others. Once all have
    run, a failure raises ListingNotRecorded, so the caller asks for the
    next location check (_record_listed_locations)."""
    from cowork.services.artifact_locations import record_listed_folders
    from cowork.services.artifact_sort_times import reconcile_sort_times
    from cowork.services.held_artifact_roots import hold_root

    if not _still_the_project_root(session, project_id=project_id, artifacts_base=artifacts_base):
        return
    failed = False
    if listing.ids or listing.unreadable:
        try:
            hold_root(session, project_id=project_id)
        except Exception:
            failed = True
            session.rollback()
            logger.warning(
                "artifact self-heal could not record that the project's artifacts folder holds an artifact",
                exc_info=True,
                extra=log_context(project_id=project_id),
            )
    try:
        reconcile_sort_times(
            session,
            project_id=project_id,
            listed=times.times,
            # Every folder found holding an id keeps its row, those whose
            # time could not be read among them: the listing timed only
            # those folders (_listed_ids_and_times).
            kept=set(listing.ids) | listing.unreadable,
            droppable=droppable,
            complete=not (listing.base_unreadable or listing.base_missing),
        )
    except Exception:
        failed = True
        session.rollback()
        logger.warning(
            "artifact self-heal could not record the sort times of the project's artifacts",
            exc_info=True,
            extra=log_context(project_id=project_id),
        )
    try:
        record_listed_folders(session, project_id=project_id, listing=listing)
    except Exception:
        failed = True
        session.rollback()
        logger.warning(
            "artifact self-heal could not check the locations of the project's listed artifacts",
            exc_info=True,
            extra=log_context(project_id=project_id),
        )
    if failed:
        raise ListingNotRecorded


async def _record_listed_locations(heal: SelfHeal) -> None:
    """Read the id of every artifact folder in `heal`'s project, and mark an
    id conflicted when its location row names another folder of the project,
    unless the listing read the row's folder and found it gone or holding
    another id, a stale row the id's next lookup moves
    (artifact_locations.record_listed_folders). A turn end with the pod's
    list opens only the turn's own folders, so a copy no turn end claims,
    such as one a shell `cp -r` makes, is found here, at the first
    self-heal after the turn that asked for one whose listing reads its
    folder. The same listing holds the root and writes each folder's sort
    time, drops the sort-time rows of folders gone from it or holding no
    id, and keeps those of folders it could not read (_record_listing).
    It reads those rows in a database unit of its own before it lists, so
    a row a turn end or a task move writes while the listing runs is never
    dropped. A heal whose path is no longer the
    project's shared root, as after a rename, lists and records nothing and
    asks for no further check: that unit reads the project's path first,
    and the record reads it again (_still_the_project_root). A listing that
    could not read a folder, the root, or a folder's time asks for the next
    location check (request_self_heal_for_unread_folders,
    request_self_heal_for_untimed_folders), and so does a check that fails
    outright or a write of it that fails (request_location_check). Never
    raises: the publish after it runs either way."""
    from cowork.db.units import run_db
    from cowork.services.artifact_locations import keeps_locations

    if not keeps_locations(heal.scope):
        return
    try:
        project_id = UUID(heal.project_id)
        droppable = await run_db(
            partial(_rows_before_the_listing, project_id=project_id, artifacts_base=heal.artifacts_base),
            scope=heal.scope,
        )
        if droppable is None:
            return
        listed = await run_efs(
            partial(_listed_ids_and_times, artifacts_base=heal.artifacts_base), scope=heal.scope, deadline=False,
        )
        # Asked for before the record, so a failed record cannot drop them.
        request_self_heal_for_unread_folders(
            listing=listed.listing, artifacts_base=heal.artifacts_base, scope=heal.scope, project_id=heal.project_id,
        )
        request_self_heal_for_untimed_folders(
            folders=listed.times.unreadable, artifacts_base=heal.artifacts_base, scope=heal.scope,
            project_id=heal.project_id,
        )
        await run_db(
            partial(
                _record_listing, project_id=project_id, artifacts_base=heal.artifacts_base, listing=listed.listing,
                times=listed.times, droppable=droppable,
            ),
            scope=heal.scope,
        )
    except ListingNotRecorded:
        # Each write that failed logged its own line, and the next location
        # check writes it again.
        request_location_check(heal)
    except Exception:
        logger.warning(
            "artifact self-heal could not check the locations of the project's artifacts", exc_info=True,
            extra=log_context(project_id=heal.project_id),
        )
        # Nothing checked the project's locations, so the next poll after
        # the interval does; the publish, if asked for, runs after this
        # either way.
        request_location_check(heal)


async def _run_self_heal(heal: SelfHeal) -> None:
    """The location check, which reads no autopublish setting, then, for a
    scope that names a user, the publish, which returns at once when the
    organization's autopublish is off. A channel turn's scope names no user,
    and the reconciler would only record that refusal, so its self-heal
    checks the locations alone."""
    await _record_listed_locations(heal)
    if heal.scope.user_id is None:
        return
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
    finished: the whole self-heal when its publish was asked for
    (request_self_heal), and otherwise its location check alone
    (request_location_check). The scheduler's poll loop calls this once per
    poll (cowork/scheduler.py). Returns the tasks it started."""
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
        publish = heal in _self_heal_publishes_requested
        _self_heal_publishes_requested.discard(heal)
        _self_heals_started_at[heal] = now
        task = asyncio.create_task(_run_self_heal(heal) if publish else _record_listed_locations(heal))
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
