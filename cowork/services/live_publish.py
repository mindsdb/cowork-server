"""Publishing a doc's page for the org republish queue (cowork.docs.publish_queue).

The queue says which doc changed; this decides whether its page must go out
again and sends it. Everything that can refuse runs before a key is minted:
the record from Redis must name a doc folder in its project's root, the doc's
owner must be known, and a page whose md5 matches the one that went out is
left alone. The upload runs on the default executor; ``on_done`` fires once
everything the publish held is let go, which after a timeout is when the
upload thread ends, not when the wait for it did.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

from cowork.db.scoped import TenantScope
from cowork.db.units import FilesystemBusy, run_db, run_efs, run_on_unit_threads
from cowork.docs.hub import DOC_PUBLISHED, doc_topic, hub
from cowork.docs.store import DOC_HTML, _has_id, is_doc_folder, load_doc
from cowork.services.artifact_access import ArtifactAccessRefused, ArtifactAudienceRejected
from cowork.services.artifact_autopublish import (
    AUTOPUBLISH_ACCESS,
    _active_workspace_id,
    _artifact_owners,
    _publish_url,
    _release,
    job_budget_for,
    lock_ttl_for,
)
from cowork.services.artifact_locks import acquire, refresh
from cowork.services.artifact_publish_key import PublishKey
from cowork.services.artifacts import load_published_map
from cowork.services.product_permissions import ProductPermissionDenied
from cowork.services.publish import (
    AccessModeChanged,
    ComparisonPublishRefused,
    MarkdownNotPublishable,
    PublishOwnedByOtherUser,
    PublishTooLarge,
    ShareOrgRequired,
    compute_publish_md5,
    kept_access,
    publish_artifact,
)

logger = logging.getLogger(__name__)

PUBLISH_TIMEOUT_S = 60.0
LOCK_TTL_S = lock_ttl_for(PUBLISH_TIMEOUT_S)
# A publish lock someone else holds is a retry, not a wait: a wait would keep a slot.
LOCK_WAIT_S = 1.0
_LOCK_POLL_S = 0.1

# AccessModeChanged and ShareOrgRequired subclass PublishRejected, which is
# otherwise a network or job failure worth retrying, so they are named here.
# A bare ValueError is not: a partial read of a file can raise one.
_PERMANENT: tuple[type[BaseException], ...] = (
    AccessModeChanged,
    ShareOrgRequired,
    PublishTooLarge,
    PublishOwnedByOtherUser,
    ProductPermissionDenied,
    ArtifactAccessRefused,
    ArtifactAudienceRejected,
    ComparisonPublishRefused,
    MarkdownNotPublishable,
)

# Lock releases and key revokes a finished upload thread asked for; awaited at shutdown.
_cleanups: set[asyncio.Task] = set()


class Outcome(str, Enum):
    PUBLISHED = "published"
    UNCHANGED = "unchanged"
    SKIPPED = "skipped"
    RETRY = "retry"
    DROPPED = "dropped"


@dataclass(frozen=True)
class PublishResult:
    outcome: Outcome
    reason: str | None = None


@dataclass(frozen=True)
class DocPublishJob:
    artifact_id: str
    project_id: str
    artifacts_base: str
    slug: str
    org_id: str


@dataclass(frozen=True)
class _PageState:
    kind: str  # "first" | "changed" | "unchanged" | "off"
    entry: dict[str, Any] | None


def is_permanent(exc: BaseException) -> bool:
    """True for a refusal that will repeat until something other than time changes."""
    return isinstance(exc, _PERMANENT)


def _valid_slug(slug: str) -> bool:
    return bool(slug) and not slug.startswith(".") and "/" not in slug and "\\" not in slug


def track(task: asyncio.Task, registry: set[asyncio.Task]) -> asyncio.Task:
    """Keep ``task`` in ``registry`` until it ends, so a shutdown can wait for it."""
    registry.add(task)
    task.add_done_callback(registry.discard)
    return task


async def drain(tasks: Iterable[asyncio.Task], *, timeout: float, outcome: str) -> None:
    """Wait up to ``timeout`` for ``tasks``; log ``outcome`` with how many there were when it runs out."""
    tasks = list(tasks)
    if not tasks or timeout <= 0:
        return
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("doc_publish", extra={"event": "doc_publish", "outcome": outcome, "count": len(tasks)})


def _doc_folder(job: DocPublishJob) -> tuple[Path | None, str | None]:
    """The doc folder the record names, or (None, why not). Read only once
    the slug is valid and _owner_of has confirmed the base is the project's root."""
    folder = Path(job.artifacts_base) / job.slug
    if not folder.is_dir() or not is_doc_folder(folder):
        return None, "not_a_doc"
    if not _has_id(load_doc(folder), job.artifact_id.replace("-", "").lower()):
        return None, "id_mismatch"
    return folder, None


async def _owner_of(job: DocPublishJob, scope: TenantScope) -> str | None:
    """The doc's recorded owner; None when unknown or when the base is not the project's root."""
    try:
        resolutions = await run_db(
            partial(_artifact_owners, artifacts_base=Path(job.artifacts_base), project_id=job.project_id,
                    slugs=[job.slug]),
            scope=scope,
        )
    except ValueError:  # a deleted project or a project id that is not a UUID
        return None
    if resolutions is None:
        return None
    resolution = resolutions[job.slug]
    if resolution.unknown or not resolution.owner_user_id:
        return None
    return str(resolution.owner_user_id)


def _page_state(folder: Path) -> _PageState:
    entry = load_published_map(folder).get(DOC_HTML)
    if not isinstance(entry, dict) or not entry.get("report_id"):
        return _PageState("first", None)
    if entry.get("published") is False:
        return _PageState("off", entry)
    md5 = compute_publish_md5(folder / DOC_HTML, artifacts_base=folder.parent)
    if md5 is not None and md5 == entry.get("last_md5"):
        return _PageState("unchanged", entry)
    return _PageState("changed", entry)


def _early(state: _PageState) -> PublishResult | None:
    if state.kind == "off":
        return PublishResult(Outcome.SKIPPED, "unpublished")
    if state.kind == "unchanged":
        return PublishResult(Outcome.UNCHANGED)
    return None


async def _try_lock(base: Path, slug: str, scope: TenantScope) -> bool:
    """One take of the publish lock. A cancel that lands once the take ran
    frees a lock it took, as artifact_autopublish._acquire does: the caller
    never sees that True."""
    taken = False

    def take() -> bool:
        nonlocal taken
        taken = acquire(base, slug, ttl_s=LOCK_TTL_S)
        return taken

    try:
        return await run_efs(take, scope=scope)
    except asyncio.CancelledError:
        if taken:
            await _release(base, slug, scope=scope)
        raise


async def _take_lock(base: Path, slug: str, scope: TenantScope) -> bool:
    deadline = time.monotonic() + LOCK_WAIT_S
    while True:
        try:
            if await _try_lock(base, slug, scope):
                return True
        except FilesystemBusy:
            pass  # a busy file store is a lock not taken this round
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(_LOCK_POLL_S)


async def _let_go(base: Path, slug: str, scope: TenantScope, *, lock: bool, key: PublishKey | None,
                  on_done: Callable[[], None] | None = None) -> None:
    try:
        if lock:
            try:
                await _release(base, slug, scope=scope)
            except Exception:  # noqa: BLE001 - the lock expires by its TTL
                logger.warning("doc_publish", exc_info=True, extra={
                    "event": "doc_publish", "outcome": "lock_release_failed", "artifact_slug": slug})
        if key is not None:
            await key.revoke()
    finally:
        if on_done is not None:
            on_done()


async def publish_doc_page(job: DocPublishJob, *, on_done: Callable[[], None]) -> PublishResult:
    """Publish ``job``'s doc page when it changed since it last went out.

    Calls ``on_done`` exactly once, after the lock and key are let go: at once
    on a path that starts no upload, else when the upload thread ends. Errors
    from the publish itself reach the caller (is_permanent tells them apart).
    """
    base = Path(job.artifacts_base)
    scope = TenantScope(org_mode=True, org_id=job.org_id)
    lock = False
    key: PublishKey | None = None
    handed_off = False
    try:
        if not _valid_slug(job.slug):
            return PublishResult(Outcome.SKIPPED, "bad_slug")
        # The base comes from Redis: no file under it is read before it proves to be the project's root.
        owner = await _owner_of(job, scope)
        if owner is None:
            return PublishResult(Outcome.SKIPPED, "owner_unknown")
        folder, why_not = await run_efs(partial(_doc_folder, job), scope=scope)
        if folder is None:
            return PublishResult(Outcome.SKIPPED, why_not)
        scope = TenantScope(org_mode=True, org_id=job.org_id, user_id=owner)
        state = await run_efs(partial(_page_state, folder), scope=scope)
        if (early := _early(state)) is not None:
            return early
        if not await _take_lock(base, job.slug, scope):
            return PublishResult(Outcome.RETRY, "lock_busy")
        lock = True
        # Another publish may have sent this page while the lock was taken.
        state = await run_efs(partial(_page_state, folder), scope=scope)
        if (early := _early(state)) is not None:
            return early
        access = AUTOPUBLISH_ACCESS if state.entry is None else kept_access(state.entry, org_mode=True)
        publish_url = await run_on_unit_threads(partial(_publish_url, scope))
        workspace_id = await run_on_unit_threads(partial(_active_workspace_id, scope))
        key = PublishKey(owner, job.org_id, min_ttl_s=LOCK_TTL_S + 60.0, workspace_id=workspace_id)
        api_key = await key.get()
        if not api_key:
            return PublishResult(Outcome.RETRY, "no_key")
        upload = partial(
            publish_artifact, folder, artifacts_base=base, api_key=api_key, publish_url=publish_url,
            access=access, scope=scope, project_id=job.project_id, background=True,
            job_budget_s=job_budget_for(LOCK_TTL_S),
            on_job_accepted=lambda _job: refresh(base, job.slug),
        )
        # run_in_executor does not carry contextvars over the way to_thread does.
        context = contextvars.copy_context()
        future = asyncio.get_running_loop().run_in_executor(None, context.run, upload)
        handed_off = True
        # Set once nobody awaits the upload any more; read and written on the loop thread only.
        abandoned = False

        def done(finished: asyncio.Future) -> None:
            # Runs when the thread ends, so a page that went out after the wait
            # timed out still reaches the editors.
            try:
                if not finished.cancelled():
                    error = finished.exception()
                    if error is None:
                        hub.publish(doc_topic(folder), {"type": DOC_PUBLISHED, "rev": 0})
                    elif abandoned:
                        logger.warning("doc_publish", extra={
                            "event": "doc_publish", "outcome": "late_failure", "artifact_slug": job.slug,
                            "error_type": type(error).__name__})
            finally:
                track(asyncio.get_running_loop().create_task(
                    _let_go(base, job.slug, scope, lock=True, key=key, on_done=on_done)), _cleanups)

        future.add_done_callback(done)
        # shield: a timeout or a cancel must not cancel the future, or its
        # callback would fire while the upload thread still runs.
        try:
            await asyncio.wait_for(asyncio.shield(future), timeout=PUBLISH_TIMEOUT_S)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            abandoned = True
            raise
        return PublishResult(Outcome.PUBLISHED)
    finally:
        if not handed_off:
            await _let_go(base, job.slug, scope, lock=lock, key=key, on_done=on_done)


async def drain_cleanup(*, timeout: float) -> None:
    """Wait for the lock releases and key revokes finished uploads asked for (shutdown)."""
    await drain(_cleanups, timeout=timeout, outcome="cleanup_unfinished")
