"""Compaction of cloud memory, run by the server after a turn that saved memory.

The pod never compacts: it builds a new anton session per turn, so the
desktop's "every 10 turns" counter never gets past one, and it holds memory
read only. Here a tier is due once it holds more than anton's threshold of
rules + lessons and has grown by `_GROWTH_BEFORE_NEXT` since the last
attempt, which Redis remembers so every replica agrees.

Keys:
    cowork:memory-compaction:count:{org}:{kind}:{owner}   entry count after the last finished attempt
    cowork:memory-compaction:lock:{org}:{kind}:{owner}    a token while one replica compacts the tier
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from anton.core.llm.client import LLMClient
from anton.core.llm.openai import OpenAIProvider
from anton.core.memory.cortex import (
    COMPACTION_MIN_ENTRIES,
    COMPACTION_THRESHOLD,
    CompactionInput,
    CompactionPlan,
    plan_compaction,
    read_compaction_input,
)
from anton.core.memory.hippocampus import Hippocampus
from redis.exceptions import WatchError

from cowork.common.settings.app_settings import MINDS_FREE_MODEL, TurnQueueSettings
from cowork.common.settings.user_settings import UserSettings, get_user_settings
from cowork.db.scoped import TenantScope
from cowork.db.units import run_db, run_to_completion
from cowork.harnesses.memory.registry import SLOT_REGISTRY, MemorySlot
from cowork.harnesses.memory.store import GlobalMemoryStore
from cowork.principal import Principal
from cowork.services.memory import apply_project_compaction, project_slots_to_compact
from cowork.turnqueue.producer import _mint_llm_block
from cowork.turnqueue.redis_client import get_redis

logger = logging.getLogger(__name__)

_GROWTH_BEFORE_NEXT = 10

# Long enough to outlive a quiet spell, short enough that a tier nobody
# writes any more does not keep its key forever.
_COUNT_TTL_SECONDS = 90 * 24 * 3600

# The whole compaction (key mint and model calls) must end before the turn
# key it runs on expires, with the margin scratchpad-controller leaves a turn.
_TURN_KEY_MARGIN_SECONDS = 120
# The lock outlives the deadline, so it never frees mid-run.
_LOCK_MARGIN_SECONDS = 300

# anton's names for the two files it compacts, lessons first as on the desktop.
_FILE_KINDS = ((MemorySlot.LESSONS, "lesson"), (MemorySlot.RULES, "rules"))

# Held so a running compaction is not garbage collected; cancelled at shutdown,
# which waits for a write already under way.
_compaction_tasks: set[asyncio.Task[None]] = set()


@dataclass(frozen=True)
class MemoryTier:
    """One compactable tier: a person's own memory, or a project's."""

    org_id: str | None
    kind: Literal["person", "project"]
    owner_id: str

    def _key(self, what: str) -> str:
        return f"cowork:memory-compaction:{what}:{self.org_id or '-'}:{self.kind}:{self.owner_id}"

    def __str__(self) -> str:
        return f"{self.kind} org={self.org_id or '-'} owner={self.owner_id}"


def compaction_deadline_seconds(settings: TurnQueueSettings) -> float:
    """How long one attempt may run on the key it mints."""
    return settings.turn_key_ttl_seconds - _TURN_KEY_MARGIN_SECONDS


async def is_due(tier: MemoryTier, entry_count: int) -> bool:
    """Whether `tier`, now holding `entry_count` rules + lessons, should be compacted."""
    client = get_redis()
    key = tier._key("count")
    stored = await client.get(key)
    last = int(stored) if stored is not None else 0
    if last > entry_count:
        # Pruned by hand since the last attempt: count growth from here.
        await client.set(key, entry_count, ex=_COUNT_TTL_SECONDS)
        last = entry_count
    return entry_count > COMPACTION_THRESHOLD and entry_count >= last + _GROWTH_BEFORE_NEXT


async def record_attempt(tier: MemoryTier, entry_count: int) -> None:
    """Remember the count after an attempt, succeeded or failed, so a tier that
    keeps failing costs one model call per `_GROWTH_BEFORE_NEXT` new entries."""
    await get_redis().set(tier._key("count"), entry_count, ex=_COUNT_TTL_SECONDS)


@asynccontextmanager
async def tier_lock(tier: MemoryTier) -> AsyncIterator[bool]:
    """Yield whether this caller holds the tier's compaction lock.

    False means another replica is compacting it now. The release deletes the
    key only while it still holds this attempt's token, so an attempt that
    outran the TTL cannot free the lock of the next one.
    """
    client = get_redis()
    key = tier._key("lock")
    token = uuid4().hex
    ttl = int(compaction_deadline_seconds(TurnQueueSettings())) + _LOCK_MARGIN_SECONDS
    acquired = bool(await client.set(key, token, nx=True, ex=ttl))
    try:
        yield acquired
    finally:
        if acquired:
            await _release(key, token)


async def _release(key: str, token: str) -> None:
    # WATCH/MULTI rather than a Lua compare-and-delete: fakeredis, which the
    # tests run on, has no EVAL. Never raises: it runs in a `finally`.
    try:
        async with get_redis().pipeline(transaction=True) as pipe:
            await pipe.watch(key)
            if await pipe.get(key) != token:
                await pipe.reset()
                return
            pipe.multi()
            pipe.delete(key)
            await pipe.execute()
    except WatchError:
        return
    except Exception as exc:
        # The TTL frees the lock anyway.
        logger.warning("memory-compaction: lock release failed (%s)", type(exc).__name__)


def start_compaction(
    *,
    scope: TenantScope,
    principal: Principal | None,
    turn_correlation_id: str,
    project_id: UUID | None,
    personal_entries: int,
    project_entries: int,
) -> None:
    """Compact, in the background, the tiers a turn just wrote to that are due.

    Never awaited by the caller: the reply does not wait for the model call.
    `turn_correlation_id` is the id of the turn that wrote, so the logs and
    the key minted for the model call lead back to it.
    """
    task = asyncio.create_task(
        _compact_after_turn(
            scope=scope,
            principal=principal,
            turn_correlation_id=turn_correlation_id,
            project_id=project_id,
            personal_entries=personal_entries,
            project_entries=project_entries,
        )
    )
    _compaction_tasks.add(task)
    task.add_done_callback(_compaction_tasks.discard)


async def cancel_compactions() -> None:
    """Cancel every running compaction, for shutdown."""
    tasks = list(_compaction_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _compact_after_turn(
    *,
    scope: TenantScope,
    principal: Principal | None,
    turn_correlation_id: str,
    project_id: UUID | None,
    personal_entries: int,
    project_entries: int,
) -> None:
    # Each tier on its own, so a failure in one leaves the other its attempt.
    # Type name only: a provider or validation error can quote memory entries.
    if personal_entries and scope.user_id:
        person = MemoryTier(org_id=scope.org_id, kind="person", owner_id=scope.user_id)
        try:
            await _compact_tier(
                person,
                scope=scope,
                turn_correlation_id=turn_correlation_id,
                target=(GlobalMemoryStore(scope=scope).root, [s for s, _ in _FILE_KINDS]),
            )
        except Exception as exc:
            logger.warning(
                "memory-compaction: %s stopped (%s), turn %s", person, type(exc).__name__, turn_correlation_id
            )
    if project_entries and project_id is not None:
        project = MemoryTier(org_id=scope.org_id, kind="project", owner_id=str(project_id))
        try:
            target = await run_db(
                partial(project_slots_to_compact, principal=principal, project_id=project_id),
                scope=scope,
            )
            if target is not None and target[1]:
                await _compact_tier(
                    project,
                    scope=scope,
                    turn_correlation_id=turn_correlation_id,
                    target=target,
                    principal=principal,
                    project_id=project_id,
                )
        except Exception as exc:
            logger.warning(
                "memory-compaction: %s stopped (%s), turn %s", project, type(exc).__name__, turn_correlation_id
            )


async def _compact_tier(
    tier: MemoryTier,
    *,
    scope: TenantScope,
    turn_correlation_id: str,
    target: tuple[Path, list[MemorySlot]],
    principal: Principal | None = None,
    project_id: UUID | None = None,
) -> None:
    root, slots = target
    turn_settings = TurnQueueSettings()
    settings = await asyncio.to_thread(get_user_settings, scope)
    if not settings.memory_enabled or settings.memory_mode == "off":
        return
    if not await is_due(tier, await asyncio.to_thread(_entry_count, root)):
        return
    async with tier_lock(tier) as held:
        # Asked again under the lock: another replica may have just finished.
        if not held or not await is_due(tier, await asyncio.to_thread(_entry_count, root)):
            return
        counted = True
        try:
            plans = await asyncio.wait_for(
                _plan_files(scope, settings, turn_settings, turn_correlation_id, root, slots),
                timeout=compaction_deadline_seconds(turn_settings),
            )
            if plans:
                counted = await _apply(tier, scope, root, plans, principal=principal, project_id=project_id)
        except Exception as exc:
            logger.warning(
                "memory-compaction: %s attempt failed (%s), turn %s", tier, type(exc).__name__, turn_correlation_id
            )
        # Not reached on a cancel: a shutdown mid-call does not use up the attempt.
        if counted:
            await record_attempt(tier, await asyncio.to_thread(_entry_count, root))


def _entry_count(root: Path) -> int:
    return Hippocampus(root).entry_count()


async def _plan_files(
    scope: TenantScope,
    settings: UserSettings,
    turn_settings: TurnQueueSettings,
    turn_correlation_id: str,
    root: Path,
    slots: list[MemorySlot],
) -> list[tuple[MemorySlot, CompactionPlan]]:
    sources: list[tuple[MemorySlot, CompactionInput]] = []
    for slot, kind in _FILE_KINDS:
        if slot not in slots:
            continue
        source = await asyncio.to_thread(
            read_compaction_input, root / SLOT_REGISTRY[slot].filename, kind
        )
        if len(source.entries) >= COMPACTION_MIN_ENTRIES:
            sources.append((slot, source))
    if not sources:
        return []
    # The same key and model a cloud turn of this person gets: minted for their
    # user, org and workspace, so the call is billed where their turns are.
    # Its own id, led by the turn's: auth hands out one key per id, and a key
    # under the turn's own id is revoked with the turn.
    block = await _mint_llm_block(
        org_id=scope.org_id,
        user_id=scope.user_id,
        correlation_id=f"{turn_correlation_id}:compact-{uuid4().hex[:8]}",
        settings=turn_settings,
        workspace_id=settings.hub_workspace_id or None,
    )
    provider = OpenAIProvider(
        api_key=block["api_key"],
        base_url=block["base_url"],
        flavor=OpenAIProvider.FLAVOR_MINDS_PASSTHROUGH,
    )
    model = block.get("coding_model") or MINDS_FREE_MODEL
    llm = LLMClient(
        planning_provider=provider, planning_model=model, coding_provider=provider, coding_model=model
    )
    try:
        plans: list[tuple[MemorySlot, CompactionPlan]] = []
        for slot, source in sources:
            plan = await plan_compaction(llm, source)
            if plan is not None:
                plans.append((slot, plan))
        return plans
    finally:
        await llm.aclose()


async def _apply(
    tier: MemoryTier,
    scope: TenantScope,
    root: Path,
    plans: list[tuple[MemorySlot, CompactionPlan]],
    *,
    principal: Principal | None,
    project_id: UUID | None,
) -> bool:
    """Write the plans; returns whether any file was rewritten.

    An attempt that wrote nothing, because every file changed under it or its
    author was refused, is not counted, so the next save tries again. A cancel
    waits for a write already under way, so neither shutdown nor the lock
    release runs while it still writes.
    """
    if tier.kind == "person":
        return await run_to_completion(asyncio.to_thread(_apply_personal, root, plans))
    return await run_db(
        partial(
            apply_project_compaction,
            principal=principal,
            project_id=project_id,
            planned_root=root,
            plans=plans,
        ),
        scope=scope,
    )


def _apply_personal(root: Path, plans: list[tuple[MemorySlot, CompactionPlan]]) -> bool:
    # A reset that removed the folder during the model call wins: writing now
    # would bring the folder back. One that lands after this check gets back
    # an empty folder holding a lock file; no memory comes back.
    if not root.is_dir():
        return False
    hippocampus = Hippocampus(root)
    # A list, not a generator: every plan is applied, not just the first that writes.
    written = [plan.apply(hippocampus) for _slot, plan in plans]
    return any(written)
