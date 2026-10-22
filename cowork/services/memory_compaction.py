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
    COMPACTION_THRESHOLD,
    CompactionPlan,
    plan_compaction,
    read_compaction_input,
)
from anton.core.memory.hippocampus import Hippocampus
from redis.exceptions import WatchError

from cowork.common.settings.app_settings import MINDS_FREE_MODEL, TurnQueueSettings
from cowork.common.settings.user_settings import UserSettings, get_user_settings
from cowork.db.scoped import TenantScope
from cowork.db.units import run_db
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
# key it runs on expires; the lock outlives it, so it never frees mid-run.
COMPACTION_DEADLINE_SECONDS = 600
_LOCK_TTL_SECONDS = COMPACTION_DEADLINE_SECONDS + 300

# anton's names for the two files it compacts, lessons first as on the desktop.
_FILE_KINDS = ((MemorySlot.LESSONS, "lesson"), (MemorySlot.RULES, "rules"))

# Held so a running compaction is not garbage collected; cancelled at shutdown,
# which is safe because nothing is written before a plan is applied.
_compaction_tasks: set[asyncio.Task[None]] = set()


@dataclass(frozen=True)
class MemoryTier:
    """One compactable tier: a person's own memory, or a project's."""

    org_id: str | None
    kind: Literal["person", "project"]
    owner_id: str

    def _key(self, what: str) -> str:
        return f"cowork:memory-compaction:{what}:{self.org_id or '-'}:{self.kind}:{self.owner_id}"


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
    acquired = bool(await client.set(key, token, nx=True, ex=_LOCK_TTL_SECONDS))
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
    project_id: UUID | None,
    personal_applied: int,
    project_applied: int,
) -> None:
    """Compact, in the background, the tiers a turn just wrote to that are due.

    Never awaited by the caller: the reply does not wait for the model call.
    """
    task = asyncio.create_task(
        _compact_after_turn(
            scope=scope,
            principal=principal,
            project_id=project_id,
            personal_applied=personal_applied,
            project_applied=project_applied,
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
    project_id: UUID | None,
    personal_applied: int,
    project_applied: int,
) -> None:
    try:
        if personal_applied and scope.user_id:
            person = MemoryTier(org_id=scope.org_id, kind="person", owner_id=scope.user_id)
            await _compact_tier(
                person, scope=scope, target=(GlobalMemoryStore(scope=scope).root, [s for s, _ in _FILE_KINDS])
            )
        if project_applied and project_id is not None:
            target = await run_db(
                partial(project_slots_to_compact, principal=principal, project_id=project_id),
                scope=scope,
            )
            if target is not None and target[1]:
                project = MemoryTier(org_id=scope.org_id, kind="project", owner_id=str(project_id))
                await _compact_tier(
                    project, scope=scope, target=target, principal=principal, project_id=project_id
                )
    except Exception as exc:
        # Type name only: a provider or validation error can quote memory entries.
        logger.warning("memory-compaction: stopped (%s)", type(exc).__name__)


async def _compact_tier(
    tier: MemoryTier,
    *,
    scope: TenantScope,
    target: tuple[Path, list[MemorySlot]],
    principal: Principal | None = None,
    project_id: UUID | None = None,
) -> None:
    root, slots = target
    settings = await asyncio.to_thread(get_user_settings, scope)
    if not settings.memory_enabled or settings.memory_mode == "off":
        return
    if not await is_due(tier, await asyncio.to_thread(_entry_count, root)):
        return
    async with tier_lock(tier) as held:
        if not held:
            return
        try:
            plans = await asyncio.wait_for(
                _plan_files(scope, settings, root, slots), timeout=COMPACTION_DEADLINE_SECONDS
            )
            if plans:
                await _apply(tier, scope, root, plans, principal=principal, project_id=project_id)
        except Exception as exc:
            logger.warning("memory-compaction: %s tier attempt failed (%s)", tier.kind, type(exc).__name__)
        # Not reached on a cancel: a shutdown mid-call does not use up the attempt.
        await record_attempt(tier, await asyncio.to_thread(_entry_count, root))


def _entry_count(root: Path) -> int:
    return Hippocampus(root).entry_count()


async def _plan_files(
    scope: TenantScope, settings: UserSettings, root: Path, slots: list[MemorySlot]
) -> list[tuple[MemorySlot, CompactionPlan]]:
    # The same key and model a cloud turn of this person gets: minted for their
    # user, org and workspace, so the call is billed where their turns are.
    block = await _mint_llm_block(
        org_id=scope.org_id,
        user_id=scope.user_id,
        correlation_id=str(uuid4()),
        settings=TurnQueueSettings(),
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
    plans: list[tuple[MemorySlot, CompactionPlan]] = []
    for slot, kind in _FILE_KINDS:
        if slot not in slots:
            continue
        source = await asyncio.to_thread(
            read_compaction_input, root / SLOT_REGISTRY[slot].filename, kind
        )
        plan = await plan_compaction(llm, source)
        if plan is not None:
            plans.append((slot, plan))
    return plans


async def _apply(
    tier: MemoryTier,
    scope: TenantScope,
    root: Path,
    plans: list[tuple[MemorySlot, CompactionPlan]],
    *,
    principal: Principal | None,
    project_id: UUID | None,
) -> None:
    # Shielded: a write already handed to its thread finishes either way, and
    # the caller must not move on as if it had not.
    if tier.kind == "person":
        await asyncio.shield(asyncio.to_thread(_apply_personal, root, plans))
        return
    await asyncio.shield(
        run_db(
            partial(
                apply_project_compaction,
                principal=principal,
                project_id=project_id,
                planned_root=root,
                plans=plans,
            ),
            scope=scope,
        )
    )


def _apply_personal(root: Path, plans: list[tuple[MemorySlot, CompactionPlan]]) -> None:
    # A reset that removed the folder during the model call wins: writing
    # now would bring the folder back.
    if not root.is_dir():
        return
    hippocampus = Hippocampus(root)
    for _slot, plan in plans:
        plan.apply(hippocampus)
