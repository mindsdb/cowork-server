"""Redis-backed per-(channel_type, org_id) lease so exactly one cowork-server
replica runs a given org's streaming ingress (e.g. the Discord Gateway) at a
time. Uses WATCH/MULTI/EXEC for compare-and-swap rather than Lua (EVAL) — the
fakeredis version this repo tests against implements neither EVAL nor
EVALSHA."""
from __future__ import annotations

import logging

from cowork.turnqueue.redis_client import get_redis

log = logging.getLogger(__name__)

LEASE_TTL_S = 45.0
RENEW_INTERVAL_S = LEASE_TTL_S / 3


def _key(channel_type: str, org_id: str) -> str:
    return f"ingress-lease:{channel_type}:{org_id}"


async def acquire(channel_type: str, org_id: str, owner: str) -> bool:
    """True if `owner` now holds the lease. Fails closed: any Redis error is
    treated as "not acquired", never as "acquired"."""
    client = get_redis()
    try:
        ok = await client.set(_key(channel_type, org_id), owner, nx=True, px=int(LEASE_TTL_S * 1000))
    except Exception:
        log.warning("ingress lease acquire failed for %s org %s", channel_type, org_id, exc_info=True)
        return False
    return bool(ok)


async def renew(channel_type: str, org_id: str, owner: str) -> bool:
    """Extend the TTL only if `owner` still holds the lease. False means the
    lease is gone — expired, released, or held by someone else — and the
    caller must stop treating itself as the owner."""
    client = get_redis()
    key = _key(channel_type, org_id)
    try:
        async with client.pipeline(transaction=True) as pipe:
            await pipe.watch(key)
            current = await pipe.get(key)
            if current != owner:
                await pipe.reset()
                return False
            pipe.multi()
            pipe.pexpire(key, int(LEASE_TTL_S * 1000))
            await pipe.execute()
            return True
    except Exception:
        log.warning("ingress lease renew failed for %s org %s", channel_type, org_id, exc_info=True)
        return False


async def release(channel_type: str, org_id: str, owner: str) -> None:
    """Delete the lease only if `owner` still holds it, so a stale release
    from an owner that already lost the lease can't delete someone else's."""
    client = get_redis()
    key = _key(channel_type, org_id)
    try:
        async with client.pipeline(transaction=True) as pipe:
            await pipe.watch(key)
            current = await pipe.get(key)
            if current != owner:
                await pipe.reset()
                return
            pipe.multi()
            pipe.delete(key)
            await pipe.execute()
    except Exception:
        log.warning("ingress lease release failed for %s org %s", channel_type, org_id, exc_info=True)
