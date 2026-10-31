"""MindsHub's public API, asked on behalf of the caller.

What every MindsHub read in this server shares: the transport
(get_mindshub_reply), the per-caller cache the reads keep their answers in
(CallerTtlCache), and the errors a route answers when MindsHub refuses a read
(MindsHubRefused) or can't be asked (MindsHubUnreachable).

The bearer is the caller's own and the host is the operator's
(``default_minds_auth_host``, derived from ENV). The stored provider key and
``minds_url`` are both tenant-settable, so using either would let an org admin
point a member's credential at a host they chose.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import MutableMapping
from enum import StrEnum
from typing import Any, Generic, NamedTuple, Protocol, TypeVar
from uuid import UUID

import httpx
from fastapi import HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Hard TOTAL budget for one call. httpx.Timeout is per-operation, so it alone
# does not bound a redirect chain or a trickled body; the outer wait_for does.
# These reads sit on menu-open and share paths, so a degraded MindsHub must not
# hang the UI.
_TIMEOUT_S = 5.0

# The caller's entitlements, read for their usage (mindshub_usage) and their
# permissions (mindshub_teams).
ENTITLEMENTS_PATH = "/entitlements/me/"


def mindshub_v1() -> str:
    """MindsHub's public ``/v1`` base for this deployment."""
    from cowork.common.settings.app_settings import default_minds_auth_host

    return f"{default_minds_auth_host().rstrip('/')}/v1"


class MindsHubAnswer(StrEnum):
    """How MindsHub answered one read (get_mindshub_reply)."""

    # A 2xx with a JSON body.
    OK = "ok"
    # A 4xx other than 429: MindsHub was asked and said no. Asking again gets
    # the same answer.
    REFUSED = "refused"
    # A 429: MindsHub was asked and is limiting the caller's reads.
    THROTTLED = "throttled"
    # No answer, a 5xx, or a body that is not JSON: MindsHub couldn't be asked.
    UNREACHABLE = "unreachable"


class MindsHubReply(NamedTuple):
    """One MindsHub read's outcome. ``payload`` is the JSON body of an OK
    answer and None otherwise; ``status_code`` is MindsHub's HTTP status, or
    None when nothing answered."""

    answer: MindsHubAnswer
    payload: Any = None
    status_code: int | None = None


async def get_mindshub_reply(path: str, bearer_token: str) -> MindsHubReply:
    """GET one MindsHub path with the caller's bearer, and say how MindsHub
    answered (MindsHubAnswer).

    Never raises and never runs longer than ``_TIMEOUT_S``. A refusal (4xx)
    and a throttle (429) are told apart from an outage, so a caller never
    reports a "no" from MindsHub as "couldn't reach MindsHub".
    """

    async def _fetch() -> httpx.Response:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_TIMEOUT_S),
            follow_redirects=True,
            max_redirects=2,
        ) as client:
            return await client.get(
                f"{mindshub_v1()}{path}",
                headers={"Authorization": f"Bearer {bearer_token}"},
            )

    try:
        response = await asyncio.wait_for(_fetch(), _TIMEOUT_S)
    except Exception as exc:
        logger.debug("MindsHub GET %s failed: %s", path, exc)
        return MindsHubReply(MindsHubAnswer.UNREACHABLE)
    code = response.status_code
    if code == status.HTTP_429_TOO_MANY_REQUESTS:
        logger.debug("MindsHub GET %s returned HTTP 429", path)
        return MindsHubReply(MindsHubAnswer.THROTTLED, status_code=code)
    if 400 <= code < 500:
        logger.debug("MindsHub GET %s returned HTTP %s", path, code)
        return MindsHubReply(MindsHubAnswer.REFUSED, status_code=code)
    if code >= 500:
        logger.debug("MindsHub GET %s returned HTTP %s", path, code)
        return MindsHubReply(MindsHubAnswer.UNREACHABLE, status_code=code)
    try:
        return MindsHubReply(MindsHubAnswer.OK, payload=response.json(), status_code=code)
    except ValueError:
        logger.debug("MindsHub GET %s returned a non-JSON body", path)
        return MindsHubReply(MindsHubAnswer.UNREACHABLE, status_code=code)


async def get_mindshub_json(path: str, bearer_token: str) -> Any | None:
    """GET one MindsHub path with the caller's bearer (get_mindshub_reply):
    the JSON body of an OK answer, None for any other.

    For the reads whose callers turn every failure into one fail-closed
    answer (the workspace listing, usage), so a surface this gates stays dark
    rather than half-rendered.
    """
    reply = await get_mindshub_reply(path, bearer_token)
    return reply.payload if reply.answer is MindsHubAnswer.OK else None


def canonical_uuid(value: object) -> str | None:
    """``value`` as a UUID in its canonical lowercase form, or None when it
    isn't one. MindsHub keys every person and team by UUID and compares them
    in this form."""
    try:
        return str(UUID(str(value).strip()))
    except ValueError:
        return None


# ── Errors a route answers for MindsHub ──────────────────────────────────


class MindsHubErrorCode(StrEnum):
    """The ``code`` of a MindsHub error's body (MindsHubErrorDetail)."""

    # MindsHub couldn't be asked: retryable (MindsHubUnreachable).
    UNREACHABLE = "mindshub_unreachable"
    # MindsHub was asked and said no (MindsHubRefused).
    REFUSED = "mindshub_refused"
    # MindsHub is limiting how often the caller may ask (MindsHubRefused).
    THROTTLED = "mindshub_throttled"


class MindsHubErrorDetail(BaseModel):
    """The ``detail`` of a route's error answer about MindsHub."""

    code: MindsHubErrorCode
    message: str


TEAMS_UNREACHABLE = "Couldn't reach MindsHub to confirm your teams. Try again in a moment."
PERMISSIONS_UNREACHABLE = "Couldn't reach MindsHub to confirm your permissions. Try again in a moment."
PEOPLE_UNREACHABLE_REFUSAL = (
    "Couldn't reach MindsHub to confirm who is in your organization. Try again in a moment."
)


class MindsHubUnreachable(HTTPException):
    """503 ``mindshub_unreachable``: a change needs an answer MindsHub
    couldn't give (the caller's teams, permissions or org members).
    Retryable, and never the 403 that would tell the caller they lack a
    permission they may hold."""

    def __init__(self, message: str) -> None:
        super().__init__(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=MindsHubErrorDetail(code=MindsHubErrorCode.UNREACHABLE, message=message).model_dump(
                mode="json"
            ),
        )


class MindsHubRefused(HTTPException):
    """MindsHub answered a read made for the caller with a 4xx: it was asked
    and said no, so this is never the retryable "couldn't reach MindsHub".
    A 429 answers 429 ``mindshub_throttled``; MindsHub's 401 or 403 answers
    403 ``mindshub_refused`` (never a 401, which the app reads as signed out);
    any other 4xx answers 400 ``mindshub_refused``."""

    def __init__(self, *, mindshub_status: int | None) -> None:
        if mindshub_status == status.HTTP_429_TOO_MANY_REQUESTS:
            code, answer = MindsHubErrorCode.THROTTLED, status.HTTP_429_TOO_MANY_REQUESTS
            message = "MindsHub is limiting how often this can be asked. Try again in a minute."
        elif mindshub_status in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN):
            code, answer = MindsHubErrorCode.REFUSED, status.HTTP_403_FORBIDDEN
            message = "MindsHub refused to answer for your organization."
        else:
            code, answer = MindsHubErrorCode.REFUSED, status.HTTP_400_BAD_REQUEST
            message = "MindsHub refused this lookup."
        self.mindshub_status = mindshub_status
        super().__init__(
            answer, detail=MindsHubErrorDetail(code=code, message=message).model_dump(mode="json")
        )


# ── The per-caller cache ─────────────────────────────────────────────────

# How long a read MindsHub answered is kept, and how long one it couldn't
# answer is: one round trip per minute per caller when MindsHub answers, and
# a quick retry when it didn't, so a degraded MindsHub neither adds a round
# trip to every menu open nor goes stale for a long stretch. A cache passes
# its own pair where its answer must be newer (usage).
MINDSHUB_TTL_OK_S = 60.0
MINDSHUB_TTL_FAIL_S = 15.0


class CallerCacheKey(NamedTuple):
    """One cache entry per credential per caller per organization per
    deployment.

    MindsHub answers its reads per caller, not per organization: an owner or
    admin gets every workspace in the org and a member only the ones they hold
    a grant on, and a person's teams and permissions are theirs. Keyed on the
    organization alone, one admin's answer is served to every member of their
    org for the whole TTL. This process serves every user of every
    organization at once in MindsHub mode, so the caller has to be in the key.

    The credential is in the key as well as the identity. ``user_id`` comes
    from the gateway-set principal, and the answer is whatever the bearer
    answered, so on their own the two can name different people. They always
    do on a desktop install, where the scope is LOCAL_SCOPE and ``user_id`` is
    empty for every request: without the credential the key collapses to one
    shared entry, and after a sign-out and a sign-in as another MindsHub
    account the previous account's answer is served for the rest of the TTL.
    A new session means a new token means a new entry.

    A digest rather than the token: keys are held for the process lifetime
    and turn up in a repr or a heap dump, and nothing here needs to reverse
    it.
    """

    mindshub_base: str
    org_id: str
    user_id: str
    credential_digest: str

    @classmethod
    def for_caller(cls, *, org_id: str, user_id: str, bearer_token: str) -> CallerCacheKey:
        digest = hashlib.sha256(bearer_token.encode("utf-8")).hexdigest()[:32]
        return cls(mindshub_v1(), org_id or "", user_id or "", digest)


class _Answer(Protocol):
    """A cached read's value: ``reachable`` false when MindsHub couldn't be
    asked."""

    reachable: bool


_V = TypeVar("_V", bound=_Answer)


class CacheEntry(NamedTuple, Generic[_V]):
    """One caller's cached read. ``stored_at`` is when it was written, the
    sweep's clock; ``answered_at`` is when MindsHub gave ``value``, which an
    entry rewritten to serve it again keeps."""

    stored_at: float
    fresh_until: float
    answered_at: float
    value: _V


class CallerTtlCache(Generic[_V]):
    """Answers MindsHub gave per caller (CallerCacheKey), each fresh for
    ``ttl_ok_s`` when MindsHub answered and ``ttl_fail_s`` when it couldn't.

    Entries are swept on write, ``keep_s`` after they were stored: nothing
    re-reads a departed caller's key, so without a sweep the cache grows for
    as long as the process lives. ``keep_s`` defaults to the longer TTL, so
    raising one never evicts a live entry; a cache that serves an entry past
    its TTL (mindshub_teams, while MindsHub throttles) keeps it that long.
    """

    def __init__(
        self,
        *,
        ttl_ok_s: float = MINDSHUB_TTL_OK_S,
        ttl_fail_s: float = MINDSHUB_TTL_FAIL_S,
        keep_s: float | None = None,
    ) -> None:
        self.ttl_ok_s = ttl_ok_s
        self.ttl_fail_s = ttl_fail_s
        self.keep_s = max(ttl_ok_s, ttl_fail_s) if keep_s is None else keep_s
        self._entries: dict[CallerCacheKey, CacheEntry[_V]] = {}

    def fresh(self, key: CallerCacheKey) -> _V | None:
        """``key``'s value while it is fresh, else None."""
        entry = self._entries.get(key)
        if entry is None or time.monotonic() >= entry.fresh_until:
            return None
        return entry.value

    def answered_within(self, key: CallerCacheKey, *, max_age_s: float) -> _V | None:
        """``key``'s value when MindsHub answered it at most ``max_age_s``
        ago, fresh or not, else None."""
        entry = self._entries.get(key)
        if entry is None or not entry.value.reachable:
            return None
        if time.monotonic() - entry.answered_at > max_age_s:
            return None
        return entry.value

    def put(self, key: CallerCacheKey, value: _V, *, ttl_s: float | None = None) -> None:
        """Store ``value`` for ``key``, answered now, fresh for ``ttl_s``:
        by default ``ttl_ok_s`` when it is reachable, else ``ttl_fail_s``."""
        now = time.monotonic()
        self._store(key, value, now=now, ttl_s=ttl_s, answered_at=now)

    def serve_again(self, key: CallerCacheKey) -> None:
        """Keep ``key``'s value for another ``ttl_fail_s``, still dated when
        MindsHub gave it, so answered_within keeps bounding it."""
        entry = self._entries.get(key)
        if entry is not None:
            self._store(
                key, entry.value, now=time.monotonic(), ttl_s=self.ttl_fail_s,
                answered_at=entry.answered_at,
            )

    def _store(
        self, key: CallerCacheKey, value: _V, *, now: float, ttl_s: float | None, answered_at: float
    ) -> None:
        cutoff = now - self.keep_s
        for stale in [k for k, entry in self._entries.items() if entry.stored_at < cutoff]:
            del self._entries[stale]
        if ttl_s is None:
            ttl_s = self.ttl_ok_s if value.reachable else self.ttl_fail_s
        self._entries[key] = CacheEntry(now, now + ttl_s, answered_at, value)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()


def sweep_cache(cache: MutableMapping[Any, tuple[Any, ...]], *, max_ttl_s: float) -> None:
    """Drop the entries of a cache of ``(stored_at, ...)`` tuples that no TTL
    up to ``max_ttl_s`` could still consider fresh, for a cache that keeps
    its own entries (the model catalog, cowork.services.providers)."""
    cutoff = time.monotonic() - max_ttl_s
    for key in [k for k, entry in cache.items() if entry[0] < cutoff]:
        del cache[key]
