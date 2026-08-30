"""Authorization and authorship for organization-shared resources.

The gateway remains the source of role vocabulary.  This module only consumes
the trusted ``Principal`` and the existing ``can_manage_org`` adapter; it does
not invent local admin roles.  Local/desktop mode keeps its single-user
behavior and does not write audit rows.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
from threading import Lock, RLock
import time
from typing import Any, Callable, Iterator
from uuid import UUID, uuid4

import sqlalchemy as sa
from fastapi import HTTPException, status
from sqlalchemy.pool import NullPool

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import ScopedSession
from cowork.models.shared_resource import (
    SharedResourceAttribution,
    SharedResourceMutation,
)
from cowork.principal import Principal, can_manage_org
from cowork.schemas.shared_resources import AttributionActor, ResourceAttribution


PROJECT = "project"
SKILL = "skill"
PROJECT_MEMORY = "project_memory"
PROJECT_INSTRUCTIONS = "project_instructions"
SKILL_PROJECT_REFERENCES = "skill_project_references"
CLAIM_TTL = timedelta(minutes=5)
RESOURCE_LOCK_TIMEOUT_SECONDS = 10.0
ADVISORY_LOCK_RETRY_SECONDS = 0.05

_PROCESS_LOCKS: dict[tuple[str, str, str], RLock] = {}
_PROCESS_LOCKS_GUARD = Lock()
_DATABASE_LOCK_ENGINES: dict[object, Any] = {}
_DATABASE_LOCK_ENGINES_GUARD = Lock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _next_modified_at(previous: datetime | None) -> datetime:
    """Return a UTC timestamp strictly newer than the persisted value."""
    now = _utc_now()
    if previous is None:
        return now
    previous_utc = (
        previous.replace(tzinfo=timezone.utc)
        if previous.tzinfo is None
        else previous.astimezone(timezone.utc)
    )
    if now <= previous_utc:
        return previous_utc + timedelta(microseconds=1)
    return now


def _process_lock(org_id: str, kind: str, key: str) -> RLock:
    identity = (org_id, kind, key)
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(identity, RLock())


def _resource_busy() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="The resource is busy; retry the request",
    )


def _database_lock_engine(bind: Any) -> Any:
    """Return an unpooled engine reserved for session advisory locks.

    ORM sessions and advisory locks have opposite acquisition orders across
    request flows. Keeping advisory connections outside the ORM QueuePool
    prevents those flows from filling that pool while each waits for a second
    slot. ``NullPool`` also makes closing the outer lock connection release the
    physical PostgreSQL session immediately.
    """
    url = bind.url
    with _DATABASE_LOCK_ENGINES_GUARD:
        engine = _DATABASE_LOCK_ENGINES.get(url)
        if engine is None:
            settings = get_app_settings()
            engine = sa.create_engine(
                url,
                poolclass=NullPool,
                pool_pre_ping=settings.database.pool_pre_ping,
                isolation_level="AUTOCOMMIT",
                connect_args={
                    "connect_timeout": max(
                        1,
                        min(10, settings.database.pool_timeout),
                    )
                },
            )
            _DATABASE_LOCK_ENGINES[url] = engine
        return engine


class _DatabaseResourceLocks:
    """One request's cross-replica locks on one dedicated connection.

    PostgreSQL session advisory locks survive commits made by the request's ORM
    session. Nested project cascades share one unpooled lock connection rather
    than opening one connection per project child. SQLite tests/local mode use
    only the keyed process lock.
    """

    def __init__(self, session: ScopedSession) -> None:
        self.session = session
        self.connection: Any | None = None
        self.depth = 0

    @contextmanager
    def lock(
        self,
        org_id: str,
        kind: str,
        key: str,
        *,
        deadline: float | None = None,
    ) -> Iterator[None]:
        bind = self.session.get_bind()
        if bind.dialect.name != "postgresql":
            yield
            return

        if deadline is None:
            deadline = time.monotonic() + RESOURCE_LOCK_TIMEOUT_SECONDS
        owns_connection = self.connection is None
        connection = self.connection or _database_lock_engine(bind).connect()
        if owns_connection:
            self.connection = connection
        self.depth += 1
        identity = f"{org_id}\0{kind}\0{key}".encode()
        lock_key = int.from_bytes(
            hashlib.blake2b(identity, digest_size=8).digest(),
            byteorder="big",
            signed=True,
        )
        acquired = False
        try:
            while True:
                acquired = bool(
                    connection.execute(
                        sa.text("SELECT pg_try_advisory_lock(:lock_key)"),
                        {"lock_key": lock_key},
                    ).scalar()
                )
                if acquired:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _resource_busy()
                time.sleep(min(ADVISORY_LOCK_RETRY_SECONDS, remaining))
            yield
        finally:
            try:
                if acquired:
                    connection.execute(
                        sa.text("SELECT pg_advisory_unlock(:lock_key)"),
                        {"lock_key": lock_key},
                    )
            finally:
                self.depth -= 1
                if owns_connection:
                    self.connection = None
                    connection.close()


def project_resource_key(project_id: UUID | str) -> str:
    return str(project_id)


def project_memory_resource_key(project_id: UUID | str, slot: str) -> str:
    return f"{project_id}:{slot}"


def actor_payload(user_id: str | None, email: str | None) -> AttributionActor | None:
    if not user_id:
        return None
    return AttributionActor(user_id=user_id, email=email or "")


class SharedResourceAccess:
    """One request's shared-resource policy and audit writer."""

    def __init__(
        self,
        session: ScopedSession,
        principal: Principal | None = None,
    ) -> None:
        self.session = session
        self.principal = principal
        self._database_locks = _DatabaseResourceLocks(session)

    @property
    def org_mode(self) -> bool:
        return self.session.scope.org_mode

    @property
    def actor_id(self) -> str | None:
        return self.session.scope.user_id

    @property
    def actor_email(self) -> str | None:
        if not self._principal_matches_scope():
            return None
        return self.principal.email or None

    @property
    def is_admin(self) -> bool:
        return self._principal_matches_scope() and can_manage_org(self.principal)

    @property
    def has_trusted_actor(self) -> bool:
        return not self.org_mode or self._principal_matches_scope()

    def _principal_matches_scope(self) -> bool:
        principal = self.principal
        scope = self.session.scope
        return bool(
            isinstance(principal, Principal)
            and principal.user_id == scope.user_id
            and principal.org_id == scope.org_id
        )

    def can_change(self, creator_id: str | None) -> bool:
        """Creator or org admin; unattributed legacy resources are admin-only."""
        if not self.org_mode:
            return True
        if not self._principal_matches_scope():
            return False
        return self.is_admin or bool(
            creator_id and self.actor_id and creator_id == self.actor_id
        )

    def require_actor(self) -> None:
        if self.org_mode:
            self._require_actor()

    def require_change(self, creator_id: str | None, *, detail: str) -> None:
        if not self.can_change(creator_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=detail,
            )

    def attribution(
        self,
        kind: str,
        key: str,
        *,
        fallback_creator_id: str | None = None,
        fallback_creator_email: str | None = None,
        fallback_modified_at: datetime | None = None,
    ) -> ResourceAttribution:
        row = self._find(kind, key) if self.org_mode else None
        if row is not None:
            if row.pending_claim_token:
                return ResourceAttribution(
                    created_by=None,
                    last_modified_by=None,
                    last_modified_at=None,
                )
            return ResourceAttribution(
                created_by=actor_payload(row.created_by_id, row.created_by_email),
                last_modified_by=actor_payload(row.updated_by_id, row.updated_by_email),
                last_modified_at=row.modified_at,
            )
        return ResourceAttribution(
            created_by=actor_payload(fallback_creator_id, fallback_creator_email),
            last_modified_by=None,
            last_modified_at=fallback_modified_at,
        )

    def creator_id(
        self,
        kind: str,
        key: str,
        *,
        fallback: str | None = None,
    ) -> str | None:
        if not self.org_mode:
            return self.actor_id or fallback
        row = self._find(kind, key)
        return row.created_by_id if row is not None else fallback

    def has_attribution(self, kind: str, key: str) -> bool:
        return self.org_mode and self._find(kind, key) is not None

    def claim_is_pending(self, kind: str, key: str) -> bool:
        row = self._find(kind, key) if self.org_mode else None
        return bool(row is not None and row.pending_claim_token)

    def recover_stale_claim(
        self,
        kind: str,
        key: str,
        *,
        resource_exists: Callable[[], bool],
    ) -> bool:
        """Recover a crashed filesystem claim after its bounded lease expires.

        An empty resource releases the key for a new first writer. If bytes
        exist, finalize with the reserved actor snapshot and a recovery action.
        This preserves an actor for every surviving mutation even though the
        originating process died before its audit commit.
        """
        if not self.org_mode:
            return False
        self._require_actor()
        # The filesystem observation is part of the recovery decision. Evaluate
        # it only after acquiring the same cross-replica identity used by every
        # writer, otherwise an expired empty claim can be released from a stale
        # pre-lock snapshot while its original writer is still finalizing bytes.
        with self.coordination_lock(kind, key):
            row = self._find(kind, key)
            if row is None or not row.pending_claim_token:
                return False
            expires_at = row.pending_claim_expires_at
            if expires_at is not None and expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at is not None and expires_at > datetime.now(timezone.utc):
                return False
            current = self._find_for_update(row.id)
            if (
                current is None
                or current.pending_claim_token != row.pending_claim_token
            ):
                self.session.rollback()
                return False
            if resource_exists():
                if current.created_by_id is None:
                    raise RuntimeError("Pending claim has no reserved actor")
                current.updated_by_id = current.created_by_id
                current.updated_by_email = current.created_by_email
                current.pending_claim_token = None
                current.pending_claim_expires_at = None
                self.session.add(current)
                self._append_event_as(
                    current.resource_kind,
                    current.resource_key,
                    "create",
                    actor_id=current.created_by_id,
                    actor_email=current.created_by_email,
                )
            else:
                self.session.delete(current)
            self.session.commit()
            return True

    @contextmanager
    def coordination_lock(
        self,
        namespace: str,
        key: str,
    ) -> Iterator[None]:
        """Serialize a derived cross-resource invariant without attribution."""
        if not self.org_mode:
            yield
            return
        self._require_actor()
        org_id = self.session.scope.org_id
        if org_id is None:
            raise RuntimeError("Organization scope disappeared during coordination")
        with self._resource_lock(org_id, namespace, key):
            yield

    @contextmanager
    def _resource_lock(
        self,
        org_id: str,
        kind: str,
        key: str,
    ) -> Iterator[None]:
        deadline = time.monotonic() + RESOURCE_LOCK_TIMEOUT_SECONDS
        process_lock = _process_lock(org_id, kind, key)
        remaining = max(0.0, deadline - time.monotonic())
        if not process_lock.acquire(timeout=remaining):
            raise _resource_busy()
        try:
            with self._database_locks.lock(
                org_id,
                kind,
                key,
                deadline=deadline,
            ):
                yield
        finally:
            process_lock.release()

    @contextmanager
    def mutation_lock(
        self,
        kind: str,
        key: str,
        *,
        fallback_creator_id: str | None = None,
        fallback_creator_email: str | None = None,
        resource_exists: Callable[[], bool] | None = None,
    ) -> Iterator[SharedResourceAttribution | None]:
        """Serialize read, filesystem mutation, and audit for one resource.

        PostgreSQL's row lock coordinates replicas. The keyed process lock
        supplies equivalent behavior for SQLite tests and same-process local
        concurrency. A legacy filesystem resource gets a null-owner row before
        the lock so concurrent admins share the same serialization identity.
        """
        if not self.org_mode:
            yield None
            return
        self._require_actor()
        org_id = self.session.scope.org_id
        if org_id is None:
            raise RuntimeError("Organization scope disappeared during mutation")
        with self._resource_lock(org_id, kind, key):
            row = self._find(kind, key)
            created_placeholder = False
            if row is None:
                # A request may have observed a legacy filesystem resource
                # before waiting behind a rename/delete. Recheck while holding
                # the serialization identity before creating a null-owner row,
                # otherwise a stale waiter leaves a ghost attribution that
                # blocks a later create of the same key.
                if resource_exists is None or not resource_exists():
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Shared resource no longer exists",
                    )
                row = SharedResourceAttribution(
                    resource_kind=kind,
                    resource_key=key,
                    created_by_id=fallback_creator_id,
                    created_by_email=fallback_creator_email,
                )
                try:
                    self.session.add(row)
                    self.session.commit()
                    created_placeholder = True
                except sa.exc.IntegrityError:
                    self.session.rollback()
                    row = self._find(kind, key)
                    if row is None:
                        raise
            current = self._find_for_update(row.id)
            if current is None:
                self.session.rollback()
                raise RuntimeError("Shared resource disappeared before mutation")
            if current.pending_claim_token:
                self.session.rollback()
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Shared-resource ownership is still being established",
                )
            try:
                yield current
            finally:
                # Audit helpers normally commit. A newly-created legacy lock
                # identity is provisional until an audit stamps an editor; if
                # the operation became a no-op, was denied after a recheck, or
                # failed and compensated, remove that pristine placeholder.
                self.session.rollback()
                if created_placeholder:
                    placeholder = self._find_for_update(row.id)
                    if (
                        placeholder is not None
                        and placeholder.updated_by_id is None
                        and placeholder.pending_claim_token is None
                    ):
                        self.session.delete(placeholder)
                        self.session.commit()
                    else:
                        self.session.rollback()

    def register(
        self,
        kind: str,
        key: str,
        *,
        action: str = "create",
        creator_id: str | None = None,
        creator_email: str | None = None,
    ) -> SharedResourceAttribution | None:
        """Register a newly created resource and append its first event."""
        row, _created = self.claim(
            kind,
            key,
            action=action,
            creator_id=creator_id,
            creator_email=creator_email,
        )
        return row

    def claim(
        self,
        kind: str,
        key: str,
        *,
        action: str = "create",
        creator_id: str | None = None,
        creator_email: str | None = None,
    ) -> tuple[SharedResourceAttribution | None, bool]:
        """Atomically create attribution and its first event.

        Filesystem-backed memory and instructions use ``reserve_claim`` around
        disk I/O. SQL-only registration and already-exclusive skill creation
        do not need a durable pending state.
        """
        if not self.org_mode:
            return None, False
        actor_id = self._require_actor()
        existing = self._find(kind, key)
        if existing is not None:
            return existing, False
        created_by_id = creator_id if creator_id is not None else actor_id
        created_by_email = creator_email if creator_id is not None else self.actor_email
        row = SharedResourceAttribution(
            resource_kind=kind,
            resource_key=key,
            created_by_id=created_by_id,
            created_by_email=created_by_email,
            updated_by_id=actor_id,
            updated_by_email=self.actor_email,
        )
        try:
            self.session.add(row)
            self._append_event(kind, key, action)
            self.session.commit()
        except sa.exc.IntegrityError:
            self.session.rollback()
            winner = self._find(kind, key)
            if winner is None:
                raise
            return winner, False
        return row, True

    def reserve_claim(
        self,
        kind: str,
        key: str,
        *,
        creator_id: str | None = None,
        creator_email: str | None = None,
    ) -> tuple[SharedResourceAttribution | None, str | None]:
        """Reserve an unowned key, returning ``(row, private claim token)``.

        The unique key is the race arbiter.  A loser rolls back its attempted
        claim and re-reads the winner before any caller touches the filesystem.
        No mutation event exists until ``finalize_claim`` succeeds.
        """
        if not self.org_mode:
            return None, None
        actor_id = self._require_actor()
        existing = self._find(kind, key)
        if existing is not None:
            # Do not let a create-shaped retry seize an existing resource.
            return existing, None
        created_by_id = creator_id if creator_id is not None else actor_id
        created_by_email = creator_email if creator_id is not None else self.actor_email
        claim_token = str(uuid4())
        row = SharedResourceAttribution(
            resource_kind=kind,
            resource_key=key,
            created_by_id=created_by_id,
            created_by_email=created_by_email,
            pending_claim_token=claim_token,
            pending_claim_expires_at=datetime.now(timezone.utc) + CLAIM_TTL,
        )
        try:
            self.session.add(row)
            self.session.commit()
        except sa.exc.IntegrityError:
            self.session.rollback()
            winner = self._find(kind, key)
            if winner is None:
                raise
            return winner, None
        return row, claim_token

    def ensure_mutation_identity(
        self,
        kind: str,
        key: str,
    ) -> tuple[SharedResourceAttribution, bool]:
        """Create a null-owner lock row for a verified legacy resource.

        Callers must verify the filesystem resource before invoking this. The
        unique key arbitrates a race with a first-writer reservation; no audit
        event or author is invented merely to establish serialization.
        """
        self._require_actor()
        existing = self._find(kind, key)
        if existing is not None:
            return existing, False
        row = SharedResourceAttribution(
            resource_kind=kind,
            resource_key=key,
        )
        try:
            self.session.add(row)
            self.session.commit()
        except sa.exc.IntegrityError:
            self.session.rollback()
            winner = self._find(kind, key)
            if winner is None:
                raise
            return winner, False
        return row, True

    def release_pristine_identity(self, row: SharedResourceAttribution) -> bool:
        """Remove an unchanged null-owner row created only for locking."""
        self._require_actor()
        current = self._find_for_update(row.id)
        if (
            current is None
            or current.created_by_id is not None
            or current.updated_by_id is not None
            or current.pending_claim_token is not None
        ):
            self.session.rollback()
            return False
        self.session.delete(current)
        self.session.commit()
        return True

    def finalize_claim(
        self,
        row: SharedResourceAttribution,
        claim_token: str,
        *,
        action: str,
    ) -> SharedResourceAttribution | None:
        """Finalize only the still-pending reservation identified by ``claim_token``."""
        actor_id = self._require_actor()
        current = self._find_for_update(row.id)
        if current is None or current.pending_claim_token != claim_token:
            self.session.rollback()
            return None
        current.pending_claim_token = None
        current.pending_claim_expires_at = None
        current.updated_by_id = actor_id
        current.updated_by_email = self.actor_email
        self.session.add(current)
        self._append_event(current.resource_kind, current.resource_key, action)
        self.session.commit()
        return current

    def record_update(
        self,
        kind: str,
        key: str,
        *,
        action: str,
        fallback_creator_id: str | None = None,
        fallback_creator_email: str | None = None,
        new_key: str | None = None,
    ) -> SharedResourceAttribution | None:
        return self.record_updates(
            kind,
            key,
            actions=[action],
            fallback_creator_id=fallback_creator_id,
            fallback_creator_email=fallback_creator_email,
            new_key=new_key,
        )

    def record_updates(
        self,
        kind: str,
        key: str,
        *,
        actions: list[str],
        fallback_creator_id: str | None = None,
        fallback_creator_email: str | None = None,
        new_key: str | None = None,
    ) -> SharedResourceAttribution | None:
        """Record all semantic actions from one filesystem mutation atomically."""
        if not self.org_mode:
            return None
        if not actions:
            self.session.rollback()
            return None
        row = self.stage_updates(
            kind,
            key,
            actions=actions,
            fallback_creator_id=fallback_creator_id,
            fallback_creator_email=fallback_creator_email,
            new_key=new_key,
        )
        self.session.commit()
        return row

    def stage_update(
        self,
        kind: str,
        key: str,
        *,
        action: str,
        fallback_creator_id: str | None = None,
        fallback_creator_email: str | None = None,
        new_key: str | None = None,
    ) -> SharedResourceAttribution | None:
        """Stage one event without committing the surrounding transaction."""
        return self.stage_updates(
            kind,
            key,
            actions=[action],
            fallback_creator_id=fallback_creator_id,
            fallback_creator_email=fallback_creator_email,
            new_key=new_key,
        )

    def stage_updates(
        self,
        kind: str,
        key: str,
        *,
        actions: list[str],
        fallback_creator_id: str | None = None,
        fallback_creator_email: str | None = None,
        new_key: str | None = None,
    ) -> SharedResourceAttribution | None:
        """Stage events and attribution in the caller's current transaction."""
        if not self.org_mode or not actions:
            return None
        actor_id = self._require_actor()
        row = self._find(kind, key)
        if row is not None and row.pending_claim_token:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Shared-resource ownership is still being established",
            )
        if row is None:
            row = SharedResourceAttribution(
                resource_kind=kind,
                resource_key=key,
                created_by_id=fallback_creator_id,
                created_by_email=fallback_creator_email,
            )
            self.session.add(row)
        if new_key is not None:
            row.resource_key = new_key
        row.updated_by_id = actor_id
        row.updated_by_email = self.actor_email
        # Assign explicitly even when the same actor edits twice. Updating only
        # the actor columns leaves SQLAlchemy with no dirty attribution fields,
        # so the model's server/on-update timestamp would not run.
        row.modified_at = _next_modified_at(row.modified_at)
        self.session.add(row)
        for action in actions:
            self._append_event(kind, new_key or key, action)
        self.session.flush()
        return row

    def release_claim(
        self,
        row: SharedResourceAttribution,
        *,
        claim_token: str,
    ) -> bool:
        """Remove only a still-pending claim whose filesystem write failed.

        Claims commit before disk I/O so the unique key can arbitrate first
        authors.  No audit event has been appended yet.  Locking and matching
        the private token prevent a delayed failure from deleting ownership
        that another turn has already finalized.
        """
        actor_id = self._require_actor()
        current = self._find_for_update(row.id)
        if (
            current is None
            or current.created_by_id != actor_id
            or current.pending_claim_token != claim_token
        ):
            self.session.rollback()
            return False
        self.session.delete(current)
        self.session.commit()
        return True

    def record_delete(
        self,
        kind: str,
        key: str,
        *,
        fallback_creator_id: str | None = None,
        fallback_creator_email: str | None = None,
    ) -> None:
        self.record_deletes([(kind, key, "delete")])

    def record_deletes(
        self,
        resources: list[tuple[str, str, str]],
        *,
        pending_claim_tokens: set[str] | None = None,
    ) -> None:
        """Audit and remove current attribution for a deletion cascade."""
        if not self.org_mode:
            return
        self.stage_deletes(
            resources,
            pending_claim_tokens=pending_claim_tokens,
        )
        self.session.commit()

    def stage_deletes(
        self,
        resources: list[tuple[str, str, str]],
        *,
        pending_claim_tokens: set[str] | None = None,
    ) -> None:
        """Stage a deletion trail and current-row cleanup without committing."""
        if not self.org_mode:
            return
        self._require_actor()
        rows: dict[UUID, SharedResourceAttribution] = {}
        for kind, key, _action in resources:
            row = self._find(kind, key)
            if (
                row is not None
                and row.pending_claim_token
                and row.pending_claim_token not in (pending_claim_tokens or set())
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Shared-resource ownership is still being established",
                )
            if row is not None:
                rows[row.id] = row
        # Events survive after current attribution disappears. A later
        # resource may reuse a human key with a new owner and clean history.
        for kind, key, action in resources:
            self._append_event(kind, key, action)
        for row in rows.values():
            self.session.delete(row)
        self.session.flush()

    def _find(self, kind: str, key: str) -> SharedResourceAttribution | None:
        return self.session.exec(
            self.session.select(SharedResourceAttribution).where(
                SharedResourceAttribution.resource_kind == kind,
                SharedResourceAttribution.resource_key == key,
            )
        ).first()

    def _find_for_update(
        self,
        attribution_id: UUID,
    ) -> SharedResourceAttribution | None:
        return self.session.exec(
            self.session.select(SharedResourceAttribution)
            .where(SharedResourceAttribution.id == attribution_id)
            .with_for_update()
        ).first()

    def _require_actor(self) -> str:
        if (
            not self.session.scope.org_id
            or not self.actor_id
            or not self._principal_matches_scope()
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Shared-resource mutation requires a trusted organization identity",
            )
        return self.actor_id

    def _append_event(self, kind: str, key: str, action: str) -> None:
        actor_id = self._require_actor()
        self._append_event_as(
            kind,
            key,
            action,
            actor_id=actor_id,
            actor_email=self.actor_email,
        )

    def _append_event_as(
        self,
        kind: str,
        key: str,
        action: str,
        *,
        actor_id: str,
        actor_email: str | None,
    ) -> None:
        self.session.add(
            SharedResourceMutation(
                resource_kind=kind,
                resource_key=key,
                action=action,
                actor_id=actor_id,
                actor_email=actor_email,
            )
        )
