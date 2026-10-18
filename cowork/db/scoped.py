"""Tenant-scoped query layer for partitioned deployments.

Services receive a ScopedSession instead of a raw Session. Statements are
built with ``scoped.select(Model)`` — when the model is org-scoped (has an
``org_id`` column) and the scope partitions it, the org filter is
pre-applied and cannot be forgotten. ``exec()`` only runs statements built
that way; there is no raw execution path on this object.

Two scopes partition. Org mode (hosted org tenancy) partitions every model
by organization. A personal scope (local tenancy with
COWORK_TRUST_IDENTITY_HEADERS on) keys its partition by the person's
partition id, and partitions only the resources COWORK_PERSONAL_RESOURCES
lists (cowork.db.personal_resources); every other model behaves as in local
mode.

Fail-closed rule: a scope that partitions an org-scoped model but has no
org in it raises MissingTenantScopeError. Models without an ``org_id`` column
pass through untouched, so the helper is inert until the tenancy migrations
land and turns on per table as columns arrive.

Local mode (the desktop sidecar, and the operator view of a trusted-identity
deployment) never filters — today's behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, Request
from sqlalchemy import event, inspect
from sqlalchemy.sql import Select
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.personal_resources import MODEL_RESOURCE, STORE_RESOURCE, PersonalResource
from cowork.db.session import get_session
from cowork.principal import Principal, get_principal


class MissingTenantScopeError(RuntimeError):
    """Partitioned data touched without an org (or partition) in scope."""


class TenantMismatchError(RuntimeError):
    """A write carried an org_id that conflicts with the request's scope."""


@dataclass(frozen=True)
class TenantScope:
    """Tenancy context for one request.

    ``org_mode`` is hosted org tenancy: every resource is keyed by ``org_id``.
    ``personal`` is one person's partition on a local deployment that trusts
    identity headers: ``org_id`` is their partition id
    (cowork.principal.personal_partition_id), ``user_id`` their ``sub``, and
    only ``personal_resources`` are keyed by ``org_id``. The others stay shared,
    as LOCAL_SCOPE has them.
    """

    org_mode: bool = False
    org_id: str | None = None
    user_id: str | None = None
    personal: bool = False
    personal_resources: frozenset[PersonalResource] = frozenset()

    @property
    def partitioned(self) -> bool:
        """True in org mode and in a personal scope, whatever its resources."""
        return self.org_mode or self.personal

    def partitioned_for(self, resource: PersonalResource) -> bool:
        """True when ``resource`` is keyed by ``org_id`` in this scope."""
        return self.org_mode or (self.personal and resource in self.personal_resources)

    def for_resource(self, resource: PersonalResource) -> TenantScope:
        """This scope where ``resource`` is partitioned, else LOCAL_SCOPE, the
        deployment-wide view a shared resource keeps."""
        return self if self.partitioned_for(resource) else LOCAL_SCOPE


LOCAL_SCOPE = TenantScope()

# Alias for background executors (scheduler loop): "don't filter by org".
SYSTEM_SCOPE = LOCAL_SCOPE


def tenant_scope_required() -> bool:
    """True when code with no scope must refuse rather than fall back to
    LOCAL_SCOPE or a base storage root: in org tenancy, and on a local
    deployment that trusts identity headers. There LOCAL_SCOPE is the operator
    view, which lists every person's conversations, running answers,
    projects, files, schedules and pins, and reads skills, connections,
    artifacts and memory from the deployment's own stores."""
    settings = get_app_settings()
    return settings.tenancy_mode == "org" or settings.trust_identity_headers


def scope_from_principal(principal: Principal | None) -> TenantScope:
    """The single way a request's TenantScope is built — request DI and
    background holders of a principal (e.g. ResponsesHandler) both go through
    here. Org tenancy without a principal gives an org scope with no org, which
    fails closed; local tenancy without one gives LOCAL_SCOPE, the operator view
    when identity headers are trusted."""
    if principal is None:
        if get_app_settings().tenancy_mode == "org":
            return TenantScope(org_mode=True)
        return LOCAL_SCOPE
    return scope_for_identity(org_id=principal.org_id, user_id=principal.user_id)


def scope_for_identity(*, org_id: str | None, user_id: str | None) -> TenantScope:
    """Scope for a known (org_id, user_id), as a principal or a stored row
    carries it. On a trusted-identity deployment ``org_id`` is the partition id
    the middleware derived, never the raw X-Organization-Id.

    Org tenancy → an org scope. Local tenancy with trusted identity headers and
    an org_id → that person's partition (``user_id`` may be None for a
    partition-owned background operation). Anything else → LOCAL_SCOPE: an
    org_id alone never turns org mode on in local tenancy.
    """
    settings = get_app_settings()
    if settings.tenancy_mode == "org":
        return TenantScope(org_mode=True, org_id=org_id, user_id=user_id)
    if settings.trust_identity_headers and org_id:
        return TenantScope(
            personal=True,
            org_id=org_id,
            user_id=user_id,
            personal_resources=settings.personal_resources,
        )
    return LOCAL_SCOPE


def scope_for_org(org_id: str | None) -> TenantScope:
    """Scope for a background or webhook-routed operation with a known org_id.
    No org returns LOCAL_SCOPE; an org_id returns its org scope in org tenancy
    and its partition on a trusted-identity deployment (scope_for_identity).
    Replaces the inline pattern:
    `SYSTEM_SCOPE if org_id is None else TenantScope(org_mode=True, org_id=org_id)`."""
    return LOCAL_SCOPE if org_id is None else scope_for_identity(org_id=org_id, user_id=None)


def get_tenant_scope(request: Request) -> TenantScope:
    """FastAPI dependency: the request's tenant scope."""
    return scope_from_principal(get_principal(request))


def scope_for_background_context() -> TenantScope:
    """Scope for code with no request principal and no owning row to derive one
    from (channels: an inbound webhook has no per-org routing yet, so there is
    nothing to build a service principal out of).

    Local mode → LOCAL_SCOPE (today's behavior). Org mode → fail closed: this
    caller has no service principal available (channels are 501-gated in org
    mode for the same reason), and org mode must never silently write unscoped
    rows that users can't see. Fail loud instead of creating invisible data.
    See `service_principal_for` for callers that DO have an owning row.
    """
    if get_app_settings().tenancy_mode != "org":
        return LOCAL_SCOPE
    raise MissingTenantScopeError(
        "background conversation creation requires a service principal (not yet implemented in org mode)"
    )


def service_principal_for(org_id: str | None, user_id: str | None) -> Principal | None:
    """Principal for a background job driven by a stored row's own org_id/created_by
    (e.g. a fired Schedule) — there is no inbound request or gateway header to build
    one from, but the row already carries who owns it from when it was created
    through the normal request-scoped path.

    Desktop, and a trusted-identity deployment whose schedules are shared → None
    (nothing to scope). Org mode, and a trusted-identity deployment whose
    schedules are personal → Principal from the row's own identity, so the job
    acts as its creator, or fail closed if the row predates stamping (a
    legacy/local row with no org_id or created_by) — same "fail loud, never
    invisible" rule as `scope_for_background_context`. The stored org_id is
    already a partition id there and is used as it is, never derived again.
    """
    settings = get_app_settings()
    personal_schedules = (
        settings.trust_identity_headers
        and PersonalResource.SCHEDULES in settings.personal_resources
    )
    if settings.tenancy_mode != "org" and not personal_schedules:
        return None
    if not org_id or not user_id:
        raise MissingTenantScopeError(
            "background job's row has no org_id/created_by to build a service principal from"
        )
    return Principal(user_id=user_id, org_id=org_id)


def _partitions_store(scope: TenantScope, store: str) -> bool:
    resource = STORE_RESOURCE.get(store)
    # Mapped to None (the sandbox store), or unmapped: partitioned wherever
    # anything is, so a new store is never shared by omission.
    # tests/test_personal_resources.py keeps the map whole.
    return scope.partitioned if resource is None else scope.partitioned_for(resource)


def _partitions_model(scope: TenantScope, model: type) -> bool:
    resource = MODEL_RESOURCE.get(getattr(model, "__tablename__", ""))
    # Unmapped: partitioned wherever anything is, as for stores above.
    return scope.partitioned if resource is None else scope.partitioned_for(resource)


def storage_segment(value: str, *, what: Literal["partition", "store", "user"]) -> str:
    """``value`` as one folder name below a storage root: returned as it is,
    or ValueError naming ``what`` (the partition, store or user).

    A segment must be its own base name. "" is dropped by pathlib, so the
    path collapses onto the folder above; "." and ".." name a folder that is
    not below; a separator names a folder further down, maybe in another
    partition, and an absolute value replaces the root. A backslash is refused
    too, so a value never means one thing here and another on a host that
    reads it as a separator."""
    segment = os.path.basename(value)
    if segment != value or segment in ("", ".", "..") or "\\" in segment:
        raise ValueError(f"invalid storage {what} segment: {value!r}")
    return segment


def scoped_storage_root(base: Path, scope: TenantScope | None, *, store: str) -> Path:
    """``base`` where the scope leaves ``store`` shared, ``<shared_root>/<org_id>/<store>``
    where it partitions it (org mode, or a personal scope whose resources include
    the store's, see STORE_RESOURCE), fail-closed without an org. Org-first so
    each org or partition is one mountable/GC-able subtree. org_id is a
    normalized UUID (TrustedHeaderMiddleware, personal_partition_id), and
    storage_segment refuses one that would leave its folder all the same, so
    the partition never rests on the middleware alone. ``store`` is required:
    deriving it from ``base`` would let a *_DIR env override silently rename an
    org's store.

    ``scope=None`` fail-closes on a partitioned deployment (tenant_scope_required).
    It cannot mean "no tenancy" there: ``base`` is the shared root every
    organization or person can read, so returning it verbatim hands a caller
    that merely forgot to thread its scope a path outside any partition. That is
    how the persisted connector vault ended up unpartitioned. A missing scope
    there is a bug at the call site, and this is where it surfaces."""
    if scope is None:
        if tenant_scope_required():
            raise MissingTenantScopeError(
                f"filesystem store {store!r} requires a TenantScope on a partitioned "
                "deployment; the caller must thread one through rather than defaulting "
                "to the shared root"
            )
        return base
    if not _partitions_store(scope, store):
        return base
    if not scope.org_id:
        raise MissingTenantScopeError(
            "filesystem store requires an organization in scope"
        )
    partition = storage_segment(scope.org_id, what="partition")
    store = storage_segment(store, what="store")
    shared = Path(get_app_settings().storage.shared_root)
    return shared / partition / store


# The segment under a store's partition that holds the per-person tiers.
USER_STORE_TIER_SEGMENT = "users"


def user_store_tiers_segments(partition: str, *, store: str) -> tuple[str, ...]:
    """The path segments, below the shared root, of the folder holding every
    person's ``store`` tier in one partition: ``(<partition>, <store>, "users")``."""
    return (partition, store, USER_STORE_TIER_SEGMENT)


def user_store_segments(partition: str, user_id: str, *, store: str) -> tuple[str, ...]:
    """The path segments, below the shared root, of one person's ``store`` tier:
    ``(<partition>, <store>, "users", <user_id>)``. scoped_user_storage_root and
    the per-person tier walks (remove_project_memory_tiers) both build their
    paths from these segments, so the layout is spelled once."""
    return (*user_store_tiers_segments(partition, store=store), user_id)


def scoped_user_storage_root(
    base: Path, scope: TenantScope | None, *, store: str
) -> Path:
    """``<shared_root>/<org_id>/<store>/users/<user_id>``, for stores that are one
    person's rather than the org's. ``base`` where the scope leaves ``store``
    shared (local mode, one user per machine, or a resource a personal scope
    keeps shared); a partitioned store fail-closes without BOTH ids, since
    silently sharing one person's store across an org is a correctness bug, not
    a lenient default (ADR-0002). Id checks run before any settings resolution.

    ``scope=None`` fail-closes on a partitioned deployment for the same reason
    it does in ``scoped_storage_root``: ``base`` is the shared namespace root
    there, so returning it verbatim gives an unscoped caller a path outside
    every partition. This matters most for the in-process harness memory root,
    whose caller reads an ambient scope that is None outside a bound turn."""
    if scope is None:
        if tenant_scope_required():
            raise MissingTenantScopeError(
                f"per-user filesystem store {store!r} requires a TenantScope on a "
                "partitioned deployment; the caller must thread one through rather "
                "than defaulting to the shared root"
            )
        return base
    if not _partitions_store(scope, store):
        return base
    if not scope.user_id:
        raise MissingTenantScopeError(
            "per-user filesystem store requires a user in scope"
        )
    user_id = storage_segment(scope.user_id, what="user")
    # scoped_storage_root checks the org and the store segment. The path below
    # joins only the values storage_segment returned, never the raw ids.
    scoped_storage_root(base, scope, store=store)
    partition = storage_segment(scope.org_id, what="partition")
    return Path(get_app_settings().storage.shared_root).joinpath(
        *user_store_segments(partition, user_id, store=store)
    )


def scope_of_session(session: Session) -> TenantScope | None:
    """The TenantScope a raw session was wrapped with, if any.

    Recovers the ORIGINAL scope (stored at wrap time) so downstream code that
    only holds a raw session — e.g. via object_session(row) — can re-wrap with
    the same authorization context instead of deriving one from row data.
    Unpartitioned wraps (LOCAL_SCOPE) store nothing; returns None.
    """
    return session.info.get("tenant_scope")


# Tables the generic scoped layer leaves alone. `settings` is here because
# SettingService owns its tenancy explicitly (routes keys to global/org/user
# rows itself), so the generic org-filter/stamp must not also touch it.
_TENANCY_DEFERRED_TABLES = frozenset({"settings"})


def _is_org_scoped(model: type) -> bool:
    if getattr(model, "__tablename__", None) in _TENANCY_DEFERRED_TABLES:
        return False
    return hasattr(model, "org_id")


class ScopedSelect:
    """Statement built by ScopedSession.select — the only kind exec() runs.

    Proxies the underlying Select so chaining (.where/.order_by/...) keeps
    returning a ScopedSelect.
    """

    def __init__(self, stmt: Any) -> None:
        self._stmt = stmt

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._stmt, name)
        if callable(attr):

            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                result = attr(*args, **kwargs)
                return ScopedSelect(result) if isinstance(result, Select) else result

            return _wrapped
        return attr


class ScopedSession:
    """Session wrapper that makes org scoping structural, not remembered.

    Each model is filtered, stamped and checked only where the scope partitions
    it: every org-scoped model in org mode, and in a personal scope the models
    whose resource (MODEL_RESOURCE) is personal.
    """

    def __init__(self, session: Session, scope: TenantScope) -> None:
        if isinstance(session, ScopedSession):
            # A wrapper-of-wrapper breaks exec() later — fail at construction.
            raise TypeError(
                "ScopedSession cannot wrap another ScopedSession; pass the raw Session"
            )
        self._session = session
        self.scope = scope
        if scope.partitioned:
            # One scope per raw session: re-wrapping with the same scope is a
            # no-op (request DI + handler paths share the request session);
            # a different scope is a bug — two tenants on one session.
            existing = session.info.get("tenant_scope")
            if existing is None:
                session.info["tenant_scope"] = scope
                # Validate every flush (incl. autoflush and commit) so a loaded
                # row mutated to another org can never persist.
                event.listen(session, "before_flush", self._before_flush)
            elif existing != scope:
                raise RuntimeError(
                    "session is already tenant-scoped to a different org"
                )

    def _before_flush(self, session: Any, flush_context: Any, instances: Any) -> None:
        # Stamp-or-reject: NULL-org writes (services not yet swept to the
        # scoped API) are adopted into the scope; a non-NULL mismatch is
        # always a bug. Deletes of NULL-org rows are allowed — pre-sweep
        # rows being cleaned up by their own tenant (e.g. cascades).
        for row in [*session.new, *session.dirty]:
            if self._partitions(type(row)):
                if row.org_id is None:
                    row.org_id = self.scope.org_id
                elif row.org_id != self.scope.org_id:
                    raise TenantMismatchError(
                        f"pending write on {type(row).__name__} with "
                        f"org_id={row.org_id!r} conflicts with scope "
                        f"org_id={self.scope.org_id!r}"
                    )
        for row in session.new:
            if (
                self.scope.user_id
                and getattr(row, "created_by", "missing") is None
                and _partitions_model(self.scope, type(row))
            ):
                row.created_by = self.scope.user_id
        for row in session.deleted:
            if self._partitions(type(row)):
                if row.org_id is not None and row.org_id != self.scope.org_id:
                    raise TenantMismatchError(
                        f"cannot delete {type(row).__name__} belonging to another org"
                    )

    def _partitions(self, model: type) -> bool:
        """True when this scope keys ``model``'s rows by org_id. Fails closed
        when it would but has no org to key them by."""
        if not (_is_org_scoped(model) and _partitions_model(self.scope, model)):
            return False
        if self.scope.org_id is None:
            raise MissingTenantScopeError(
                f"{model.__name__} is org-scoped but the request has no org in scope"
            )
        return True

    def select(self, model: type) -> ScopedSelect:
        stmt = select(model)
        if self._partitions(model):
            stmt = stmt.where(model.org_id == self.scope.org_id)
        return ScopedSelect(stmt)

    def exec(self, stmt: ScopedSelect) -> Any:
        if not isinstance(stmt, ScopedSelect):
            raise TypeError(
                "ScopedSession.exec only runs statements built by .select()"
            )
        return self._session.exec(stmt._stmt)

    def get(self, model: type, ident: Any) -> Any:
        if self._partitions(model):
            row = self._session.get(model, ident)
            if row is not None and row.org_id != self.scope.org_id:
                return None
            return row
        return self._session.get(model, ident)

    def add(self, row: Any) -> Any:
        if self._partitions(type(row)):
            if row.org_id is None:
                row.org_id = self.scope.org_id
            elif row.org_id != self.scope.org_id:
                raise TenantMismatchError(
                    f"{type(row).__name__}.org_id={row.org_id!r} conflicts with scope "
                    f"org_id={self.scope.org_id!r}"
                )
        # Attribution: stamp the author on any model carrying created_by
        # (covers child rows like messages that have no org_id of their own),
        # where the scope partitions that model's resource.
        if (
            self.scope.user_id
            and inspect(row).transient
            and getattr(row, "created_by", "missing") is None
            and _partitions_model(self.scope, type(row))
        ):
            row.created_by = self.scope.user_id
        self._session.add(row)
        return row

    def delete(self, row: Any) -> None:
        if self._partitions(type(row)):
            if row.org_id != self.scope.org_id:
                raise TenantMismatchError(
                    f"cannot delete {type(row).__name__} belonging to another org"
                )
        self._session.delete(row)

    def commit(self) -> None:
        self._session.commit()

    def refresh(self, row: Any) -> None:
        self._session.refresh(row)

    def flush(self) -> None:
        self._session.flush()

    def rollback(self) -> None:
        self._session.rollback()

    def get_bind(self) -> Any:
        """Return the scoped session's engine for server-owned coordination.

        A unit's session is bound to one connection (cowork.db.units), and the
        advisory-lock engine reads its database URL from the engine, which a
        connection does not carry.
        """
        return self._session.get_bind().engine

    def close(self) -> None:
        self._session.close()


def unsafe_unscoped_session(scoped: ScopedSession) -> Session:
    """Deliberate escape hatch to the raw session (system jobs, migrations).

    Named to be greppable; never use it in service/request code.
    """
    return scoped._session


def adopt_scoped_session(session: Session) -> ScopedSession:
    """Wrap an existing raw Session in a ScopedSession, reusing the scope it was
    opened under. A partitioned scope is stamped into `session.info` (see
    ScopedSession.__init__); LOCAL_SCOPE has none, so fall back to LOCAL_SCOPE.

    Lets code holding only `object_session(row)` (e.g. harnesses replaying
    conversation history) reach the scoped query API without re-deriving the
    principal. Re-wrapping with the same scope is a no-op.
    """
    scope = session.info.get("tenant_scope") or LOCAL_SCOPE
    return ScopedSession(session, scope)


def get_scoped_session(
    session: Session = Depends(get_session),
    scope: TenantScope = Depends(get_tenant_scope),
) -> ScopedSession:
    """FastAPI dependency: the request's tenant-scoped session."""
    return ScopedSession(session, scope)


ScopedSessionDep = Annotated[ScopedSession, Depends(get_scoped_session)]
