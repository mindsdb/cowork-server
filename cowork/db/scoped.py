"""Tenant-scoped query layer for org (multi-tenant) deployments.

Services receive a ScopedSession instead of a raw Session. Statements are
built with ``scoped.select(Model)`` — when the model is org-scoped (has an
``org_id`` column) and the deployment runs in org mode, the org filter is
pre-applied and cannot be forgotten. ``exec()`` only runs statements built
that way; there is no raw execution path on this object.

Fail-closed rule: in org mode, touching an org-scoped model without an org
in scope raises MissingTenantScopeError. Models without an ``org_id`` column
pass through untouched, so the helper is inert until the tenancy migrations
land and turns on per table as columns arrive.

Local mode (the desktop sidecar) never filters — today's behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Request
from sqlalchemy import event
from sqlalchemy.sql import Select
from sqlmodel import Session, select

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.session import get_session
from cowork.principal import Principal, get_principal


class MissingTenantScopeError(RuntimeError):
    """Org-scoped data touched in org mode without an org in scope."""


class TenantMismatchError(RuntimeError):
    """A write carried an org_id that conflicts with the request's scope."""


@dataclass(frozen=True)
class TenantScope:
    """Tenancy context for one request."""

    org_mode: bool = False
    org_id: str | None = None
    user_id: str | None = None


LOCAL_SCOPE = TenantScope()


def scope_from_principal(principal: Principal | None) -> TenantScope:
    """The single way a TenantScope is built — request DI and background
    holders of a principal (e.g. ResponsesHandler) both go through here."""
    if get_app_settings().tenancy_mode != "org":
        return LOCAL_SCOPE
    return TenantScope(
        org_mode=True,
        org_id=principal.org_id if principal else None,
        user_id=principal.user_id if principal else None,
    )


def get_tenant_scope(request: Request) -> TenantScope:
    """FastAPI dependency: the request's tenant scope."""
    return scope_from_principal(get_principal(request))


def scope_for_background_context() -> TenantScope:
    """Scope for code with no request principal (scheduler, channels).

    Local mode → LOCAL_SCOPE (today's behavior). Org mode → fail closed:
    background work needs a service principal (deferred to the service-identity
    ticket), and org mode must never silently write unscoped rows that users
    can't see. Fail loud instead of creating invisible data.
    """
    if get_app_settings().tenancy_mode != "org":
        return LOCAL_SCOPE
    raise MissingTenantScopeError(
        "background conversation creation requires a service principal (not yet implemented in org mode)"
    )


def scope_of_session(session: Session) -> TenantScope | None:
    """The TenantScope a raw session was wrapped with, if any.

    Recovers the ORIGINAL scope (stored at wrap time) so downstream code that
    only holds a raw session — e.g. via object_session(row) — can re-wrap with
    the same authorization context instead of deriving one from row data.
    Local-mode wraps store nothing; returns None.
    """
    return session.info.get("tenant_scope")


def _is_org_scoped(model: type) -> bool:
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
    """Session wrapper that makes org scoping structural, not remembered."""

    def __init__(self, session: Session, scope: TenantScope) -> None:
        self._session = session
        self.scope = scope
        if scope.org_mode:
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
            if _is_org_scoped(type(row)):
                self._require_org(type(row))
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
            ):
                row.created_by = self.scope.user_id
        for row in session.deleted:
            if _is_org_scoped(type(row)):
                self._require_org(type(row))
                if row.org_id is not None and row.org_id != self.scope.org_id:
                    raise TenantMismatchError(
                        f"cannot delete {type(row).__name__} belonging to another org"
                    )

    def _require_org(self, model: type) -> None:
        if self.scope.org_mode and self.scope.org_id is None:
            raise MissingTenantScopeError(
                f"{model.__name__} is org-scoped but the request has no org in scope"
            )

    def select(self, model: type) -> ScopedSelect:
        stmt = select(model)
        if _is_org_scoped(model):
            self._require_org(model)
            if self.scope.org_mode:
                stmt = stmt.where(model.org_id == self.scope.org_id)
        return ScopedSelect(stmt)

    def exec(self, stmt: ScopedSelect) -> Any:
        if not isinstance(stmt, ScopedSelect):
            raise TypeError("ScopedSession.exec only runs statements built by .select()")
        return self._session.exec(stmt._stmt)

    def get(self, model: type, ident: Any) -> Any:
        if _is_org_scoped(model):
            self._require_org(model)
            row = self._session.get(model, ident)
            if row is not None and self.scope.org_mode and row.org_id != self.scope.org_id:
                return None
            return row
        return self._session.get(model, ident)

    def add(self, row: Any) -> Any:
        if _is_org_scoped(type(row)) and self.scope.org_mode:
            self._require_org(type(row))
            if row.org_id is None:
                row.org_id = self.scope.org_id
            elif row.org_id != self.scope.org_id:
                raise TenantMismatchError(
                    f"{type(row).__name__}.org_id={row.org_id!r} conflicts with scope "
                    f"org_id={self.scope.org_id!r}"
                )
        # Attribution: stamp the author on any model carrying created_by
        # (covers child rows like messages that have no org_id of their own).
        if (
            self.scope.org_mode
            and self.scope.user_id
            and getattr(row, "created_by", "missing") is None
        ):
            row.created_by = self.scope.user_id
        self._session.add(row)
        return row

    def delete(self, row: Any) -> None:
        if _is_org_scoped(type(row)) and self.scope.org_mode:
            self._require_org(type(row))
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

    def close(self) -> None:
        self._session.close()


def unsafe_unscoped_session(scoped: ScopedSession) -> Session:
    """Deliberate escape hatch to the raw session (system jobs, migrations).

    Named to be greppable; never use it in service/request code.
    """
    return scoped._session


def get_scoped_session(
    session: Session = Depends(get_session),
    scope: TenantScope = Depends(get_tenant_scope),
) -> ScopedSession:
    """FastAPI dependency: the request's tenant-scoped session."""
    return ScopedSession(session, scope)


ScopedSessionDep = Annotated[ScopedSession, Depends(get_scoped_session)]
