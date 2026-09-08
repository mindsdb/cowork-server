"""The permission front door: every cowork-server route declares who may
call it here (ENG-2094).

Ported from mindshub_inference's front door (ENG-1564, minds/api/v1/permissions.py)
— same shape, starting with only the two primitives every route can already
use. A capability check against auth's resolved org_role/permissions[] (never
the raw X-User-Roles header) lands once ENG-2094's transport decision — header
injection vs. an entitlements call — is picked.

``require(permission)`` turns a ``Permission`` class into a FastAPI dependency
a route declares directly, e.g. ``principal: Principal = Depends(require(Authenticated))``,
instead of a route registering into a separate path-keyed table. Pass the
class itself, not an instance: neither check takes any config, and a class is
already a stable, hashable object (the same one on every import), so
``require`` can be plain ``lru_cache``d and every ``require(Authenticated)``
call anywhere returns the identical dependency function — that's what lets
FastAPI's own per-request dependency cache collapse repeats.
"""

from __future__ import annotations

import inspect
from functools import cache
from typing import Protocol

from fastapi import Depends, HTTPException, Request, status

from cowork.principal import Principal, get_principal


class Permission(Protocol):
    """Something a route can pass to ``require(...)``."""

    async def check(self, request: Request) -> object:
        """Raise ``HTTPException`` to deny. Return the resolved identity to
        allow — ``Open`` returns ``None`` since there is nothing to resolve."""
        ...


class Open:
    """No identity required.

    Declaring this — rather than leaving a route with no permission
    dependency at all — is what lets the CI route walker (added once every
    route carries a declaration) tell "open on purpose" apart from "open
    because someone forgot the line"; both look identical at runtime
    without it.
    """

    async def check(self, request: Request) -> None:
        return None


class Authenticated:
    """A verified ``Principal`` is required; no further check.

    401s when the caller has no principal — no identity headers in org
    enforce mode, or audit/local mode where none was ever built. Use this
    for a route that is inherently a multi-tenant concept (e.g. the
    organization-switch capability) rather than one shared with desktop's
    single-user local mode, which never has a principal to check.

    Takes ``principal`` as a declared ``Depends(get_principal)`` rather than
    calling that function directly on ``request``: a subclass overriding
    ``check`` for its own reason (see ``NoStoreAuthenticated``) still gets
    ``principal`` resolved through FastAPI's own dependency graph, which is
    what lets a test's ``app.dependency_overrides[get_principal]`` reach it —
    a bare function call bypasses that entirely.
    """

    async def check(self, request: Request, principal: Principal | None = Depends(get_principal)) -> Principal:
        if principal is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        return principal


@cache
def require(permission_cls: type[Permission]):
    """Turn a ``Permission`` class into a FastAPI dependency.

    Stamps ``permission_cls`` onto the returned dependency so a route's
    declaration is visible by inspecting ``route.dependant.dependencies``
    directly — the CI walker needs a way to tell "this route called
    ``require(...)``" apart from an ordinary data dependency like
    ``Depends(get_principal)``, and this is the one place that can mark that
    without a second, hand-maintained table that could drift from what's
    actually wired.

    ``dependency``'s own ``__signature__`` is replaced with ``check``'s
    (minus ``self``): FastAPI decides what to resolve for a dependency by
    inspecting *that callable's* signature, not the ``Permission`` class
    behind it. Copying it now — even though ``Open``/``Authenticated`` only
    ever need ``request`` — is what lets a later ``Permission`` (a typed
    body model, a nested ``Depends(...)``) be resolved by FastAPI without
    this function changing.
    """
    permission = permission_cls()
    check_params = [p for name, p in inspect.signature(permission_cls.check).parameters.items() if name != "self"]

    async def dependency(**resolved):
        return await permission.check(**resolved)

    dependency.__signature__ = inspect.Signature(check_params)
    dependency.permission_cls = permission_cls
    return dependency
