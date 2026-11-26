"""Live organization authority for hosted execution and artifact mutation."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError

from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.db.scoped import TenantScope

ProductPermission = Literal["product.execute", "artifact.manage"]

logger = logging.getLogger(__name__)

# The most of auth's error code or detail a permission_unavailable line keeps.
AUTH_REASON_MAX_CHARS = 200
AUTHORIZE_ROUTE = "/internal/permissions/authorize/"


class PermissionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed: StrictBool


class ProductPermissionDenied(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            403,
            detail={
                "code": "permission_denied",
                "message": "Your current role does not allow this action.",
            },
            headers={"X-MindsHub-Reason": "permission_denied"},
        )


@dataclass(frozen=True)
class UnavailableCause:
    """Why an auth call left a permission unverified.

    ``route`` is auth's route, then auth's HTTP ``status_code`` with its error
    code or detail as ``reason``, the exception class as ``error_type``, or a
    fixed ``reason`` for a check this server made. Never a header, a request
    body or a mint answer, which holds the key.
    """

    route: str
    status_code: int | None = None
    reason: str | None = None
    error_type: str | None = None

    def log_fields(self) -> dict[str, str | int]:
        fields: dict[str, str | int] = {"event": "permission_unavailable", "route": self.route}
        for name, value in (("status_code", self.status_code), ("reason", self.reason),
                            ("error_type", self.error_type)):
            if value is not None:
                fields[name] = value
        return fields


class ProductPermissionUnavailable(HTTPException):
    def __init__(self, auth_cause: UnavailableCause | None = None) -> None:
        super().__init__(
            503,
            detail={
                "code": "permission_unavailable",
                "message": "Your current permissions could not be verified. Try again.",
            },
        )
        #: Why auth left the permission unverified; None for another service's answer.
        self.auth_cause = auth_cause
        #: Set once report_permission_unavailable has logged it.
        self.reported = False


def auth_refusal_reason(*, response: httpx.Response) -> str | None:
    """auth's error ``code``, else its ``detail`` text, cut to AUTH_REASON_MAX_CHARS.

    None when the body is not JSON or names neither as a string.
    """
    try:
        body = response.json()
    except (ValueError, RecursionError):
        return None
    if not isinstance(body, dict):
        return None
    for field in ("code", "detail"):
        value = body.get(field)
        if isinstance(value, str) and value:
            return value[:AUTH_REASON_MAX_CHARS]
    return None


def permission_unavailable(*, route: str, cause: BaseException | str) -> ProductPermissionUnavailable:
    """The 503 for the caller to raise, carrying why auth left a permission unverified.

    ``cause`` is what the call raised, or a fixed phrase for a check this
    server made, such as ``not_configured``. Nothing is logged here: a caller
    may catch it and go on without what auth couldn't confirm. The place that
    answers it logs it (report_permission_unavailable).
    """
    if isinstance(cause, str):
        return ProductPermissionUnavailable(UnavailableCause(route=route, reason=cause))
    if isinstance(cause, httpx.HTTPStatusError):
        return ProductPermissionUnavailable(UnavailableCause(
            route=route, status_code=cause.response.status_code,
            reason=auth_refusal_reason(response=cause.response),
        ))
    return ProductPermissionUnavailable(UnavailableCause(route=route, error_type=type(cause).__name__))


def report_permission_unavailable(exc: ProductPermissionUnavailable) -> None:
    """Log one WARNING for a permission_unavailable that reaches a person.

    Called where it is answered: the HTTP 503 (cowork.server), a turn's
    turn_failed or its admission check (cowork.turnqueue.producer), and a
    scheduled run's error (cowork.scheduler). Once per exception, however
    many of those it passes. An exception from another service's answer has
    no auth cause and logs nothing here.
    """
    if exc.reported or exc.auth_cause is None:
        return
    exc.reported = True
    logger.warning("auth could not verify a permission", extra=exc.auth_cause.log_fields())


async def has_product_permission(
    scope: TenantScope, permission: ProductPermission
) -> bool:
    """Ask auth for a live decision; an unavailable answer is never a denial.

    Only server-configured internal credentials leave this service. Principal
    identity comes from the gateway or the stored schedule, never a body field.
    Desktop has no organization authority and keeps its single-user boundary.
    """
    if not scope.org_mode:
        return True
    if not scope.org_id or not scope.user_id:
        raise ProductPermissionDenied()
    settings = TurnQueueSettings()
    if not settings.auth_internal_base_url or not settings.auth_internal_secret:
        raise permission_unavailable(route=AUTHORIZE_ROUTE, cause="not_configured")
    try:
        async with asyncio.timeout(5.0):
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
                response = await client.post(
                    f"{settings.auth_internal_base_url.rstrip('/')}{AUTHORIZE_ROUTE}",
                    headers={"X-Internal-Auth": settings.auth_internal_secret},
                    json={
                        "user_id": scope.user_id,
                        "organization_id": scope.org_id,
                        "permission": permission,
                    },
                )
                response.raise_for_status()
                return PermissionDecision.model_validate(response.json()).allowed
    except (httpx.HTTPError, TimeoutError, ValidationError, ValueError) as exc:
        raise permission_unavailable(route=AUTHORIZE_ROUTE, cause=exc) from exc


async def require_product_permission(
    scope: TenantScope, permission: ProductPermission
) -> None:
    if not await has_product_permission(scope, permission):
        raise ProductPermissionDenied()
