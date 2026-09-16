"""Live organization authority for hosted execution and artifact mutation."""

from __future__ import annotations

import asyncio
from typing import Literal

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError

from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.db.scoped import TenantScope

ProductPermission = Literal["product.execute", "artifact.manage"]


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


class ProductPermissionUnavailable(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            503,
            detail={
                "code": "permission_unavailable",
                "message": "Your current permissions could not be verified. Try again.",
            },
        )


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
        raise ProductPermissionUnavailable()
    try:
        async with asyncio.timeout(5.0):
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
                response = await client.post(
                    f"{settings.auth_internal_base_url.rstrip('/')}/internal/permissions/authorize/",
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
        raise ProductPermissionUnavailable() from exc


async def require_product_permission(
    scope: TenantScope, permission: ProductPermission
) -> None:
    if not await has_product_permission(scope, permission):
        raise ProductPermissionDenied()
