"""Versioned capability contracts shared with canonical Cowork web."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from cowork.api.v1.permissions import Authenticated, require
from cowork.common.settings.app_settings import get_app_settings
from cowork.principal import Principal, get_principal

router = APIRouter()


class NoStoreAuthenticated(Authenticated):
    """``Authenticated``, with ``Cache-Control: no-store`` on the 401 too.

    ``Authenticated.check()`` raises a bare ``HTTPException`` with no headers.
    This route sets ``no-store`` on every response — including denial, so a
    stale "unauthorized" answer from mid-organization-switch is never served
    from a cache once the caller is fenced again — and FastAPI's exception
    handling never sees a mutated ``Response`` param, only what's attached to
    the raised exception itself, so the header has to be re-attached here.

    Declares its own ``principal: Depends(get_principal)`` (rather than
    letting the ``request``-only call below resolve it) and passes it through
    to ``super().check`` explicitly — ``require()`` copies *this* class's
    ``check`` signature onto the dependency FastAPI resolves, so this is what
    keeps ``app.dependency_overrides[get_principal]`` working here too.
    """

    async def check(
        self, request: Request, principal: Principal | None = Depends(get_principal)
    ) -> Principal:
        try:
            return await super().check(request, principal=principal)
        except HTTPException as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.detail,
                headers={**(exc.headers or {}), "Cache-Control": "no-store"},
            ) from exc


class OrganizationSwitchCapability(BaseModel):
    """Protocol support the web client must verify before listing tenants."""

    model_config = ConfigDict(populate_by_name=True)

    protocol_version: Literal[1] = Field(default=1, alias="protocolVersion")
    expected_organization_enforced: bool = Field(alias="expectedOrganizationEnforced")
    enabled: bool


@router.get(
    "/organization-switch",
    response_model=OrganizationSwitchCapability,
    response_model_by_alias=True,
)
def organization_switch_capability(
    response: Response, principal: Principal = Depends(require(NoStoreAuthenticated))
) -> OrganizationSwitchCapability:
    """Advertise switching only after every request is fenced by organization."""
    response.headers["Cache-Control"] = "no-store"

    settings = get_app_settings()
    boundary_enforced = (
        settings.tenancy_mode == "org"
        and settings.identity_enforce == "enforce"
        and settings.organization_boundary_mode == "enforce"
    )
    return OrganizationSwitchCapability(
        expected_organization_enforced=boundary_enforced,
        enabled=boundary_enforced and settings.organization_switch_enabled,
    )
