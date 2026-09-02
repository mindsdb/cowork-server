"""Versioned capability contracts shared with canonical Cowork web."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field

from cowork.common.settings.app_settings import get_app_settings
from cowork.principal import Principal, get_principal

router = APIRouter()
PrincipalDep = Annotated[Principal | None, Depends(get_principal)]


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
    response: Response, principal: PrincipalDep
) -> OrganizationSwitchCapability:
    """Advertise switching only after every request is fenced by organization."""
    response.headers["Cache-Control"] = "no-store"
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"Cache-Control": "no-store"},
        )

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
