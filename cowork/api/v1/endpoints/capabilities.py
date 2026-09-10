"""Versioned capability contracts shared with canonical Cowork web."""

from typing import Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, ConfigDict, Field

from cowork.api.v1.permissions import Authenticated, require

from cowork.common.settings.app_settings import get_app_settings
from cowork.principal import Principal

router = APIRouter()


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
# Authenticated, not AuthenticatedInOrgMode: switching organizations is a
# multi-tenant concept with no desktop counterpart, so a caller with no
# principal has nothing to be told about and gets a 401 in either mode.
def organization_switch_capability(
    response: Response, principal: Principal = Depends(require(Authenticated))
) -> OrganizationSwitchCapability:
    """Advertise switching only after every request is fenced by organization.

    ``Cache-Control: no-store`` on the denial too, so a stale "unauthorized"
    from mid-organization-switch is never replayed from a cache once the
    caller is fenced again. That comes from ``_NoStoreMiddleware``
    (cowork/server.py), which stamps every response under
    ``/api/v1/capabilities`` from outside the route — including the ones
    raised as exceptions, which never see the ``response`` below. Pinned by
    tests/test_no_store_cache.py.
    """
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
