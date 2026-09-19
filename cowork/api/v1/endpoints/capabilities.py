"""Versioned capability contracts shared with canonical Cowork web."""

from typing import Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, ConfigDict, Field

from cowork.api.v1.permissions import Authenticated, require

from cowork.common.settings.app_settings import get_app_settings
from cowork.principal import Principal
from cowork.services.connectors.datasource_capabilities import load_datasource_capabilities

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
        settings.tenancy_mode == "org" and settings.identity_enforce == "enforce"
    )
    return OrganizationSwitchCapability(
        expected_organization_enforced=boundary_enforced,
        enabled=boundary_enforced and settings.organization_switch_enabled,
    )


class DatasourceMethodCapability(BaseModel):
    available: bool


class DatasourceCapability(BaseModel):
    methods: dict[str, DatasourceMethodCapability]


class DatasourceCapabilities(BaseModel):
    """What this deployment may run as a cloud datasource.

    Every method whose spec declares a cloud block appears, available or not,
    so a client can tell "this deployment does not do databases" from "it does
    and this one is switched off".
    """

    manifest_version: int = Field(alias="manifestVersion")
    datasources: dict[str, DatasourceCapability]


@router.get(
    "/datasources",
    response_model=DatasourceCapabilities,
    response_model_by_alias=True,
)
# Authenticated, matching the capability beside it: a caller with no principal
# has nothing to be told about, in either tenancy mode.
def datasource_capabilities(
    response: Response, principal: Principal = Depends(require(Authenticated))
) -> DatasourceCapabilities:
    # Same reason as the capability beside it: what a deployment runs can be
    # switched off between two requests, and a cached "available" would
    # outlive the switch.
    response.headers["Cache-Control"] = "no-store"
    caps = load_datasource_capabilities()
    return DatasourceCapabilities(
        manifestVersion=caps.manifest_version,
        datasources={
            connector_id: DatasourceCapability(
                methods={
                    method: DatasourceMethodCapability(available=available)
                    for method, available in methods.items()
                }
            )
            for connector_id, methods in caps.methods.items()
        },
    )

