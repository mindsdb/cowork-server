"""Who created a resource people can share, and who may change it.

On a deployment that trusts identity headers (COWORK_TRUST_IDENTITY_HEADERS),
every saved connection and every skill records who created it when it is
created: the person's sub, and nobody for the operator view. A later change
keeps the stored creator. Connections record it in
cowork.services.connectors.layered_vault.LayeredDataVault, skills in
cowork.services.skills.SkillService.

Where the resource is left off COWORK_PERSONAL_RESOURCES, everyone reads the
deployment's one base store, and only a resource's creator changes it there;
one with no creator, made before creators were recorded or by the operator
view, only the operator view changes (Changer.may_change).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from cowork.common.settings.app_settings import get_app_settings

if TYPE_CHECKING:
    from cowork.db.personal_resources import PersonalResource
    from cowork.db.scoped import TenantScope


class Changer(BaseModel):
    """Who creates and changes one kind of resource under one scope, on a
    deployment that trusts identity headers."""

    model_config = ConfigDict(frozen=True)

    # The person's sub; None for the operator view (a request with no person)
    # and for a person's partition acting with no user.
    user_id: str | None
    operator_view: bool
    # The scope's store is the deployment's base store, which every person
    # reads (the resource left shared): only a resource's creator changes it
    # there, and one with no creator only the operator view.
    deployment_wide: bool

    @property
    def creator(self) -> str | None:
        """The creator a resource this changer creates records: the person's
        sub, and nobody for the operator view."""
        return None if self.operator_view else self.user_id

    def may_change(self, creator: str | None) -> bool:
        """Whether this changer may change a resource whose recorded creator
        is ``creator``: any resource of a store that is the scope's own, and
        in the deployment's base store only its creator's, or one with no
        creator for the operator view."""
        if not self.deployment_wide:
            return True
        if self.operator_view:
            return creator is None
        return creator is not None and creator == self.user_id


def changer_for(scope: TenantScope | None, *, resource: PersonalResource) -> Changer | None:
    """Who creates and changes ``resource`` under ``scope``, or None where
    nothing records a creator: with COWORK_TRUST_IDENTITY_HEADERS off (the
    desktop and hosted org tenancy), and with no scope."""
    if scope is None or not get_app_settings().trust_identity_headers:
        return None
    return Changer(
        user_id=scope.user_id if scope.personal else None,
        operator_view=not scope.personal,
        deployment_wide=not scope.partitioned_for(resource),
    )
