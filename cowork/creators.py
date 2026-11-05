"""Who created a resource people can share, and who may change it.

On a deployment that trusts identity headers (COWORK_TRUST_IDENTITY_HEADERS),
every saved connection and every skill records who created it when it is
created: the person's sub, and nobody for the operator view or a turn with no
person. A later change keeps the stored creator. Connections record it in
cowork.services.connectors.layered_vault.LayeredDataVault, skills in
cowork.services.skills.SkillService.

Where the resource is left off COWORK_PERSONAL_RESOURCES, everyone reads the
deployment's one base store. A person changes only what they created there,
the operator view (a request with no person) changes everything, and a turn
with no person, such as a scheduled run where schedules are shared, changes
nothing that is already there (Changer.may_change, turn_changer_for).
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

    # The person's sub; None for the operator view, for a turn with no person
    # and for a person's partition acting with no user.
    user_id: str | None
    # A request with no person (COWORK_IDENTITY_ENFORCE=audit) or a boot
    # step: never a turn (turn_changer_for).
    operator_view: bool
    # The scope's store is the deployment's base store, which every person
    # reads (the resource left shared): a person changes only what they
    # created there, and the operator view changes everything.
    deployment_wide: bool

    @property
    def creator(self) -> str | None:
        """The creator a resource this changer creates records: the person's
        sub, and nobody for the operator view."""
        return None if self.operator_view else self.user_id

    def may_change(self, creator: str | None) -> bool:
        """Whether this changer may change a resource whose recorded creator
        is ``creator``: any resource of a store that is the scope's own. In
        the deployment's base store, the operator view may change any, a
        person only one whose creator is them, and a changer with no person
        that is not the operator view none."""
        if not self.deployment_wide or self.operator_view:
            return True
        return creator is not None and creator == self.user_id


def changer_for(scope: TenantScope | None, *, resource: PersonalResource) -> Changer | None:
    """Who creates and changes ``resource`` under ``scope`` in a request or an
    operator command, or None where nothing records a creator: with
    COWORK_TRUST_IDENTITY_HEADERS off (the desktop and hosted org tenancy),
    and with no scope. A scope with no person is the operator view."""
    if scope is None or not get_app_settings().trust_identity_headers:
        return None
    return Changer(
        user_id=scope.user_id if scope.personal else None,
        operator_view=not scope.personal,
        deployment_wide=not scope.partitioned_for(resource),
    )


def turn_changer_for(scope: TenantScope | None, *, resource: PersonalResource) -> Changer | None:
    """Who creates and changes ``resource`` in an agent turn under ``scope``:
    the person, as changer_for has it. A turn with no person, such as a
    scheduled run where schedules are shared or a channel's turn, is no
    operator: in the deployment's base store it changes nothing that already
    exists, and what it creates records no creator."""
    changer = changer_for(scope, resource=resource)
    if changer is None or not changer.operator_view:
        return changer
    return Changer(user_id=None, operator_view=False, deployment_wide=changer.deployment_wide)
