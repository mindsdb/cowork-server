"""Runtime hand-over of the MindsHub credential from the desktop app.

Write-only on purpose. The desktop app is the only caller and it already holds
the value it is sending, so there is nothing to read back and no route offers
it. ``GET /settings/reveal-key/minds`` stays the one place a local caller can
read the resolved credential, under the same loopback guard it always had.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from cowork.api.v1.permissions import LoopbackDesktopOnly, require
from cowork.common.settings import runtime_credential
from cowork.common.settings.runtime_credential import (
    clear_minds_credential,
    set_minds_credential,
)
from cowork.services.inference_key import refresh_inference_key, revoke_inference_key

# Both guards sit on the router rather than on the route, so a second route
# added here inherits them instead of having to remember them. Loopback because
# this accepts a bearer token and a network-exposed deployment must not let a
# remote peer choose which credential the agent spends; desktop-only because an
# org pod is handed a per-turn credential and has no use for a stored one.
router = APIRouter(dependencies=[Depends(require(LoopbackDesktopOnly))])


class MindsCredentialBody(BaseModel):
    """The credential being handed over.

    Blank clears it, which is what sign-out sends. Modelled rather than read
    off a raw dict so the shape the desktop app writes against is declared in
    one place and validated before it reaches the holder.
    """

    value: str = ""
    # The organization the desktop has mounted. When sent, LLM calls bill a
    # turn key pinned to it rather than the session token.
    organization_id: str | None = None


@router.put("/minds")
def put_minds_credential(body: MindsCredentialBody) -> dict[str, bool]:
    if not body.value:
        key, token = (
            runtime_credential.get_inference_key(),
            runtime_credential.get_minds_credential(),
        )
        clear_minds_credential()
        if key and token:
            revoke_inference_key(token, key)
        return {"ok": True}

    set_minds_credential(body.value)
    # A hand-entered mdb_ key already names its organization, and auth won't mint with one.
    organization_id = None if body.value.startswith("mdb_") else body.organization_id
    runtime_credential.set_organization(organization_id)
    if organization_id:
        refresh_inference_key(body.value, organization_id)
    return {"ok": True}
