"""Runtime hand-over of the MindsHub credential from the desktop app.

Write-only on purpose. The desktop app is the only caller and it already holds
the value it is sending, so there is nothing to read back and no route offers
it. ``GET /settings/reveal-key/minds`` stays the one place a local caller can
read the resolved credential, under the same loopback guard it always had.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from cowork.api.v1.endpoints.guards import require_local, require_local_tenancy
from cowork.common.settings.runtime_credential import (
    clear_minds_credential,
    set_minds_credential,
)

# Both guards sit on the router rather than on the route, so a second route
# added here inherits them instead of having to remember them. Loopback because
# this accepts a bearer token and a network-exposed deployment must not let a
# remote peer choose which credential the agent spends; desktop-only because an
# org pod is handed a per-turn credential and has no use for a stored one.
router = APIRouter(dependencies=[Depends(require_local), Depends(require_local_tenancy)])


class MindsCredentialBody(BaseModel):
    """The credential being handed over.

    Blank clears it, which is what sign-out sends. Modelled rather than read
    off a raw dict so the shape the desktop app writes against is declared in
    one place and validated before it reaches the holder.
    """

    value: str = ""


@router.put("/minds")
def put_minds_credential(body: MindsCredentialBody) -> dict[str, bool]:
    if body.value:
        set_minds_credential(body.value)
    else:
        clear_minds_credential()
    return {"ok": True}
