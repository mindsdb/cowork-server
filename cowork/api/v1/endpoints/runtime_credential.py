"""Runtime hand-overs from the desktop app: the MindsHub credential, and the
address and launch key of the app's own browser.

Write-only on purpose. The desktop app is the only caller and it already holds
the values it is sending, so there is nothing to read back and no route offers
them. ``GET /settings/reveal-key/minds`` stays the one place a local caller can
read the resolved credential, under the same loopback guard it always had.
"""

import re
from typing import Self

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator, model_validator

from cowork.api.v1.permissions import LoopbackDesktopOnly, require
from cowork.common.settings.runtime_credential import (
    LocalBrowserHost,
    clear_browser_host,
    clear_minds_credential,
    set_browser_host,
    set_minds_credential,
)

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


@router.put("/minds")
def put_minds_credential(body: MindsCredentialBody) -> dict[str, bool]:
    if body.value:
        set_minds_credential(body.value)
    else:
        clear_minds_credential()
    return {"ok": True}


#: Where the app's browser listens: loopback, by address rather than by name,
#: since anton's browser tool takes exactly this form
#: (anton.core.browser.config.BrowserConfig).
_BROWSER_URL = re.compile(r"http://127\.0\.0\.1:(?P<port>[1-9][0-9]{0,4})")
#: The app's per-launch key: 32 random bytes, base64url without padding.
_BROWSER_KEY = re.compile(r"[A-Za-z0-9_-]{43}")


class BrowserHostBody(BaseModel):
    """The desktop app's browser being handed over: ``url`` and ``token``
    together, or both blank to clear it, which the app sends when its browser
    stops. A malformed body answers 422 and the error never echoes the key."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    url: str = ""
    token: SecretStr = SecretStr("")

    @field_validator("url")
    @classmethod
    def _loopback_port(cls, value: str) -> str:
        match = _BROWSER_URL.fullmatch(value)
        if value and (match is None or not 1 <= int(match["port"]) <= 65535):
            raise ValueError("url must be http://127.0.0.1:<port>")
        return value

    @field_validator("token")
    @classmethod
    def _launch_key(cls, value: SecretStr) -> SecretStr:
        secret = value.get_secret_value()
        if secret and not _BROWSER_KEY.fullmatch(secret):
            raise ValueError("token must be 43 base64url characters")
        return value

    @model_validator(mode="after")
    def _both_or_neither(self) -> Self:
        if bool(self.url) != bool(self.token.get_secret_value()):
            raise ValueError("url and token go together: both set, or both blank to clear")
        return self


@router.put("/browser")
async def put_browser_host(body: BrowserHostBody) -> dict[str, bool]:
    if body.url:
        set_browser_host(LocalBrowserHost(url=body.url, token=body.token))
        # Read the organization's switch now, so a MindsHub outage at the
        # first turn keeps this answer rather than offering the browser.
        from cowork.services.browser import desktop_organization_off

        await desktop_organization_off()
    else:
        clear_browser_host()
    return {"ok": True}
