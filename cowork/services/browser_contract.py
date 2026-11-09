"""What the agent's browser agrees on with MindsHub, the browser router and anton.

Each name here has a mirror, named next to it. MindsHub's
``GET /v1/entitlements/me/`` answers the ``browser`` block, and the browser
router reads the same block with the same caller's credential, under the same
names. anton's browser tool reads the router's error codes. Change a name here
and in its mirrors together.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool

#: A browser session id: one task's tabs inside the person's browser. The
#: browser router and the desktop app's browser accept the same pattern under
#: the name ``SESSION_ID``, and anton's ``BrowserConfig`` takes it as
#: ``profile`` (tests/test_browser.py pins it against anton's).
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: The key the browser entitlement sits under on MindsHub's
#: ``GET /v1/entitlements/me/``, held under this name there too.
BROWSER_ENTITLEMENT_KEY = "browser"

#: The code the browser router refuses a caller with when their
#: ``BrowserEntitlement`` is off, and the ``/browse/status`` status this server
#: answers for them. anton maps the router's code to its own reason.
NOT_ENTITLED = "not_entitled"

#: The ``/browse/status`` status for a caller whose organization's admin
#: turned the browser off: MindsHub's ``disabled_reason`` of the same name.
ORGANIZATION_OFF = "organization_off"

#: The header that carries ``scope_key`` on the router's ``_scope``, beside the
#: turn key: the router scopes a turn key only for a caller holding it, and
#: only this server does.
SCOPE_KEY_HEADER = "X-Browser-Scope-Key"


class BrowserDisabledReason(StrEnum):
    """Why ``enabled`` is false; null on the wire when it is true."""

    #: The caller does not hold ``browser.use`` in the active organization.
    NOT_PERMITTED = "not_permitted"
    #: An admin of the organization turned the browser off for everyone in it.
    #: It also reaches a MindsHub-signed-in desktop, which ``not_permitted``
    #: does not.
    ORGANIZATION_OFF = "organization_off"


class BrowserEntitlement(BaseModel):
    """The ``browser`` block of MindsHub's ``GET /v1/entitlements/me/``: whether
    the caller may use the browser on the web, and how many bytes its traffic
    may carry for them in a day. This server reads ``enabled``; the router
    enforces both."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    enabled: StrictBool
    daily_bytes: int = Field(ge=0)
    disabled_reason: BrowserDisabledReason | None = None

    @classmethod
    def for_caller(cls, *, permitted: bool, daily_bytes: int) -> Self:
        """The block for a caller whose standing was read: permitted or not."""
        return cls(
            enabled=permitted,
            daily_bytes=daily_bytes,
            disabled_reason=None if permitted else BrowserDisabledReason.NOT_PERMITTED,
        )
