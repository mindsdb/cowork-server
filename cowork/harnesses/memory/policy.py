"""Whether learned memory is on, and whose it is, on a deployment that trusts
identity headers.

Two settings the operator already has decide it there. MEMORY_ENABLED, the
``memory_enabled`` setting, turns learned memory on or off; it is
operator-pinned, so a stored global row wins over the environment.
COWORK_PERSONAL_RESOURCES keeps memory per person when it lists ``memory``.
Left off, the deployment's own tiers are one store that every person's turns
read and learn into, profile included. Switching either moves nothing: each
person's tiers and the shared store are separate folders.

With memory off, in either scope, no turn reads or writes learned memory, and
what is stored stays as it is, unread, until memory goes back on. Each turn
gets two empty tiers (cowork.harnesses.memory.turn_layout.empty_turn_memory)
and still gets the operator's managed rules, the Memories listing is empty
(MemoryService.list_memory_sync), a read, change or delete of one slot
through MemoryService answers 409 (MEMORY_OFF_KEPT), and the operator's clear
refuses (cowork.ops.memory).

The desktop and hosted org mode keep their own behavior: there
``memory_enabled`` only turns anton's memory mode off, and neither field below
is ever true.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from cowork.common.settings.app_settings import get_app_settings
from cowork.common.settings.user_settings import get_user_settings
from cowork.db.personal_resources import PersonalResource
from cowork.db.scoped import TenantScope

# Why stored memory is neither read nor changed while memory is off. The
# memory API's refusals and the operator commands build on it.
MEMORY_OFF_KEPT = (
    "Memory is off (MEMORY_ENABLED is false), so stored memory is kept as it is, "
    "unread, until memory goes back on."
)


class MemoryPolicy(BaseModel):
    """What a deployment that trusts identity headers does with learned memory."""

    model_config = ConfigDict(frozen=True)

    # MEMORY_ENABLED is false: no turn reads or writes learned memory, and the
    # stored memory is kept, unread.
    off: bool
    # COWORK_PERSONAL_RESOURCES leaves out memory: the deployment's own tiers
    # are one store every person shares.
    shared: bool

    def start_warning(self) -> str | None:
        """What the server logs at start where memory is shared, or None."""
        if not self.shared:
            return None
        if self.off:
            return (
                "COWORK_PERSONAL_RESOURCES leaves out memory, so memory is shared, and "
                "MEMORY_ENABLED is off: no turn learns, and the shared store is kept, "
                "unread, until memory goes back on"
            )
        return (
            "COWORK_PERSONAL_RESOURCES leaves out memory, so memory is shared: every "
            "person's turns learn into one store and read it back, profile included"
        )


def memory_shared() -> bool:
    """Whether memory is one store every person shares: identity headers are
    trusted and COWORK_PERSONAL_RESOURCES leaves out memory. It reads the app
    settings alone, never the stored ones."""
    settings = get_app_settings()
    return settings.trust_identity_headers and PersonalResource.MEMORY not in settings.personal_resources


def memory_policy(*, memory_enabled: bool) -> MemoryPolicy:
    """The policy for ``memory_enabled`` as the caller resolved it.

    A turn passes the value its own settings hold. Everything else calls
    resolve_memory_policy, which resolves it the same way.
    """
    return MemoryPolicy(
        off=get_app_settings().trust_identity_headers and not memory_enabled,
        shared=memory_shared(),
    )


def resolve_memory_policy(*, scope: TenantScope | None = None) -> MemoryPolicy:
    """The policy, with ``memory_enabled`` read by get_user_settings for
    ``scope``, the resolution a turn's settings come from (stored rows first,
    then the environment).

    The settings are read only on a deployment that trusts identity headers,
    since neither field can be true anywhere else.
    """
    if not get_app_settings().trust_identity_headers:
        return MemoryPolicy(off=False, shared=False)
    return memory_policy(memory_enabled=get_user_settings(scope).memory_enabled)


def memory_start_warning() -> str | None:
    """What the server logs at start, or None.

    Only shared memory has a warning, so MEMORY_ENABLED, whose resolution
    loads the stored settings, is resolved only where memory is shared.
    """
    if not memory_shared():
        return None
    return resolve_memory_policy().start_warning()
