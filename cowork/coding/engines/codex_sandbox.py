from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai_codex.client import CodexClient


def windows_sandbox_override(client: CodexClient) -> tuple[str, ...]:
    """Enable the no-admin sandbox only when no Windows sandbox is configured.

    Without one, Codex rejects workspace patches even when their paths are
    writable. Ask Codex for its effective configuration so existing elevated
    setups and managed requirements are respected, including stale setups
    which must be repaired rather than downgraded. The override is process-local;
    Codex still validates it against managed requirements at the next launch.
    """
    if sys.platform != "win32":
        return ()

    from openai_codex.generated.v2_all import (
        WindowsSandboxReadiness,
        WindowsSandboxReadinessResponse,
    )

    readiness = client.request(
        "windowsSandbox/readiness", {}, response_model=WindowsSandboxReadinessResponse,
    )
    if readiness.status == WindowsSandboxReadiness.not_configured:
        return ('windows.sandbox="unelevated"',)
    return ()
