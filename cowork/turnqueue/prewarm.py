"""Ask scratchpad-controller for a conversation's pod ahead of its turn.

The controller creates the pod a persistent turn would use, so its start overlaps
the router call, and a delegated turn takes it over. Best effort: a lost request
only means the turn starts its own pod. Contract: scratchpad-controller README,
"Pod prewarm".
"""
from __future__ import annotations

import json

from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.turnqueue.redis_client import get_redis

# Bounds the stream; the controller drops requests older than a few seconds anyway.
_STREAM_MAXLEN = 10_000


async def send_prewarm(*, conversation_id: str, org_id: str, user_id: str | None,
                       project_id: str | None, workspace_rel_path: str) -> None:
    """Carries the ids the turn's job will carry, so both name the same pod, and no credentials."""
    payload = {
        "v": 1,
        "op": "prewarm",
        "conversation_id": conversation_id,
        "organization_id": org_id,
        "user_id": user_id,
        "project_id": project_id,
        "workspace_path": workspace_rel_path.lstrip("/"),
    }
    await get_redis().xadd(
        TurnQueueSettings().prewarm_stream,
        {"payload": json.dumps(payload)},
        maxlen=_STREAM_MAXLEN,
        approximate=True,
    )
