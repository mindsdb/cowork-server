"""Remote turn producer: enqueue a whole-turn job, tail the reply stream,
and emit SSE frames for the turn.

Emits `response.created`, one `response.output_text.delta` per `turn_delta`
reply, and closes the buffer only on a terminal reply (`turn_completed` or
`turn_failed`) with a final `response.completed` frame. No S3, no
heartbeat. Wired into the responses handler in a later task.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from cowork.build_info import KEY_ANTON_VERSION, build_trace_metadata, surface
from cowork.handlers.turn_errors import remote_turn_error
from cowork.services.providers import minds_chat_base_url
from cowork.turnqueue.auth_keys import list_active_connections, mint_turn_key
from cowork.turnqueue.models import TurnJob, TurnReply
from cowork.streaming.turn_index import record_turn
from cowork.turnqueue.redis_client import get_redis
from cowork.common.settings.app_settings import TurnQueueSettings, default_turn_minds_api_host

logger = logging.getLogger(__name__)

# The pod reads the request as one line and truncates at 10 MiB (anton's
# MAX_REQUEST_BYTES); a truncated line won't parse. Margin covers the newline +
# controller-added fields.
_MAX_REQUEST_BYTES = 10 * 1024 * 1024
_REQUEST_BYTES_MARGIN = 64 * 1024


def _trace_block() -> dict[str, str]:
    """Attribution the pod cannot work out for itself, for the job's params.

    Kept tiny and total-failure-tolerant: telemetry must never be the reason a
    turn does not start, and an absent block reads as "no attribution" on the
    pod side rather than an error. That is also what lets the three repos in
    this chain deploy in any order.
    """
    try:
        resolved = surface()
        block = build_trace_metadata({"surface": resolved} if resolved else None)
        # Drop OUR anton version: the pod runs a different anton entirely (its
        # own pinned `minds-anton-scratchpad` image, bumped independently of
        # this server's vendored dep), so the value would be wrong on the wire.
        # It is harmless today only because anton overwrites it when building
        # its headers — shipping a knowingly-wrong value and relying on a
        # downstream overwrite is a trap for whoever touches that overwrite.
        # ENG-1279 sends it in-process as a fallback for antons too old to
        # self-report; the pod image is never that old.
        block.pop(KEY_ANTON_VERSION, None)
        return block
    except Exception:  # pragma: no cover - defensive: never block a turn
        logger.warning("could not build the turn's trace attribution", exc_info=True)
        return {}


def _request_wire_size(params: dict) -> int:
    """Bytes the controller's request line will occupy for these params."""
    return len(json.dumps(params, separators=(",", ":")).encode("utf-8"))


def _fit_request(params: dict, conversation_id: str) -> dict:
    """Warn when the request line will not fit the pod's stdin cap.

    This used to shed ``skills`` then ``memory``, which were re-sent every turn
    and degraded gracefully. Both now live on the shared mount and never enter
    the payload, so there is nothing optional left to drop: what remains is
    input, model, llm and history, and none of them can be silently discarded
    without changing the turn's meaning.

    So this no longer trims, it reports. An oversized line is a real problem
    (the pod's readline will truncate it) and history is the only thing that
    grows unboundedly, so the fix belongs in history windowing upstream, not in
    a silent drop here.
    """
    budget = _MAX_REQUEST_BYTES - _REQUEST_BYTES_MARGIN
    size = _request_wire_size(params)
    if size > budget:
        logger.warning(
            "[producer] turn %s request line is %d bytes, over the %d-byte cap; "
            "nothing is sheddable now that skills and memory read off the shared mount, "
            "so this needs history windowing upstream",
            conversation_id, size, budget,
        )
    return params


def _new_correlation_id() -> str:
    return str(uuid.uuid4())


async def _mint_llm_block(*, org_id: str | None, user_id: str | None,
                          correlation_id: str, settings: TurnQueueSettings) -> dict:
    """Mint a short-TTL MindsHub turn key and build the job's `llm` block.

    The mint call is authenticated with the internal shared secret
    (`X-Internal-Auth`) only - there is no per-tenant credential to look up or
    send. `org_id`/`user_id` (the request principal's identity) tell auth
    which tenant the key is scoped to; auth resolves them itself, so no
    per-user provider key is needed or read here.
    """
    api_key = await mint_turn_key(
        user_id=user_id, org_id=org_id, correlation_id=correlation_id,
        ttl_seconds=settings.turn_key_ttl_seconds, settings=settings,
    )
    base_url = settings.minds_base_url or minds_chat_base_url(default_turn_minds_api_host())
    block = {"provider": "minds-cloud", "api_key": api_key, "base_url": base_url}
    # Must be a MINDS alias (the pod runs on minds-cloud). Default to the
    # free-bucket model: a fresh org has no wallet balance, so premium 402s.
    from cowork.common.settings.app_settings import MINDS_FREE_MODEL
    coding_model = settings.minds_coding_model or MINDS_FREE_MODEL
    if coding_model:
        block["coding_model"] = coding_model
    return block


async def _mint_oauth_block(*, org_id: str | None, user_id: str | None,
                            disabled: list[dict] | None,
                            settings: TurnQueueSettings) -> dict | None:
    """Build everything the job's `oauth` block needs except the turn key —
    the list of connections anton is allowed to use this turn. Deliberately
    doesn't take the turn key `_mint_llm_block` mints: this function's own
    network call (listing active connections) never uses it, only the final
    block does, so the caller runs the two mints concurrently and folds the
    turn key in afterward instead of sequencing this behind the llm mint.

    `disabled` mirrors the desktop path's `disabled_connections` — the same
    general per-conversation concept (`cowork/schemas/conversations.py`),
    just not previously threaded into this remote-producer path.

    None when there's nothing to offer this turn: no org context (local/
    desktop callers never reach this path), auth has no active connections
    for this org, or every connection is disabled — an absent block is the
    pod's existing signal for "no connector tokens available," same as
    before this feature existed.
    """
    if not org_id or not user_id:
        return None
    try:
        connections = await list_active_connections(org_id=org_id, user_id=user_id, settings=settings)
    except Exception:
        # Never block a turn over connector availability — see _trace_block's
        # identical reasoning above. A turn that can't reach the connections
        # list still runs, just without connector tools this time.
        logger.warning(
            "[producer] could not list active connections for org %s; oauth block omitted",
            org_id, exc_info=True,
        )
        return None
    if disabled:
        disabled_keys = {(d.get("engine"), d.get("name")) for d in disabled}
        connections = [c for c in connections if (c.get("engine"), c.get("name")) not in disabled_keys]
    if not connections:
        return None
    return {
        "base_url": settings.auth_internal_base_url,
        "connections": connections,
    }


# What the reply loop reports when the worker stops answering. Shaped like the
# pod's own scrubbed "ExceptionType: message" errors so remote_turn_error can
# classify it (today: the generic redacted message and the generic error card).
UNRESPONSIVE_WORKER_ERROR = "TurnWorkerUnresponsive: the turn worker stopped responding"


def step_stream_events(data: dict) -> list:
    """Reconstruct anton Stream* events from one `turn_step` reply.

    The output feeds the same `format_responses_stream` the in-process path
    uses, so remote turns render steps/thinking identically to desktop."""
    from anton.core.llm.provider import (
        LLMResponse,
        StreamComplete,
        StreamContextCompacted,
        StreamTaskProgress,
        StreamToolResult,
        StreamToolUseDelta,
        StreamToolUseEnd,
        StreamToolUseStart,
        ToolCall,
    )

    step = data.get("step")
    if step == "tool_start":
        return [StreamToolUseStart(id=data.get("id") or "", name=data.get("name") or "")]
    if step == "tool_end":
        tool_id = data.get("id") or ""
        args = data.get("args") or ""
        events: list = []
        if args:
            # The pod accumulated the args deltas; replay them as one delta so
            # the formatter's join produces the same payload as in-process.
            events.append(StreamToolUseDelta(id=tool_id, json_delta=args))
        events.append(StreamToolUseEnd(id=tool_id))
        return events
    if step == "progress":
        return [StreamTaskProgress(
            phase=data.get("phase") or "",
            message=data.get("message") or "",
            eta_seconds=data.get("eta_seconds"),
            id=data.get("id"),
            ok=data.get("ok"),
        )]
    if step == "tool_result":
        return [StreamToolResult(
            name=data.get("name") or "",
            content=data.get("content") or "",
            action=data.get("action"),
            id=data.get("id"),
        )]
    if step == "compacted":
        return [StreamContextCompacted(message=data.get("message") or "")]
    if step == "round_end":
        # Only stop_reason and tool_calls truthiness drive the formatter's
        # round-break decision; a placeholder call carries the truthiness.
        calls = [ToolCall(id="", name="", input={})] if data.get("had_tool_calls") else []
        return [StreamComplete(response=LLMResponse(
            content="", tool_calls=calls, stop_reason=data.get("stop_reason"),
        ))]
    return []


async def stream_remote_replies(*, conversation_id: str, org_id: str | None,
                                user_id: str | None, input_text: str,
                                model: str | None,
                                turn_id: int = 0,
                                history: list | None = None,
                                project_id: str | None = None,
                                workspace_rel_path: str = "projects/general",
                                correlation_id: str | None = None,
                                llm: dict | None = None,
                                disabled: list[dict] | None = None):
    """Mint, enqueue, then yield this turn's replies as (kind, data) tuples.

    Yields turn_delta / turn_step / turn_memory in arrival order and ends with
    exactly one terminal — turn_completed, or turn_failed (classified with the
    same (code, message) the caller streams and persists; synthesized locally
    when the worker goes quiet past the idle timeout).
    `correlation_id`/`llm` reuse a turn key the routing gate already minted."""
    settings = TurnQueueSettings()
    r = get_redis()
    corr = correlation_id or _new_correlation_id()
    # A flag left by an earlier turn would cancel this one on its first line.
    await r.delete(f"cowork:cancel:{corr}")
    reply_stream = f"scratchpad:reply:{conversation_id}"

    # No client-picked model → the deployment's resolved default (org mode: the
    # free-bucket model). Resolved here so the model reaching the pod is always
    # a valid minds alias, independent of any harness's built-in default.
    if not model:
        from cowork.common.settings.user_settings import get_user_settings
        from cowork.db.scoped import TenantScope
        scope = TenantScope(org_mode=bool(org_id), org_id=org_id, user_id=user_id)
        model = get_user_settings(scope).resolved_planning_model

    # Independent network round trips (llm's turn-key mint, oauth's active-
    # connections list) — run concurrently rather than paying both latencies
    # in sequence on every turn's hot path. Skipped for llm when a turn key
    # was already minted upstream (the `llm` param), since there's nothing
    # to overlap with in that case.
    oauth_connections_coro = _mint_oauth_block(
        org_id=org_id, user_id=user_id, disabled=disabled, settings=settings,
    )
    if llm:
        llm_block = llm
        oauth_connections = await oauth_connections_coro
    else:
        llm_block, oauth_connections = await asyncio.gather(
            _mint_llm_block(org_id=org_id, user_id=user_id, correlation_id=corr, settings=settings),
            oauth_connections_coro,
        )
    # Reuses the turn key already minted for llm_block — never mints a
    # second one. Org/cloud mode only: local-mode turns build in-process
    # and never reach this function at all.
    oauth_block = {**oauth_connections, "turn_key": llm_block.get("api_key", "")} if oauth_connections else None

    # Org-relative, never absolute. cowork-server sees the shared tree at
    # <root>/<org_id> while the pod mounts its own org's access point AT
    # <root>, so the two sit at different depths and an absolute path from
    # here is wrong inside the pod. Each side joins its own root.
    #
    # Skills and memory are no longer shipped: the pod reads them off the same
    # mount, which also removes most of what _fit_request exists to trim.
    params = {"input": input_text, "workspace_path": workspace_rel_path.lstrip("/"),
              "model": model, "history": history or [], "llm": llm_block,
              # Absent entirely (not an empty dict) when there's nothing to
              # offer — see _mint_oauth_block's docstring for why.
              **({"oauth": oauth_block} if oauth_block else {}),
              # Trace attribution for the pod (ENG-1459). The remote turn runs in
              # a scratchpad pod that has no cowork-server installed, so nothing
              # there can derive the surface, this server's version, or its
              # install channel — measured on prod, 0 of 68 cloud traces carried
              # any of them. Same helper the in-process path uses, so the two
              # cannot drift. Observability only; the pod must never act on it.
              "trace": _trace_block()}
    params = _fit_request(params, conversation_id)

    job = TurnJob(
        op="anton_turn",
        conversation_id=conversation_id,
        correlation_id=corr,
        reply_stream=reply_stream,
        organization_id=org_id,
        user_id=user_id,
        project_id=project_id,
        params=params,
    )
    # Registry first: a conversation whose stream exists but isn't registered would
    # be invisible to the controller. The reverse is harmless, it prunes empty queues.
    await r.sadd(f"{settings.jobs_stream}:queues", conversation_id)
    await r.xadd(f"{settings.jobs_stream}:{conversation_id}", {"payload": job.model_dump_json()})
    # Any replica can now find this turn: its turn_id to open the buffer, its
    # correlation_id to cancel it, its org to authorize the caller. Recorded
    # here rather than after the first buffer record, because this generator
    # does not own the buffer; _shared_turn covers the gap between the two.
    await record_turn(
        conversation_id, turn_id=turn_id, correlation_id=corr,
        org_id=org_id, user_id=user_id, client=r,
    )

    last_id = "0-0"
    idle_timeout = settings.reply_idle_timeout_seconds
    last_reply_at = time.monotonic()
    while True:
        resp = await r.xread({reply_stream: last_id}, count=10, block=5000)
        if not resp:
            # Unbounded, this loop spun forever whenever the worker was down:
            # only a terminal reply ends the turn, so the SSE response never
            # ended. Bound the wait and fail the turn.
            if idle_timeout > 0 and time.monotonic() - last_reply_at > idle_timeout:
                code, message = remote_turn_error(UNRESPONSIVE_WORKER_ERROR)
                logger.warning(
                    "Remote turn abandoned: no reply for %.0fs conversation=%s correlation_id=%s",
                    idle_timeout, conversation_id, corr,
                )
                yield "turn_failed", {"error": UNRESPONSIVE_WORKER_ERROR,
                                      "code": code, "message": message}
                return
            continue
        for _stream, entries in resp:
            for entry_id, fields in entries:
                last_id = entry_id
                reply = TurnReply.model_validate_json(fields["payload"])
                if reply.correlation_id != corr:
                    # Deliberately does NOT refresh the idle clock: liveness
                    # means "this turn is progressing", and another turn's
                    # replies say nothing about ours.
                    continue
                last_reply_at = time.monotonic()
                kind = reply.kind
                data = reply.data or {}
                if kind == "turn_failed":
                    # Classify once; the SSE frame and the persisted events
                    # log must carry the same (code, message).
                    code, message = remote_turn_error(data.get("error"))
                    data = {**data, "code": code, "message": message}
                    logger.warning(
                        "Remote turn failed conversation=%s correlation_id=%s error=%s",
                        conversation_id, corr, data.get("error"),
                    )
                if kind in ("turn_delta", "turn_step", "turn_memory", "turn_skill",
                            "turn_history", "turn_completed", "turn_failed"):
                    yield kind, data
                if kind in ("turn_completed", "turn_failed"):
                    return
