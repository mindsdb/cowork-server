"""One bounded startup invocation of the configured router, using synthetic input.

The adapter's SDK retries and provider retention policy still apply. This does
not measure the gate's latency budget, create a conversation, or execute a tool.
"""

from __future__ import annotations

import asyncio
import logging

from anton.core.llm.provider import LLMResponse, RequestRefusedError, StreamComplete

from cowork.common.logger import exception_http_status
from cowork.common.settings.user_settings import UserSettings
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, unsafe_unscoped_session
from cowork.db.units import run_db
from cowork.handlers import response_routing as routing
from cowork.services.providers import build_llm_client
from cowork.services.settings import SettingService

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_SECONDS = 30.0
_CLOSE_TIMEOUT_SECONDS = 2.0
_read_tasks: set[asyncio.Task] = set()


def _load_settings(session: ScopedSession) -> UserSettings:
    # SettingService routes settings itself; LOCAL_SCOPE reads global rows only.
    return SettingService(unsafe_unscoped_session(session), LOCAL_SCOPE).load()


def _valid_response(response: LLMResponse) -> bool:
    # The adapters also emit StreamComplete after a cut stream with partial
    # output. A positive terminal reason is required, not just that event.
    if response.stop_reason not in {"completed", "stop", "end_turn", "tool_calls", "tool_use"}:
        return False
    if response.tool_calls:
        return all(
            isinstance(call.id, str)
            and bool(call.id.strip())
            and call.name == routing._DELEGATE_TOOL["name"]
            and not call.parse_error
            and not call.repaired
            and isinstance(call.input, dict)
            and isinstance(call.input.get("reason"), str)
            for call in response.tool_calls
        )
    return isinstance(response.content, str) and bool(response.content.strip())


def _unverified(reason: str, *, provider: str, exc: Exception | None = None) -> None:
    # Never format the exception or its body: SDK errors can contain requests,
    # credentials, provider text, or database values. Values ride as fields,
    # since the log formatter drops a template's values.
    logger.warning(
        "Router startup probe could not verify the router; startup continues. "
        "Recheck the router configuration and the provider's availability",
        extra={
            "event": "router_startup_probe_unverified",
            "outcome": "unverified",
            "reason": reason,
            "provider": provider,
            "error_type": type(exc).__name__ if exc is not None else None,
            "status_code": exception_http_status(exc=exc) if exc is not None else None,
        },
    )


def _cleanup_unverified(error_type: str, status: int | None = None) -> None:
    logger.warning(
        "Router startup probe cleanup did not finish; startup continues",
        extra={
            "event": "router_startup_probe_cleanup_unverified",
            "error_type": error_type,
            "status_code": status,
        },
    )


async def _close(resource) -> None:
    if resource is None:
        return
    try:
        async with asyncio.timeout(_CLOSE_TIMEOUT_SECONDS):
            await resource.aclose()
    except Exception as exc:
        _cleanup_unverified(type(exc).__name__, exception_http_status(exc=exc))


async def _read_response(events) -> LLMResponse | None:
    async for event in events:
        if isinstance(event, StreamComplete):
            return event.response
    return None


def _read_finished(task: asyncio.Task) -> None:
    _read_tasks.discard(task)
    if not task.cancelled():
        task.exception()


async def _stop_read(task: asyncio.Task | None) -> None:
    if task is None:
        return
    try:
        # The first cancellation can enter Anton's stream-finally HTTP close.
        # If that await stalls, cancel again to interrupt the close itself.
        # asyncio.wait bounds each drain without awaiting cancellation forever.
        for _ in range(2):
            if task.done():
                return
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=_CLOSE_TIMEOUT_SECONDS)
            if done:
                return
        _cleanup_unverified("TimeoutError")
    finally:
        if not task.done():
            task.cancel()
        # Code that suppresses repeated cancellation cannot be forced to end.
        # Keep its task tracked until completion, and retrieve its exception.


async def run_router_startup_probe() -> None:
    """Check the resolved gate request once; every unverified result is nonfatal."""
    client = None
    events = None
    reader = None
    provider = "unknown"
    model = None
    failure_reason = "configuration_unavailable"
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _PROBE_TIMEOUT_SECONDS
    try:
        async with asyncio.timeout_at(deadline):
            settings = await run_db(_load_settings, scope=LOCAL_SCOPE)
            provider = settings.resolved_router_provider.value
            model = settings.resolved_gate_model
            if not model:
                _unverified(failure_reason, provider=provider)
                return
            # Keep the whole client so all constructed role providers close.
            # Missing credentials, including hosted per-turn credentials, fail
            # here without creating a request context or minting a turn key.
            client = build_llm_client(settings=settings)
            # The builder can omit a router whose constructor rejects its
            # configuration. Anton then substitutes coding; that cannot verify
            # the router's configured endpoint, transport, or effort.
            if client.router_provider is client.coding_provider:
                _unverified(failure_reason, provider=provider)
                return
            failure_reason = "provider_unavailable"
            events = client.router_provider.stream(
                model=model,
                system=routing.with_deployment_identity(routing._SYSTEM_PROMPT),
                messages=[{"role": "user", "content": "Hello"}],
                tools=[routing._DELEGATE_TOOL],
                max_tokens=routing._DIRECT_MAX_TOKENS,
            )
            reader = asyncio.create_task(_read_response(events), name="router-startup-probe-read")
            _read_tasks.add(reader)
            reader.add_done_callback(_read_finished)
            # Awaiting the reader directly would make our deadline wait for
            # its cancellation cleanup. Keep that drain in the bounded finally.
            done, _ = await asyncio.wait({reader}, timeout=max(0, deadline - loop.time()))
            if not done:
                raise TimeoutError
            response = reader.result()
            if response is not None and _valid_response(response):
                logger.info(
                    "Router startup probe passed",
                    extra={"event": "router_startup_probe_passed", "outcome": "passed", "provider": provider},
                )
            else:
                _unverified("invalid_response", provider=provider)
    except TimeoutError as exc:
        _unverified("timeout", provider=provider, exc=exc)
    except Exception as exc:
        status = exception_http_status(exc=exc)
        if failure_reason == "provider_unavailable" and (
            isinstance(exc, RequestRefusedError)
            or (status is not None and 400 <= status < 500 and status != 429)
        ):
            # Refusal includes authentication and access failures. It does not
            # establish that the API, effort, or model is unsupported.
            failure_reason = "provider_refused"
        _unverified(failure_reason, provider=provider, exc=exc)
    finally:
        try:
            await _stop_read(reader)
        finally:
            try:
                await _close(events)
            finally:
                await _close(client)
