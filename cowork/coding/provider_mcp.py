from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import threading
from collections import Counter
from collections.abc import Callable
from contextlib import AsyncExitStack, suppress
from typing import Any

from anton.core.mcp.access import allowed_for_mode
from anton.core.mcp.client import McpSession
from anton.core.mcp.errors import McpPermanentError, classify_mcp_error
from anton.core.mcp.names import namespaced_tool_name
from anton.core.mcp.servers import mcp_server_url

from cowork.coding.project_models import CodeProject, ProjectConnection

SessionFactory = Callable[[str, str], McpSession]

# Codex gives an MCP server about ten seconds to answer its first tools/list.
_OPEN_TIMEOUT_SECONDS = 5.0


class ProviderTools:
    """Forwards the read tools of each attached connection's own hosted MCP server.

    Codex launches this process and sees only tool names and results: the
    connection's credential is resolved and used here, never handed to Codex.
    Only tools the shared Connectors classification marks read are exposed,
    so a tool a provider adds later stays hidden until it is classified.
    """

    def __init__(
        self,
        project: CodeProject,
        token_for: Callable[[ProjectConnection], str | None],
        session_factory: SessionFactory | None = None,
    ) -> None:
        self._project = project
        self._token_for = token_for
        self._session_factory = session_factory or (lambda url, token: McpSession(url=url, access_token=token))
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="provider-mcp", daemon=True)
        self._thread.start()
        self._requests: asyncio.Queue[tuple[McpSession, str, dict[str, Any], concurrent.futures.Future] | None] | None = None
        self._ready: concurrent.futures.Future[list[dict[str, Any]]] | None = None
        self._serving: concurrent.futures.Future[None] | None = None
        self._routes: dict[str, tuple[McpSession, str]] = {}
        self.status: dict[str, str] = {}

    def tools(self) -> list[dict[str, Any]]:
        if self._ready is None:
            self._ready = concurrent.futures.Future()
            self._serving = asyncio.run_coroutine_threadsafe(self._serve(self._ready), self._loop)
        return self._ready.result()

    def owns(self, name: str) -> bool:
        self.tools()
        return name in self._routes

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.tools()
        session, tool = self._routes[name]
        result: concurrent.futures.Future[tuple[str | list[dict], bool]] = concurrent.futures.Future()
        assert self._requests is not None
        self._loop.call_soon_threadsafe(self._requests.put_nowait, (session, tool, arguments, result))
        content, is_error = result.result()
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        return {"content": blocks, "isError": is_error}

    def close(self) -> None:
        if self._serving is not None:
            self.tools()
            assert self._requests is not None
            self._loop.call_soon_threadsafe(self._requests.put_nowait, None)
            # Let the serving task close its sessions before the loop stops.
            with suppress(Exception):
                self._serving.result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    async def _serve(self, ready: concurrent.futures.Future[list[dict[str, Any]]]) -> None:
        # One task owns every session from open to close: the SDK's transports
        # must be exited by the task that entered them.
        self._requests = asyncio.Queue()
        async with AsyncExitStack() as stack:
            listed: list[dict[str, Any]] = []
            try:
                listed = await self._open(stack)
            finally:
                ready.set_result(listed)
            while (request := await self._requests.get()) is not None:
                session, tool, arguments, result = request
                try:
                    result.set_result(await session.call_tool(tool, arguments))
                except Exception as exc:  # noqa: BLE001 - every failure belongs to the caller's tools/call.
                    result.set_exception(exc)

    async def _open(self, stack: AsyncExitStack) -> list[dict[str, Any]]:
        hosted = [item for item in self._project.connections if mcp_server_url(item.provider)]
        per_provider = Counter(item.provider for item in hosted)
        listed: list[dict[str, Any]] = []
        for connection in hosted:
            label = connection.label or connection.name
            try:
                token = self._token_for(connection)
                if token is None:
                    self.status[label] = "No hosted tools for this connection"
                    continue
                session = self._session_factory(mcp_server_url(connection.provider) or "", token)
                async with asyncio.timeout(_OPEN_TIMEOUT_SECONDS):
                    await stack.enter_async_context(session)
                    tools = await session.list_tools()
            except Exception as exc:  # noqa: BLE001 - one dead connection must not hide the others.
                error = classify_mcp_error(exc) if not isinstance(exc, TimeoutError) else exc
                self.status[label] = "Reconnect this connection" if isinstance(error, McpPermanentError) else "Unavailable"
                print(f"{connection.provider} MCP for {label!r} unavailable: {exc}", file=sys.stderr)
                continue
            exposed = 0
            for tool in tools:
                if not allowed_for_mode(connection.provider, tool.name, "read"):
                    continue
                name = namespaced_tool_name(
                    connection.provider,
                    tool.name,
                    connection_name=connection.name if per_provider[connection.provider] > 1 else None,
                )
                self._routes[name] = (session, tool.name)
                listed.append({**tool.model_dump(by_alias=True, exclude_none=True, mode="json"), "name": name})
                exposed += 1
            self.status[label] = f"{exposed} read-only tools"
        return listed
