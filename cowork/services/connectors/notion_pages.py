"""Notion pages for the chat's "Add pages from Notion" picker: searching a
workspace over Notion MCP, and describing a project's added pages to the agent."""
from __future__ import annotations

import json
import logging
from typing import Any

_log = logging.getLogger("cowork.connectors.notion_pages")

MAX_RESULTS = 20


class NotionSearchUnavailable(Exception):
    """The installed anton cannot talk to Notion MCP at all."""


class NotionReconnectRequired(Exception):
    """Notion rejected the stored token; the user has to reconnect."""


def _decode(content: Any) -> Any:
    # call_mcp_tool returns the tool's text joined into one string, or a list
    # of MCP content blocks; Notion puts its JSON in the text.
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    if isinstance(content, str):
        try:
            return json.loads(content)
        except ValueError:
            return None
    return content


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_search_results(content: Any) -> list[dict[str, str]]:
    """Pages and databases from a `notion-search` result. The response shape
    isn't documented, so unknown fields are ignored and anything without an
    id and a URL is dropped rather than shown as an unopenable row."""
    data = _decode(content)
    items = data.get("results") if isinstance(data, dict) else data
    if not isinstance(items, list):
        _log.warning("notion-search returned an unrecognised shape: %.300r", content)
        return []
    pages: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        page_id, url = _text(item.get("id")), _text(item.get("url"))
        kind = _text(item.get("type")) or "page"
        if not page_id or not url or kind == "user":
            continue
        pages.append({
            "id": page_id,
            "title": _text(item.get("title")) or _text(item.get("name")) or "Untitled",
            "url": url,
            "type": kind,
            "timestamp": _text(item.get("timestamp")),
        })
        if len(pages) >= MAX_RESULTS:
            break
    return pages


async def search_pages(access_token: str, query: str) -> list[dict[str, str]]:
    try:
        from anton.core.mcp.errors import McpPermanentError
        from anton.core.mcp.servers import mcp_server_url
        from anton.core.mcp.wiring import call_mcp_tool
    except ImportError as exc:
        raise NotionSearchUnavailable("installed anton has no MCP client") from exc
    if mcp_server_url("notion") is None:
        raise NotionSearchUnavailable("installed anton has no Notion MCP server URL")
    try:
        content = await call_mcp_tool("notion", access_token, "notion-search", query=query)
    except McpPermanentError as exc:
        raise NotionReconnectRequired(str(exc)) from exc
    return parse_search_results(content)


def describe_project_pages(pages_by_connection: dict[str, list[dict]]) -> str:
    """System-prompt guidance naming the Notion pages added to the current
    project, so the agent can read them in any later turn, not only in the
    message they were added with. Empty when there are none."""
    lines = [
        f"- {page.get('name') or 'Untitled'} (url: {page.get('url') or page.get('id')}, connection: {conn_name})"
        for conn_name, pages in pages_by_connection.items()
        for page in pages
        if isinstance(page, dict)
    ]
    if not lines:
        return ""
    return (
        "\n\nNotion pages the user added to this project's files:\n"
        + "\n".join(lines)
        + "\nWhen the user refers to the project's files, documents or pages, these are included. "
        "Read one with the Notion connection's notion-fetch tool, passing its URL; do not search for it first."
    )
