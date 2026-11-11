"""Notion pages for the chat's "Add pages from Notion" picker: searching a
workspace over Notion MCP, fetching a picked page's text, and describing a
project's added pages to the agent."""
from __future__ import annotations

import json
import logging
from typing import Any

_log = logging.getLogger("cowork.connectors.notion_pages")

MAX_RESULTS = 20
# A picked page's text rides in the user message and is replayed every turn,
# so a long page is cut here; the agent is told to fetch the rest itself.
MAX_PAGE_CHARS = 20_000


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
        # raw_decode tolerates a note Notion may append after the JSON.
        try:
            return json.JSONDecoder().raw_decode(content.strip())[0]
        except ValueError:
            return None
    return content


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _shape(value: Any) -> str:
    if isinstance(value, dict):
        return f"dict with keys {sorted(map(str, value))[:20]}"
    if isinstance(value, (list, str)):
        return f"{type(value).__name__} of length {len(value)}"
    return type(value).__name__


def parse_search_results(content: Any) -> list[dict[str, str]]:
    """Pages and databases from a `notion-search` result. The response shape
    isn't documented, so unknown fields are ignored and anything without an
    id and a URL is dropped rather than shown as an unopenable row."""
    data = _decode(content)
    items = data.get("results") if isinstance(data, dict) else data
    if not isinstance(items, list):
        # Shape only: the content can hold private page titles.
        _log.warning("notion-search returned an unrecognised shape: %s", _shape(content))
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


def _http_status_error_types() -> tuple[type[BaseException], ...]:
    import httpx

    types: list[type[BaseException]] = [httpx.HTTPStatusError]
    try:
        import httpx2  # anton's MCP client; its errors are not httpx errors
    except ImportError:
        pass
    else:
        types.append(httpx2.HTTPStatusError)
    return tuple(types)


def _token_rejected_type() -> type[BaseException] | None:
    # anton raises this for a 401/403, which the MCP SDK otherwise reports
    # with no HTTP status; older antons don't have it.
    try:
        from anton.core.mcp.errors import McpTokenRejected
    except ImportError:
        return None
    return McpTokenRejected


def is_token_rejection(exc: BaseException) -> bool:
    """True when the MCP server answered 401/403, looking through the
    exception groups, causes and contexts the MCP client wraps it in."""
    status_errors = _http_status_error_types()
    rejected = _token_rejected_type()
    seen: list[BaseException] = [exc]
    visited: set[int] = set()
    while seen:
        e = seen.pop()
        if id(e) in visited:
            continue
        visited.add(id(e))
        if rejected is not None and isinstance(e, rejected):
            return True
        if isinstance(e, status_errors) and e.response.status_code in (401, 403):
            return True
        seen.extend(getattr(e, "exceptions", ()))
        for linked in (e.__cause__, e.__context__):
            if linked is not None:
                seen.append(linked)
    return False


async def _call_notion(access_token: str, tool_name: str, **kwargs: Any) -> Any:
    try:
        from anton.core.mcp.errors import McpPermanentError
        from anton.core.mcp.servers import mcp_server_url
        from anton.core.mcp.wiring import call_mcp_tool
    except ImportError as exc:
        raise NotionSearchUnavailable("installed anton has no MCP client") from exc
    if mcp_server_url("notion") is None:
        raise NotionSearchUnavailable("installed anton has no Notion MCP server URL")
    try:
        return await call_mcp_tool("notion", access_token, tool_name, **kwargs)
    except McpPermanentError as exc:
        raise NotionReconnectRequired(str(exc)) from exc
    except Exception as exc:
        if is_token_rejection(exc):
            raise NotionReconnectRequired(str(exc)) from exc
        raise


async def search_pages(access_token: str, query: str) -> list[dict[str, str]]:
    return parse_search_results(await _call_notion(access_token, "notion-search", query=query))


def parse_fetched_page(content: Any) -> dict[str, Any]:
    """Title and text of a `notion-fetch` result, the text cut to
    MAX_PAGE_CHARS. Content that isn't the expected JSON is used as plain text."""
    data = _decode(content)
    if isinstance(data, dict):
        title, text = _text(data.get("title")), _text(data.get("text"))
    else:
        title, text = "", _text(content) if isinstance(content, str) else ""
    return {"title": title, "text": text[:MAX_PAGE_CHARS], "truncated": len(text) > MAX_PAGE_CHARS}


async def fetch_page(access_token: str, page: str) -> dict[str, Any]:
    return parse_fetched_page(await _call_notion(access_token, "notion-fetch", id=page))


MAX_LISTED_PAGES = 50
MAX_PAGE_NAME_CHARS = 200
MAX_PAGE_URL_CHARS = 500


def _capped(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe_project_pages(pages_by_connection: dict[str, list[dict]]) -> str:
    """System-prompt list of the Notion pages added to the current project,
    so the agent can read them in any later turn, not only in the message
    they were added with. Empty when there are none. How to read a page is
    in notion.json's usage_notes; this is per-project data only.

    Names and urls come from the user's workspace, so each is capped, and so
    is the list, to keep the prompt bounded."""
    lines = [
        f"- {_capped(str(page.get('name') or 'Untitled'), MAX_PAGE_NAME_CHARS)} "
        f"(url: {_capped(str(page.get('url') or page.get('id')), MAX_PAGE_URL_CHARS)}, connection: {conn_name})"
        for conn_name, pages in pages_by_connection.items()
        for page in pages
        if isinstance(page, dict)
    ]
    if not lines:
        return ""
    listed = lines[:MAX_LISTED_PAGES]
    if len(lines) > MAX_LISTED_PAGES:
        listed.append(f"and {len(lines) - MAX_LISTED_PAGES} more")
    return (
        "\n\nNotion pages the user added to this project's files:\n"
        + "\n".join(listed)
        + "\nWhen the user refers to the project's files, documents or pages, these are included."
    )
