"""Agent tools for the desktop app's embedded browser.

The Electron main process runs a loopback "browser bridge" HTTP service
(127.0.0.1, random port, bearer-token auth) that exposes the user's REAL,
visible browser tabs — the agent and the user share one live browser, and
every action the agent takes is instantly visible on screen.

These ToolDefs are thin async wrappers over that bridge. Discovery happens
at CALL time (never at build time, so tools register fine even when the
desktop app isn't running) from two ordered candidates:

1. ``COWORK_BROWSER_BRIDGE_PORT`` + ``COWORK_BROWSER_BRIDGE_TOKEN`` env vars
   (the desktop app sets these for the server process it spawns).
2. ``<cowork_home>/browser-bridge.json`` — ``{port, token, pid}`` written by
   the desktop app on launch. Uses the shared ``cowork_home()`` helper so
   dev/preview homes (``COWORK_HOME=~/.cowork-dev``) work too.

The env vars are frozen at server spawn, but the app generates a NEW
port+token on every launch and rewrites the file — so after an app
restart (e.g. an adopted server, ENG-439) the env candidate is stale and
the file is fresh. Every bridge call tries the candidates in order,
falling through only on connection/auth failure, and a file candidate
whose recorded pid is dead (crashed app) is skipped up front.

Every failure mode — no discovery info, connection refused, timeout, bridge
``{error}`` payload — is converted into a plain-language string for the LLM.
Handlers NEVER raise into the agent loop.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from cowork.common.paths import cowork_home

logger = logging.getLogger(__name__)

# navigate/click/type wait for page load on the bridge side (≤5 s settle) —
# give them headroom; everything else is immediate.
_SLOW_TIMEOUT = 30.0
_FAST_TIMEOUT = 10.0

_UNAVAILABLE_MSG = (
    "The desktop browser is unavailable right now — the MindsHub Cowork "
    "desktop app may not be running or the Browser feature hasn't been "
    "opened yet. Tell the user, don't retry in a loop."
)

# Local safety caps (the bridge caps too; these keep pathological payloads
# from blowing up the context).
_READ_HARD_CAP = 30000
_SNAPSHOT_MAX_LINES = 200


class _BridgeUnavailable(Exception):
    """No discovery info, or the bridge couldn't be reached."""


class _BridgeHTTPError(Exception):
    """Bridge answered with a non-2xx — message is its ``{error}`` string."""


def _pid_alive(pid: Any) -> bool:
    """Best-effort liveness check for the pid recorded in the discovery file.

    Only a definitive "no such process" filters the candidate out — anything
    inconclusive (no permission, a platform without signal-0 semantics, a
    junk value) keeps it, since a genuinely dead bridge is caught by the
    connection fallthrough anyway.
    """
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError, TypeError, ValueError):
        return True


def _resolve_bridge_candidates() -> list[tuple[str, str]]:
    """Ordered ``(base_url, token)`` candidates for the bridge, resolved fresh
    on every call — env vars first, then the discovery file (skipped when its
    recorded pid is dead). Returns [] when neither source is available — never
    raises, so a missing/corrupt discovery file just means "browser
    unavailable"."""
    candidates: list[tuple[str, str]] = []
    port = os.environ.get("COWORK_BROWSER_BRIDGE_PORT")
    token = os.environ.get("COWORK_BROWSER_BRIDGE_TOKEN")
    if port and token:
        candidates.append((f"http://127.0.0.1:{port}", token))
    try:
        info = json.loads((cowork_home() / "browser-bridge.json").read_text())
        port, token = int(info["port"]), str(info["token"])
        pid = info.get("pid")
        if port and token and (pid is None or _pid_alive(pid)):
            candidate = (f"http://127.0.0.1:{port}", token)
            if candidate not in candidates:
                candidates.append(candidate)
    except Exception:
        pass  # absent / unreadable / corrupt file → not a candidate
    return candidates


async def _bridge_call(
    method: str,
    path: str,
    *,
    params: dict | None = None,
    body: dict | None = None,
    timeout: float = _FAST_TIMEOUT,
) -> Any:
    """One authenticated round-trip to the bridge, tried against each
    discovery candidate in order. A candidate that can't be connected to or
    that rejects the token (401) is stale — the app was relaunched with a new
    port/token — so the call falls through to the next candidate
    transparently. A 4xx/5xx from a live bridge is a REAL error and stops the
    fallthrough. Raises only ``_BridgeUnavailable`` / ``_BridgeHTTPError`` —
    both expected, both converted to friendly strings by ``_call_and_format``."""
    candidates = _resolve_bridge_candidates()
    if not candidates:
        raise _BridgeUnavailable("no bridge discovery info (env/file)")
    if params:
        params = {k: v for k, v in params.items() if v is not None}
    if body:
        body = {k: v for k, v in body.items() if v is not None}
    unavailable: _BridgeUnavailable | None = None
    for base_url, token in candidates:
        try:
            async with httpx.AsyncClient(
                base_url=base_url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=timeout,
                # Loopback only — never route through a user's HTTP(S)_PROXY,
                # which would both break the call and expose the bridge token.
                trust_env=False,
            ) as client:
                resp = await client.request(method, path, params=params, json=body)
        except (httpx.HTTPError, OSError) as exc:
            unavailable = _BridgeUnavailable(str(exc))
            continue  # dead/stale candidate → try the next one
        if resp.status_code == 401:
            # A live bridge rejecting the token means the app rotated it →
            # the next candidate's token may still be fresh.
            unavailable = _BridgeUnavailable(f"{base_url} rejected the bridge token (401)")
            continue
        if resp.status_code >= 400:
            error = None
            try:
                payload = resp.json()
                if isinstance(payload, dict):
                    error = payload.get("error")
            except Exception:
                pass
            raise _BridgeHTTPError(str(error) if error else f"HTTP {resp.status_code}")
        try:
            return resp.json()
        except Exception as exc:
            raise _BridgeHTTPError(f"invalid response ({exc})") from exc
    raise unavailable or _BridgeUnavailable("bridge unreachable")


async def _call_and_format(
    name: str,
    method: str,
    path: str,
    *,
    params: dict | None = None,
    body: dict | None = None,
    timeout: float = _FAST_TIMEOUT,
    ok,
) -> str:
    """Shared error funnel for every browser tool handler: bridge call +
    success formatting, with all failures returned as strings."""
    try:
        data = await _bridge_call(method, path, params=params, body=body, timeout=timeout)
    except _BridgeUnavailable:
        logger.info("%s: desktop browser bridge unavailable", name)
        return _UNAVAILABLE_MSG
    except _BridgeHTTPError as exc:
        return f"Browser error: {exc}"
    except Exception as exc:  # belt-and-braces: never raise into the agent loop
        logger.exception("Cowork %s failed", name)
        return f"{name}: unexpected error ({exc})"
    try:
        return ok(data if isinstance(data, dict) else {})
    except Exception as exc:
        logger.exception("Cowork %s response formatting failed", name)
        return f"{name}: unexpected response from browser bridge ({exc})"


# ---------------------------------------------------------------------------
# Response formatting (compact, LLM-friendly)
# ---------------------------------------------------------------------------


def _clean(text: Any, limit: int = 80) -> str:
    return " ".join(str(text or "").split())[:limit]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[… truncated — {len(text) - limit} more characters]"


def _fmt_navigate(data: dict) -> str:
    url = data.get("url") or ""
    tab_id = data.get("tabId") or ""
    return (
        f"Navigated to {url} (tabId: {tab_id}).\n"
        "Next: browser_read to consume the content, or browser_snapshot to "
        "interact with the page."
    )


def _origin_of(url: str) -> str:
    try:
        from urllib.parse import urlparse

        parts = urlparse(url or "")
        return f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else ""
    except Exception:
        return ""


def _fmt_tabs(data: dict) -> str:
    tabs = data.get("tabs") or []
    if not tabs:
        return "No tabs are open in the desktop browser. Use browser_navigate to open one."
    active_id = data.get("activeTabId")
    # /state carries the apps registry; label tabs that belong to a pinned app
    # so the agent can address tools by name ("check Gmail") not tabIds.
    app_names = {a.get("origin", ""): a.get("name", "") for a in data.get("apps") or [] if a.get("origin")}
    lines = []
    for i, tab in enumerate(tabs, 1):
        title = _clean(tab.get("title")) or "New tab"
        url = (tab.get("url") or "").strip() or "(blank)"
        marker = "[active] " if tab.get("id") == active_id else ""
        loading = " (loading…)" if tab.get("isLoading") else ""
        app_label = f"  [app: {app_names[_origin_of(tab.get('url') or '')]}]" if _origin_of(tab.get("url") or "") in app_names else ""
        lines.append(f"{i}. {marker}{title} — {url}{loading} (tabId: {tab.get('id')}){app_label}")
    return "Open tabs:\n" + "\n".join(lines)


def _fmt_read(data: dict) -> str:
    title = _clean(data.get("title"), 200) or "(untitled)"
    url = data.get("url") or ""
    text = _truncate(str(data.get("text") or ""), _READ_HARD_CAP)
    if not text.strip():
        text = "(no readable text on this page — try browser_snapshot or browser_screenshot)"
    return f"{title}\n{url}\n\n{text}"


def _fmt_element(el: dict) -> str:
    idx = el.get("index")
    tag = str(el.get("tag") or "").lower()
    role = str(el.get("role") or "").lower()
    text = _clean(el.get("text"))
    href = el.get("href")
    if tag == "a" or role == "link":
        label = "link"
    elif tag == "input":
        label = f"input({_clean(el.get('inputType'), 20) or 'text'})"
    elif tag in ("textarea", "select"):
        label = tag
    elif tag == "button" or role == "button":
        label = "button"
    else:
        label = role or tag or "element"
    line = f"[{idx}] {label} '{text}'"
    if href:
        line += f" -> {href}"
    return line


def _fmt_snapshot(data: dict) -> str:
    title = _clean(data.get("title"), 200) or "(untitled)"
    url = data.get("url") or ""
    els = data.get("elements") or []
    v = data.get("v")
    if isinstance(v, int) and not isinstance(v, bool):
        count_line = (
            f"snapshot v={v} — {len(els)} interactive elements — pass [index] "
            f"and snapshot_v={v} to browser_click/browser_type:"
        )
    else:  # older bridge without versioning
        count_line = (
            f"Interactive elements ({len(els)}) — pass [index] to browser_click/browser_type:"
        )
    header = f"{title}\n{url}\n\n{count_line}"
    if not els:
        return header + "\n(none — the page may still be loading, or it has no links/buttons/inputs)"
    lines = [_fmt_element(el) for el in els[:_SNAPSHOT_MAX_LINES]]
    if len(els) > _SNAPSHOT_MAX_LINES:
        lines.append(f"… and {len(els) - _SNAPSHOT_MAX_LINES} more — scroll and re-snapshot for the rest.")
    return header + "\n" + "\n".join(lines)


def _fmt_screenshot(data: dict) -> str:
    path = data.get("path") or ""
    w, h, scale = data.get("cssWidth"), data.get("cssHeight"), data.get("scale")
    size = ""
    if w and h and scale:
        size = (
            f"\nThe image covers the viewport's {w}x{h} CSS pixels at {scale}x scale "
            f"(so the PNG is ~{int(w * scale)}x{int(h * scale)} px). browser_click_at "
            f"takes CSS pixels: DIVIDE any pixel coordinate you read off the image by "
            f"{scale} — forgetting this puts clicks {scale}x too far down-right."
        )
    return f"Screenshot saved to: {path}\nView it with the read_image tool using this exact path.{size}"


# ---------------------------------------------------------------------------
# Per-turn browser context (Browser Agent dock)
# ---------------------------------------------------------------------------

# Short timeout: this runs inside the request path of every dock turn, so a
# wedged bridge must cost the turn ~1s, not the tool-call timeouts.
_CONTEXT_TIMEOUT = 2.0
_CONTEXT_MAX_TABS = 10

_CONTEXT_GUIDANCE = (
    "You are the user's browsing copilot. A real Chromium browser is open beside this "
    "chat in the MindsHub Cowork desktop app, and the user can watch everything you do "
    "in it. Drive it with the browser_* tools (navigate, read, snapshot, click, type, "
    "scroll, back, tabs, screenshot, click_at, press_key, insert_text, paste, close_tab).\n"
    "Whenever the request relates to an open page — or the user asks you to check, read, "
    "summarize, or do something on the web — act on the browser with those tools and "
    "answer from the LIVE page, not from memory or generic web search. Default to the "
    "active tab unless the user says otherwise; open background tabs for follow-up "
    "research and bring one forward only when it has the result. When you act, briefly "
    "say what you did in the browser.\n"
    "Canvas-rendered apps (Google Sheets, Figma…) have no per-element DOM: work them "
    "visually (screenshot → click_at → insert_text/press_key) and use browser_paste "
    "with tab/newline-separated text for bulk spreadsheet data."
)


async def build_browser_turn_context() -> str:
    """Copilot guidance + live tab state for a Browser Agent dock turn.

    Injected into the LLM input only (never persisted), rebuilt per turn so it
    tracks the user as they browse. Returns '' when the bridge is unreachable
    or has nothing to report — the turn then proceeds without context and the
    tools themselves still degrade gracefully if called.
    """
    try:
        state = await _bridge_call("GET", "/state", timeout=_CONTEXT_TIMEOUT)
    except (_BridgeUnavailable, _BridgeHTTPError):
        return ""
    tabs = state.get("tabs") or []
    active_id = state.get("activeTabId")
    if not tabs:
        state_block = (
            "No tabs are open right now — if the user wants you to look something up "
            "or do something on the web, open a tab with browser_navigate and go from there."
        )
    else:
        lines = []
        for i, tab in enumerate(tabs[: _CONTEXT_MAX_TABS], 1):
            title = _clean(tab.get("title"), 60) or "New tab"
            url = (tab.get("url") or "").strip() or "(blank)"
            marker = "[active] " if tab.get("id") == active_id else ""
            lines.append(f"{i}. {marker}{title} — {url}")
        if len(tabs) > _CONTEXT_MAX_TABS:
            lines.append(f"… and {len(tabs) - _CONTEXT_MAX_TABS} more (browser_tabs for the full list).")
        state_block = "Currently open tabs (live):\n" + "\n".join(lines)
    return f"{_CONTEXT_GUIDANCE}\n\n{state_block}"


# ---------------------------------------------------------------------------
# Tool handlers — arg extraction style matches tools.py (str(...).strip(),
# "tool_name: `arg` is required." error strings, logger on failures).
# ---------------------------------------------------------------------------


def _tab_id(tc_input: dict) -> str | None:
    return str(tc_input.get("tab_id") or "").strip() or None


def _index_arg(name: str, tc_input: dict) -> int | str:
    """Element index, or an error string. 0 is a valid index; bools are not."""
    index = tc_input.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        return f"{name}: `index` (integer from the latest browser_snapshot) is required."
    return index


def _snapshot_v_arg(tc_input: dict) -> int | None:
    """Optional snapshot version token for staleness checks; bools are not ints."""
    v = tc_input.get("snapshot_v")
    return v if isinstance(v, int) and not isinstance(v, bool) else None


async def _browser_navigate(session: Any, tc_input: dict) -> str:
    url = str(tc_input.get("url", "")).strip()
    if not url:
        return "browser_navigate: `url` is required."
    if tc_input.get("new_tab") and tc_input.get("background"):
        return await _call_and_format(
            "browser_navigate", "POST", "/tabs",
            body={"url": url, "activate": False}, timeout=_SLOW_TIMEOUT,
            ok=lambda data: (
                f"Opened background tab for {url} (tabId: {data.get('tabId') or ''}) — "
                "the user's view stays on their current tab. Use browser_read/"
                "browser_snapshot with this tab_id; bring it forward only if it "
                "has the result worth showing."
            ),
        )
    body: dict[str, Any] = {"url": url}
    if tc_input.get("new_tab"):
        body["newTab"] = True
    return await _call_and_format(
        "browser_navigate", "POST", "/navigate",
        body=body, timeout=_SLOW_TIMEOUT, ok=_fmt_navigate,
    )


async def _browser_tabs(session: Any, tc_input: dict) -> str:
    return await _call_and_format("browser_tabs", "GET", "/state", ok=_fmt_tabs)


async def _browser_read(session: Any, tc_input: dict) -> str:
    max_chars = tc_input.get("max_chars")
    params = {"tabId": _tab_id(tc_input)}
    if isinstance(max_chars, int) and not isinstance(max_chars, bool) and max_chars > 0:
        params["maxChars"] = max_chars
    return await _call_and_format("browser_read", "GET", "/read", params=params, ok=_fmt_read)


async def _browser_snapshot(session: Any, tc_input: dict) -> str:
    return await _call_and_format(
        "browser_snapshot", "GET", "/snapshot",
        params={"tabId": _tab_id(tc_input)}, ok=_fmt_snapshot,
    )


async def _browser_click(session: Any, tc_input: dict) -> str:
    index = _index_arg("browser_click", tc_input)
    if isinstance(index, str):
        return index
    return await _call_and_format(
        "browser_click", "POST", "/click",
        body={"tabId": _tab_id(tc_input), "index": index, "v": _snapshot_v_arg(tc_input)},
        timeout=_SLOW_TIMEOUT,
        ok=lambda data: (
            f"Clicked element [{index}]. If the page changed, take a fresh "
            "browser_snapshot before the next click/type — old indexes are stale."
        ),
    )


async def _browser_type(session: Any, tc_input: dict) -> str:
    index = _index_arg("browser_type", tc_input)
    if isinstance(index, str):
        return index
    text = tc_input.get("text")
    if text is None:
        return "browser_type: `text` is required."
    text = str(text)
    submit = bool(tc_input.get("submit"))
    return await _call_and_format(
        "browser_type", "POST", "/type",
        body={
            "tabId": _tab_id(tc_input), "index": index, "text": text,
            "submit": submit, "v": _snapshot_v_arg(tc_input),
        },
        timeout=_SLOW_TIMEOUT,
        ok=lambda data: (
            f"Typed into element [{index}]{ ' and submitted (Enter)' if submit else ''}. "
            "If the page changed, re-snapshot before the next interaction."
        ),
    )


_SCROLL_DIRECTIONS = ("up", "down", "top", "bottom")


async def _browser_scroll(session: Any, tc_input: dict) -> str:
    direction = str(tc_input.get("direction", "")).strip().lower()
    if direction not in _SCROLL_DIRECTIONS:
        return f"browser_scroll: `direction` must be one of {', '.join(_SCROLL_DIRECTIONS)}."
    amount = tc_input.get("amount")
    body: dict[str, Any] = {"tabId": _tab_id(tc_input), "direction": direction}
    if isinstance(amount, int) and not isinstance(amount, bool) and amount > 0:
        body["amount"] = amount
    return await _call_and_format(
        "browser_scroll", "POST", "/scroll", body=body,
        ok=lambda data: (
            f"Scrolled {direction}. Content above/below may have changed — "
            "re-snapshot if you need fresh element indexes."
        ),
    )


async def _browser_back(session: Any, tc_input: dict) -> str:
    return await _call_and_format(
        "browser_back", "POST", "/back",
        body={"tabId": _tab_id(tc_input)},
        ok=lambda data: (
            "Already at the earliest page in this tab's history."
            if data.get("ok") and data.get("moved") is False
            else "Went back to the previous page. Re-snapshot before interacting — "
            "element indexes have changed."
        ),
    )


async def _browser_screenshot(session: Any, tc_input: dict) -> str:
    return await _call_and_format(
        "browser_screenshot", "POST", "/screenshot",
        body={"tabId": _tab_id(tc_input)}, ok=_fmt_screenshot,
    )


async def _browser_close_tab(session: Any, tc_input: dict) -> str:
    tab_id = _tab_id(tc_input)
    return await _call_and_format(
        "browser_close_tab", "POST", "/tabs/close",
        body={"tabId": tab_id},
        ok=lambda data: f"Closed {'the active tab' if not tab_id else f'tab {tab_id}'}.",
    )


def _app_field(app: dict, key: str) -> str:
    """Registry fields can be null on corrupt/partial entries — never .lower() None."""
    return str(app.get(key) or "").lower()


def _match_app(apps: list[dict], query: str) -> dict | None:
    """Resolve a user/agent-supplied app name or id: exact (case-insensitive),
    then prefix, then substring across name and origin."""
    q = query.strip().lower()
    if not q:
        return None
    for app in apps:
        if _app_field(app, "id") == q or _app_field(app, "name") == q:
            return app
    for app in apps:
        if _app_field(app, "name").startswith(q) or q in _app_field(app, "origin"):
            return app
    for app in apps:
        if q in _app_field(app, "name"):
            return app
    return None


async def _browser_open_app(session: Any, tc_input: dict) -> str:
    name = str(tc_input.get("name") or "").strip()
    if not name:
        return "browser_open_app: `name` is required — the app's name as pinned in the sidebar (e.g. 'Gmail', 'Slack')."

    try:
        apps_data = await _bridge_call("GET", "/apps", timeout=_FAST_TIMEOUT)
    except _BridgeUnavailable:
        return _UNAVAILABLE_MSG
    except _BridgeHTTPError as exc:
        return f"Browser error: {exc}"
    apps = apps_data if isinstance(apps_data, list) else []
    app = _match_app(apps, name)
    if not app:
        known = ", ".join(a.get("name", "?") for a in apps[:8]) or "none yet"
        return (
            f"No app matches '{name}'. Pinned apps: {known}. "
            "Ask the user to pin it (sidebar → Add app), or browser_navigate to its URL instead."
        )
    return await _call_and_format(
        "browser_open_app", "POST", "/apps/open",
        body={"appId": app["id"]}, timeout=_SLOW_TIMEOUT,
        ok=lambda data: (
            f"Opened {app['name']} — {'a fresh pinned tab' if data.get('created') else 'its existing tab is now active'}. "
            "Follow with browser_read or browser_snapshot to work it."
        ),
    )


_MODIFIER_NAMES = ("cmd", "ctrl", "alt", "shift")


async def _browser_click_at(session: Any, tc_input: dict) -> str:
    x, y = tc_input.get("x"), tc_input.get("y")
    if (
        isinstance(x, bool)
        or not isinstance(x, (int, float))
        or isinstance(y, bool)
        or not isinstance(y, (int, float))
    ):
        return (
            "browser_click_at: `x` and `y` (viewport CSS pixel coordinates, as seen "
            "in the latest browser_screenshot) are required."
        )
    return await _call_and_format(
        "browser_click_at", "POST", "/click-at",
        body={"tabId": _tab_id(tc_input), "x": x, "y": y},
        timeout=_SLOW_TIMEOUT,
        ok=lambda data: (
            f"Clicked at ({x}, {y}) — a real trusted click, like a mouse. If the "
            "page changed, re-screenshot or re-snapshot before the next action."
        ),
    )


async def _browser_press_key(session: Any, tc_input: dict) -> str:
    key = str(tc_input.get("key") or "").strip()
    if not key:
        return (
            "browser_press_key: `key` is required — e.g. 'enter', 'tab', 'escape', "
            "'backspace', 'arrowup/down/left/right', or a single character."
        )
    modifiers = tc_input.get("modifiers")
    mods = (
        [str(m).lower() for m in modifiers if str(m).lower() in _MODIFIER_NAMES]
        if isinstance(modifiers, list)
        else None
    )
    return await _call_and_format(
        "browser_press_key", "POST", "/press",
        body={"tabId": _tab_id(tc_input), "key": key, "modifiers": mods},
        timeout=_SLOW_TIMEOUT,
        ok=lambda data: (
            f"Pressed {key}"
            + (f" with {'+'.join(mods)}" if mods else "")
            + " — a real trusted key press, default actions included."
        ),
    )


async def _browser_insert_text(session: Any, tc_input: dict) -> str:
    text = tc_input.get("text")
    if text is None:
        return "browser_insert_text: `text` is required."
    text = str(text)
    return await _call_and_format(
        "browser_insert_text", "POST", "/insert-text",
        body={"tabId": _tab_id(tc_input), "text": text},
        ok=lambda data: (
            f"Inserted {len(text)} characters at the focused element (trusted input, "
            "as if typed). Follow with browser_press_key 'enter' to commit/submit."
        ),
    )


async def _browser_paste(session: Any, tc_input: dict) -> str:
    text = tc_input.get("text")
    if text is None:
        return (
            "browser_paste: `text` is required — for spreadsheets use tab-separated "
            "columns and newline-separated rows; the app splits it into cells."
        )
    text = str(text)
    return await _call_and_format(
        "browser_paste", "POST", "/paste",
        body={"tabId": _tab_id(tc_input), "text": text},
        timeout=_SLOW_TIMEOUT,
        ok=lambda data: (
            f"Pasted {len(text)} characters at the focused element. In spreadsheet "
            "apps, tab/newline-separated text lands as a cell range — verify with a "
            "screenshot or by reading the target area."
        ),
    )


# ---------------------------------------------------------------------------
# Schemas + builder
# ---------------------------------------------------------------------------

_TAB_ID_PROP = {
    "type": "string",
    "description": "Tab id from browser_tabs. Omit to use the currently active tab.",
}

_NAVIGATE_PROMPT = (
    "BROWSER — you drive the user's REAL, visible desktop browser; they see every action live.\n"
    "- To consume a page, use `browser_read` (clean text). To decide what to click or fill, use\n"
    "  `browser_snapshot` FIRST — click/type indexes are only valid for the latest snapshot.\n"
    "- After anything that changes the page (navigate/click/type/back), the old snapshot is stale —\n"
    "  re-snapshot before the next interaction, and pass the snapshot's `v` as `snapshot_v` to\n"
    "  click/type; a stale-snapshot error means re-snapshot and retry ONCE.\n"
    "- Parallel research: open background tabs (new_tab=true, background=true) so the user's view\n"
    "  stays put, read/snapshot them by tab_id, and only bring one forward (navigate its URL in\n"
    "  the active tab) when it has the result worth showing. Close tabs you opened.\n"
    "- When the user asks you to open/show something, navigate in the active tab (default) or a\n"
    "  foreground new tab (new_tab=true) so they actually see it.\n"
    "- When the user names one of their tools by name ('check my email', 'open Linear'), use\n"
    "  browser_open_app — it opens their pinned, already-logged-in app rather than a cold URL.\n"
    "- Never loop a failed action — re-read or re-snapshot once, and if the browser is unavailable, tell the user."
)


_CANVAS_PLAYBOOK_PROMPT = (
    "Canvas-rendered app playbook (Google Sheets/Docs, Figma, Canva, maps, drawing tools):\n"
    "- These render into a <canvas> — cells/shapes/zones are NOT in browser_snapshot, and\n"
    "  synthetic clicks/typing are ignored. Work them visually:\n"
    "  browser_screenshot → browser_click_at(x, y) → browser_insert_text / browser_press_key\n"
    "  (Enter commits, Tab/arrows move) → re-screenshot to verify each macro step.\n"
    "- COORDINATES: screenshots are device-scaled (the tool result gives the CSS size and\n"
    "  scale). Divide image pixels by the scale before browser_click_at, and verify the\n"
    "  result (Name Box / formula bar / a re-screenshot) before continuing — one wrong\n"
    "  click pastes a table in the wrong place.\n"
    "- Bulk data beats per-cell typing: click the target cell FIRST (the paste lands at\n"
    "  the focused/selected cell), then browser_paste tab/newline-separated text — the\n"
    "  app splits it into a cell range.\n"
    "- Google Sheets specifics: jump to any cell via the Name Box (a real DOM input):\n"
    "  click it (browser_click_at), browser_insert_text the reference (e.g. 'B2'),\n"
    "  browser_press_key 'enter' — trusted input commits reliably, where synthetic typing\n"
    "  can leave the grid unfocused. The formula bar and toolbars are real DOM too. Never\n"
    "  type data into the Name Box — it only takes references and creates named ranges\n"
    "  from anything else. On macOS, use 'cmd' for shortcuts (e.g. cmd+home for A1),\n"
    "  not 'ctrl'.\n"
    "- If a Google Drive connection exists and the sheet is one the user picked there,\n"
    "  the Sheets API (via scratchpad) is the most reliable bulk-write path of all."
)


def build_browser_tools():
    """ToolDefs driving the desktop app's embedded browser via the loopback
    bridge. Discovery is deferred to call time, so this always succeeds —
    tools simply return a friendly "unavailable" string when the desktop
    app isn't running."""
    from anton.core.tools.tool_defs import ToolDef

    return [
        ToolDef(
            name="browser_navigate",
            description=(
                "Open a URL (or search phrase) in the user's live desktop browser. "
                "Use to start any browsing task, open a link the user gave you, or "
                "open a new tab (new_tab=true) for parallel research — background=true "
                "opens it without stealing the user's view; bring it forward only "
                "once it has the result worth showing. When the user asked you to "
                "open/show something, use the active tab (default) or a foreground "
                "new tab so they see it. Returns the tab id — follow up with "
                "browser_read to consume the page or browser_snapshot to interact "
                "with it."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to open (e.g. 'https://example.com'). A bare domain or search phrase works too — the browser normalizes it.",
                    },
                    "new_tab": {
                        "type": "boolean",
                        "description": "Open in a new tab instead of reusing the active one. Default false.",
                    },
                    "background": {
                        "type": "boolean",
                        "description": "With new_tab=true, open the tab in the background — the user's view stays on their current tab. Use for parallel research. Default false.",
                    },
                },
                "required": ["url"],
            },
            handler=_browser_navigate,
            prompt=_NAVIGATE_PROMPT,
        ),
        ToolDef(
            name="browser_tabs",
            description=(
                "List all open tabs in the user's desktop browser (title, url, tab "
                "id, which is active). Use to find the tabId other browser_* tools "
                "take, or to see what the user currently has open."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_browser_tabs,
        ),
        ToolDef(
            name="browser_read",
            description=(
                "Extract the readable text of a web page (article-style, nav/ads/"
                "footers stripped). Use this to consume, summarize, or quote page "
                "content — NOT to decide what to click; use browser_snapshot for that."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "tab_id": _TAB_ID_PROP,
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters of page text to return (bridge default 20000). Lower it for a quick skim.",
                    },
                },
            },
            handler=_browser_read,
        ),
        ToolDef(
            name="browser_snapshot",
            description=(
                "List the interactive elements of a page (links, buttons, inputs) "
                "with numeric indexes. Call this BEFORE browser_click/browser_type — "
                "indexes are only valid for the latest snapshot of that tab."
            ),
            input_schema={
                "type": "object",
                "properties": {"tab_id": _TAB_ID_PROP},
            },
            handler=_browser_snapshot,
        ),
        ToolDef(
            name="browser_click",
            description=(
                "Click an element by its index from the latest browser_snapshot. "
                "Pass that snapshot's v as snapshot_v when you can — if the page "
                "changed since, you get a stale-snapshot error: re-snapshot and "
                "retry ONCE. Waits for the page to settle; re-snapshot afterwards "
                "before clicking or typing again."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Element index from the latest browser_snapshot.",
                    },
                    "snapshot_v": {
                        "type": "integer",
                        "description": "The v from your most recent browser_snapshot — lets the bridge detect a stale page. Optional.",
                    },
                    "tab_id": _TAB_ID_PROP,
                },
                "required": ["index"],
            },
            handler=_browser_click,
        ),
        ToolDef(
            name="browser_type",
            description=(
                "Type text into an input or textarea by snapshot index (focuses and "
                "sets the value like a real user). Set submit=true to press Enter "
                "afterwards — e.g. search boxes and login forms. Use browser_snapshot "
                "first to find the index, and pass that snapshot's v as snapshot_v "
                "when you can — if the page changed since, you get a stale-snapshot "
                "error: re-snapshot and retry ONCE."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Element index from the latest browser_snapshot.",
                    },
                    "text": {"type": "string", "description": "Text to type into the element."},
                    "submit": {
                        "type": "boolean",
                        "description": "Press Enter after typing. Default false.",
                    },
                    "snapshot_v": {
                        "type": "integer",
                        "description": "The v from your most recent browser_snapshot — lets the bridge detect a stale page. Optional.",
                    },
                    "tab_id": _TAB_ID_PROP,
                },
                "required": ["index", "text"],
            },
            handler=_browser_type,
        ),
        ToolDef(
            name="browser_scroll",
            description=(
                "Scroll the page up/down, or jump to the top/bottom. Use to reveal "
                "more content before reading further or re-snapshotting."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": list(_SCROLL_DIRECTIONS),
                        "description": "Scroll direction: 'up'/'down' one screen, or 'top'/'bottom' of the page.",
                    },
                    "tab_id": _TAB_ID_PROP,
                },
                "required": ["direction"],
            },
            handler=_browser_scroll,
        ),
        ToolDef(
            name="browser_back",
            description=(
                "Go back one page in the tab's history. Use to return to the "
                "previous page after following a link."
            ),
            input_schema={
                "type": "object",
                "properties": {"tab_id": _TAB_ID_PROP},
            },
            handler=_browser_back,
        ),
        ToolDef(
            name="browser_screenshot",
            description=(
                "Capture the visible page as a PNG and return its file path. Use "
                "when layout/visuals matter or text extraction comes back empty — "
                "then view the file with the read_image tool. The result includes "
                "the CSS viewport size and pixel scale — divide image pixels by "
                "the scale before passing coordinates to browser_click_at. Only "
                "works on a tab the user can currently see (the active tab with "
                "the Browser view open); if it errors, activate the tab or ask "
                "the user to open the Browser view first."
            ),
            input_schema={
                "type": "object",
                "properties": {"tab_id": _TAB_ID_PROP},
            },
            handler=_browser_screenshot,
        ),
        ToolDef(
            name="browser_close_tab",
            description=(
                "Close a browser tab. Use to clean up tabs you opened once research "
                "is done; omit tab_id to close the active tab."
            ),
            input_schema={
                "type": "object",
                "properties": {"tab_id": _TAB_ID_PROP},
            },
            handler=_browser_close_tab,
        ),
        ToolDef(
            name="browser_open_app",
            description=(
                "Open one of the user's pinned web apps by NAME (e.g. 'Gmail', 'Slack', "
                "'Linear') — activates its existing tab or opens a fresh pinned tab, "
                "already logged in. Use this first whenever the user names a tool they "
                "use ('check my email', 'what's in Slack'); follow with browser_read "
                "or browser_snapshot to work the page."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "App name as pinned in the sidebar (case-insensitive; prefix/substring ok).",
                    },
                },
                "required": ["name"],
            },
            handler=_browser_open_app,
        ),
        ToolDef(
            name="browser_click_at",
            description=(
                "Trusted click at viewport pixel coordinates, like a real mouse — "
                "for canvas-rendered apps (Google Sheets/Docs, Figma, maps) whose "
                "elements don't appear in browser_snapshot. Take a browser_screenshot "
                "first to aim (coordinates = CSS pixels: screenshot pixels divided "
                "by the reported scale), then verify the result with another "
                "screenshot. Prefer browser_click whenever the element IS in a "
                "snapshot — it's more reliable."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "Viewport x in CSS pixels."},
                    "y": {"type": "number", "description": "Viewport y in CSS pixels."},
                    "tab_id": _TAB_ID_PROP,
                },
                "required": ["x", "y"],
            },
            handler=_browser_click_at,
            prompt=_CANVAS_PLAYBOOK_PROMPT,
        ),
        ToolDef(
            name="browser_press_key",
            description=(
                "Press a real key (trusted, default actions included) in the focused "
                "element: 'enter', 'tab', 'escape', 'backspace', 'delete', arrows "
                "('left'/'right'/'up'/'down' or 'arrowleft' etc.), 'home'/'end', "
                "'pageup'/'pagedown', or a single character. Optional modifiers from "
                "[cmd, ctrl, alt, shift] — on macOS use 'cmd' for app shortcuts "
                "(e.g. cmd+a, cmd+home). Use after browser_click_at/"
                "browser_insert_text to commit edits (Enter) or move between "
                "cells/fields (Tab/arrows) in canvas apps."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Key name or single character."},
                    "modifiers": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["cmd", "ctrl", "alt", "shift"]},
                        "description": "Optional modifier keys to hold.",
                    },
                    "tab_id": _TAB_ID_PROP,
                },
                "required": ["key"],
            },
            handler=_browser_press_key,
        ),
        ToolDef(
            name="browser_insert_text",
            description=(
                "Insert text at the focused element as if typed by a real user "
                "(trusted input, fast for long text). Use after browser_click_at "
                "focused the target — e.g. typing into a spreadsheet cell, a Figma "
                "text layer, or a canvas app's hidden input. Follow with "
                "browser_press_key 'enter' to commit. For plain DOM inputs, "
                "browser_type is more precise."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to insert at the focused element."},
                    "tab_id": _TAB_ID_PROP,
                },
                "required": ["text"],
            },
            handler=_browser_insert_text,
        ),
        ToolDef(
            name="browser_paste",
            description=(
                "Paste text at the focused element as a real clipboard paste. The "
                "spreadsheet superpower: send tab-separated columns and "
                "newline-separated rows and Google Sheets/Excel split it into a "
                "cell range in ONE action — far better than typing cell by cell. "
                "Click the top-left target cell first (Name Box or browser_click_at), "
                "then paste, then verify with a screenshot."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to paste. TSV (tabs + newlines) becomes a cell range in spreadsheets.",
                    },
                    "tab_id": _TAB_ID_PROP,
                },
                "required": ["text"],
            },
            handler=_browser_paste,
        ),
    ]
