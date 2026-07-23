"""browser_* tools: schemas, bridge discovery, happy paths, and error strings.

The bridge HTTP layer is faked by monkeypatching ``_resolve_bridge_candidates``
(fixed endpoint list) and ``httpx.AsyncClient`` (request recorder) — no real
server, same style as the other harness tool tests.
"""

import inspect
import json
import os

import httpx
import pytest

from cowork.harnesses.anton_harness import browser_tools
from cowork.harnesses.anton_harness.browser_tools import build_browser_tools


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def _patch_client(monkeypatch, responder):
    """Swap httpx.AsyncClient for a recorder (discovery untouched).

    ``responder(method, path, params, json_body)`` returns ``(payload, status)``
    or an Exception instance to raise. Returns the list of captured requests
    (each includes the client kwargs: base_url, headers, timeout).
    """
    captured = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def request(self, method, path, params=None, json=None):
            captured.append(
                {"method": method, "path": path, "params": params, "json": json, **self.kwargs}
            )
            result = responder(method, path, params, json)
            if isinstance(result, Exception):
                raise result
            payload, status = result
            return _FakeResponse(payload, status)

    monkeypatch.setattr(browser_tools.httpx, "AsyncClient", FakeClient)
    return captured


def _patch_bridge(monkeypatch, responder):
    """Point discovery at a single fake bridge candidate and swap in the recorder."""
    monkeypatch.setattr(
        browser_tools, "_resolve_bridge_candidates",
        lambda: [("http://127.0.0.1:39999", "test-token")],
    )
    return _patch_client(monkeypatch, responder)


def _stale_env_plus_fresh_file(monkeypatch, tmp_path):
    """Real discovery in the ENG-439 shape: a stale env candidate (frozen at
    server spawn) and a fresh file candidate (rewritten by the relaunched app,
    pid alive)."""
    monkeypatch.setenv("COWORK_BROWSER_BRIDGE_PORT", "1111")
    monkeypatch.setenv("COWORK_BROWSER_BRIDGE_TOKEN", "stale-tok")
    monkeypatch.setenv("COWORK_HOME", str(tmp_path))
    (tmp_path / "browser-bridge.json").write_text(
        json.dumps({"port": 2222, "token": "fresh-tok", "pid": os.getpid()})
    )


def _tool(name):
    return next(t for t in build_browser_tools() if t.name == name)


# ---------------------------------------------------------------------------
# Schema / builder shape
# ---------------------------------------------------------------------------


class TestBuilderShape:
    EXPECTED = [
        "browser_navigate",
        "browser_tabs",
        "browser_read",
        "browser_snapshot",
        "browser_click",
        "browser_type",
        "browser_scroll",
        "browser_back",
        "browser_screenshot",
        "browser_close_tab",
        "browser_open_app",
        "browser_click_at",
        "browser_press_key",
        "browser_insert_text",
        "browser_paste",
    ]

    def test_names_exact(self):
        assert [t.name for t in build_browser_tools()] == self.EXPECTED

    def test_schemas_are_strict_objects(self):
        for tool in build_browser_tools():
            schema = tool.input_schema
            assert schema["type"] == "object", tool.name
            assert isinstance(schema["properties"], dict), tool.name
            assert inspect.iscoroutinefunction(tool.handler), tool.name
            assert tool.description, tool.name

    def test_required_arrays(self):
        assert _tool("browser_navigate").input_schema["required"] == ["url"]
        assert _tool("browser_click").input_schema["required"] == ["index"]
        assert _tool("browser_type").input_schema["required"] == ["index", "text"]
        assert _tool("browser_scroll").input_schema["required"] == ["direction"]

    def test_snake_case_args_and_enum(self):
        assert "new_tab" in _tool("browser_navigate").input_schema["properties"]
        assert "max_chars" in _tool("browser_read").input_schema["properties"]
        for name in self.EXPECTED[2:]:  # everything but navigate/tabs/open_app takes tab_id
            if name == "browser_open_app":
                assert "name" in _tool(name).input_schema["properties"]
                continue
            assert "tab_id" in _tool(name).input_schema["properties"], name
        assert _tool("browser_scroll").input_schema["properties"]["direction"]["enum"] == [
            "up", "down", "top", "bottom",
        ]

    def test_new_args_are_optional(self):
        navigate = _tool("browser_navigate").input_schema
        assert navigate["properties"]["background"]["type"] == "boolean"
        assert navigate["required"] == ["url"]
        for name in ("browser_click", "browser_type"):
            schema = _tool(name).input_schema
            assert schema["properties"]["snapshot_v"]["type"] == "integer", name
            assert "snapshot_v" not in schema["required"], name

    def test_prompts_only_where_intended(self):
        for tool in build_browser_tools():
            if tool.name == "browser_navigate":
                assert tool.prompt and "browser_read" in tool.prompt
            elif tool.name == "browser_click_at":
                assert tool.prompt and "canvas" in tool.prompt.lower()
            else:
                assert tool.prompt is None, tool.name

    def test_new_input_tools_schemas(self):
        click_at = _tool("browser_click_at").input_schema
        assert click_at["required"] == ["x", "y"]
        assert click_at["properties"]["x"]["type"] == "number"
        press = _tool("browser_press_key").input_schema
        assert press["required"] == ["key"]
        assert press["properties"]["modifiers"]["items"]["enum"] == ["cmd", "ctrl", "alt", "shift"]
        assert _tool("browser_insert_text").input_schema["required"] == ["text"]
        assert _tool("browser_paste").input_schema["required"] == ["text"]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_env_vars_first_candidate(self, monkeypatch, tmp_path):
        monkeypatch.setenv("COWORK_BROWSER_BRIDGE_PORT", "1234")
        monkeypatch.setenv("COWORK_BROWSER_BRIDGE_TOKEN", "env-tok")
        monkeypatch.setenv("COWORK_HOME", str(tmp_path))  # no discovery file
        assert browser_tools._resolve_bridge_candidates() == [
            ("http://127.0.0.1:1234", "env-tok"),
        ]

    def test_env_precedes_file_candidate(self, monkeypatch, tmp_path):
        monkeypatch.setenv("COWORK_BROWSER_BRIDGE_PORT", "1234")
        monkeypatch.setenv("COWORK_BROWSER_BRIDGE_TOKEN", "env-tok")
        monkeypatch.setenv("COWORK_HOME", str(tmp_path))
        (tmp_path / "browser-bridge.json").write_text(
            json.dumps({"port": 4321, "token": "file-tok", "pid": os.getpid()})
        )
        assert browser_tools._resolve_bridge_candidates() == [
            ("http://127.0.0.1:1234", "env-tok"),
            ("http://127.0.0.1:4321", "file-tok"),
        ]

    def test_identical_env_and_file_deduped(self, monkeypatch, tmp_path):
        monkeypatch.setenv("COWORK_BROWSER_BRIDGE_PORT", "1234")
        monkeypatch.setenv("COWORK_BROWSER_BRIDGE_TOKEN", "same-tok")
        monkeypatch.setenv("COWORK_HOME", str(tmp_path))
        (tmp_path / "browser-bridge.json").write_text(
            json.dumps({"port": 1234, "token": "same-tok", "pid": os.getpid()})
        )
        assert browser_tools._resolve_bridge_candidates() == [
            ("http://127.0.0.1:1234", "same-tok"),
        ]

    def test_discovery_file_via_cowork_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("COWORK_BROWSER_BRIDGE_PORT", raising=False)
        monkeypatch.delenv("COWORK_BROWSER_BRIDGE_TOKEN", raising=False)
        monkeypatch.setenv("COWORK_HOME", str(tmp_path))
        (tmp_path / "browser-bridge.json").write_text(
            json.dumps({"port": 4321, "token": "file-tok", "pid": os.getpid()})
        )
        assert browser_tools._resolve_bridge_candidates() == [
            ("http://127.0.0.1:4321", "file-tok"),
        ]

    def test_file_without_pid_is_accepted(self, monkeypatch, tmp_path):
        monkeypatch.delenv("COWORK_BROWSER_BRIDGE_PORT", raising=False)
        monkeypatch.delenv("COWORK_BROWSER_BRIDGE_TOKEN", raising=False)
        monkeypatch.setenv("COWORK_HOME", str(tmp_path))
        (tmp_path / "browser-bridge.json").write_text(
            json.dumps({"port": 4321, "token": "file-tok"})
        )
        assert browser_tools._resolve_bridge_candidates() == [
            ("http://127.0.0.1:4321", "file-tok"),
        ]

    def test_file_with_dead_pid_is_skipped(self, monkeypatch, tmp_path):
        monkeypatch.delenv("COWORK_BROWSER_BRIDGE_PORT", raising=False)
        monkeypatch.delenv("COWORK_BROWSER_BRIDGE_TOKEN", raising=False)
        monkeypatch.setenv("COWORK_HOME", str(tmp_path))
        (tmp_path / "browser-bridge.json").write_text(
            json.dumps({"port": 4321, "token": "file-tok", "pid": 424242})
        )

        def _dead_kill(pid, sig):
            raise ProcessLookupError

        monkeypatch.setattr(os, "kill", _dead_kill)
        assert browser_tools._resolve_bridge_candidates() == []

    def test_missing_file_returns_empty_not_crash(self, monkeypatch, tmp_path):
        monkeypatch.delenv("COWORK_BROWSER_BRIDGE_PORT", raising=False)
        monkeypatch.delenv("COWORK_BROWSER_BRIDGE_TOKEN", raising=False)
        monkeypatch.setenv("COWORK_HOME", str(tmp_path))  # no browser-bridge.json
        assert browser_tools._resolve_bridge_candidates() == []

    def test_corrupt_file_returns_empty_not_crash(self, monkeypatch, tmp_path):
        monkeypatch.delenv("COWORK_BROWSER_BRIDGE_PORT", raising=False)
        monkeypatch.delenv("COWORK_BROWSER_BRIDGE_TOKEN", raising=False)
        monkeypatch.setenv("COWORK_HOME", str(tmp_path))
        (tmp_path / "browser-bridge.json").write_text("not json {")
        assert browser_tools._resolve_bridge_candidates() == []


# ---------------------------------------------------------------------------
# Candidate fallthrough (ENG-439: stale env must not shadow the fresh file)
# ---------------------------------------------------------------------------


class TestCandidateFallthrough:
    @pytest.mark.asyncio
    async def test_dead_env_falls_through_to_live_file(self, monkeypatch, tmp_path):
        _stale_env_plus_fresh_file(monkeypatch, tmp_path)
        attempts = []

        def responder(m, p, params, body):
            attempts.append(1)
            if len(attempts) == 1:
                return httpx.ConnectError("connection refused")
            return ({"tabs": [], "activeTabId": None}, 200)

        calls = _patch_client(monkeypatch, responder)
        result = await browser_tools._browser_tabs(session=None, tc_input={})
        assert "No tabs are open" in result
        assert len(calls) == 2
        # first the stale env candidate, then the fresh file candidate
        assert calls[0]["base_url"] == "http://127.0.0.1:1111"
        assert calls[0]["headers"]["Authorization"] == "Bearer stale-tok"
        assert calls[1]["base_url"] == "http://127.0.0.1:2222"
        assert calls[1]["headers"]["Authorization"] == "Bearer fresh-tok"

    @pytest.mark.asyncio
    async def test_auth_failure_falls_through_to_next_candidate(self, monkeypatch, tmp_path):
        _stale_env_plus_fresh_file(monkeypatch, tmp_path)
        attempts = []

        def responder(m, p, params, body):
            attempts.append(1)
            if len(attempts) == 1:
                return ({"error": "unauthorized"}, 401)
            return ({"title": "T", "url": "u", "text": "ok"}, 200)

        calls = _patch_client(monkeypatch, responder)
        result = await browser_tools._browser_read(session=None, tc_input={})
        assert "T\nu\n\nok" in result
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_all_candidates_dead_returns_unavailable(self, monkeypatch, tmp_path):
        _stale_env_plus_fresh_file(monkeypatch, tmp_path)
        calls = _patch_client(monkeypatch, lambda *a: httpx.ConnectError("connection refused"))
        result = await browser_tools._browser_tabs(session=None, tc_input={})
        assert "desktop browser is unavailable" in result
        # both candidates were tried before giving up
        assert [c["base_url"] for c in calls] == [
            "http://127.0.0.1:1111",
            "http://127.0.0.1:2222",
        ]

    @pytest.mark.asyncio
    async def test_live_bridge_4xx_does_not_fall_through(self, monkeypatch, tmp_path):
        _stale_env_plus_fresh_file(monkeypatch, tmp_path)
        calls = _patch_client(monkeypatch, lambda *a: ({"error": "no such tab"}, 404))
        result = await browser_tools._browser_snapshot(session=None, tc_input={"tab_id": "nope"})
        assert result == "Browser error: no such tab"
        assert len(calls) == 1  # real error from a live bridge — no retry

    @pytest.mark.asyncio
    async def test_dead_pid_file_candidate_never_called(self, monkeypatch, tmp_path):
        monkeypatch.delenv("COWORK_BROWSER_BRIDGE_PORT", raising=False)
        monkeypatch.delenv("COWORK_BROWSER_BRIDGE_TOKEN", raising=False)
        monkeypatch.setenv("COWORK_HOME", str(tmp_path))
        (tmp_path / "browser-bridge.json").write_text(
            json.dumps({"port": 4321, "token": "file-tok", "pid": 424242})
        )

        def _dead_kill(pid, sig):
            raise ProcessLookupError

        monkeypatch.setattr(os, "kill", _dead_kill)
        calls = _patch_client(monkeypatch, lambda *a: ({"tabs": []}, 200))
        result = await browser_tools._browser_tabs(session=None, tc_input={})
        assert "desktop browser is unavailable" in result
        assert calls == []  # stale file from a crashed app — skipped up front


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestNavigate:
    @pytest.mark.asyncio
    async def test_happy_path(self, monkeypatch):
        calls = _patch_bridge(
            monkeypatch,
            lambda m, p, params, body: ({"tabId": "t1", "url": "https://example.com"}, 200),
        )
        result = await browser_tools._browser_navigate(
            session=None, tc_input={"url": "example.com", "new_tab": True}
        )
        assert "Navigated to https://example.com (tabId: t1)" in result
        call = calls[0]
        assert call["method"] == "POST" and call["path"] == "/navigate"
        # snake_case args map to the bridge's camelCase payload
        assert call["json"] == {"url": "example.com", "newTab": True}
        assert call["headers"]["Authorization"] == "Bearer test-token"
        assert call["timeout"] == 30.0

    @pytest.mark.asyncio
    async def test_missing_url_is_arg_error_not_bridge_call(self, monkeypatch):
        calls = _patch_bridge(monkeypatch, lambda *a: ({}, 200))
        result = await browser_tools._browser_navigate(session=None, tc_input={})
        assert result == "browser_navigate: `url` is required."
        assert calls == []

    @pytest.mark.asyncio
    async def test_background_new_tab_uses_tabs_endpoint_without_activate(self, monkeypatch):
        calls = _patch_bridge(
            monkeypatch,
            lambda m, p, params, body: ({"tabId": "t7"}, 200),
        )
        result = await browser_tools._browser_navigate(
            session=None,
            tc_input={"url": "example.com", "new_tab": True, "background": True},
        )
        assert "Opened background tab" in result and "tabId: t7" in result
        call = calls[0]
        assert call["method"] == "POST" and call["path"] == "/tabs"
        assert call["json"] == {"url": "example.com", "activate": False}
        assert call["timeout"] == 30.0

    @pytest.mark.asyncio
    async def test_background_without_new_tab_still_uses_navigate(self, monkeypatch):
        calls = _patch_bridge(
            monkeypatch,
            lambda m, p, params, body: ({"tabId": "t1", "url": "https://example.com"}, 200),
        )
        result = await browser_tools._browser_navigate(
            session=None, tc_input={"url": "example.com", "background": True}
        )
        assert "Navigated to https://example.com" in result
        assert calls[0]["path"] == "/navigate"
        assert calls[0]["json"] == {"url": "example.com"}


class TestTabs:
    @pytest.mark.asyncio
    async def test_numbered_list_with_active_marker(self, monkeypatch):
        state = {
            "activeTabId": "b",
            "tabs": [
                {"id": "a", "title": "Pricing", "url": "https://x.com/pricing", "isLoading": False},
                {"id": "b", "title": "", "url": "", "isLoading": True},
            ],
        }
        _patch_bridge(monkeypatch, lambda m, p, params, body: (state, 200))
        result = await browser_tools._browser_tabs(session=None, tc_input={})
        assert result.splitlines() == [
            "Open tabs:",
            "1. Pricing — https://x.com/pricing (tabId: a)",
            "2. [active] New tab — (blank) (loading…) (tabId: b)",
        ]


class TestRead:
    @pytest.mark.asyncio
    async def test_pass_through_and_param_mapping(self, monkeypatch):
        calls = _patch_bridge(
            monkeypatch,
            lambda m, p, params, body: (
                {"title": "Hello", "url": "https://x.com", "text": "body text"}, 200,
            ),
        )
        result = await browser_tools._browser_read(
            session=None, tc_input={"tab_id": "t9", "max_chars": 500}
        )
        # Untrusted-content wrapper frames the page text as data (P6).
        assert result.startswith('<untrusted-page-content source="https://x.com">')
        assert "Hello\nhttps://x.com\n\nbody text" in result
        assert result.endswith("</untrusted-page-content>")
        assert calls[0]["params"] == {"tabId": "t9", "maxChars": 500}

    @pytest.mark.asyncio
    async def test_long_text_truncated_politely(self, monkeypatch):
        _patch_bridge(
            monkeypatch,
            lambda m, p, params, body: (
                {"title": "T", "url": "u", "text": "x" * 40000}, 200,
            ),
        )
        result = await browser_tools._browser_read(session=None, tc_input={})
        assert "truncated" in result
        assert len(result) < 31000


class TestSnapshot:
    @pytest.mark.asyncio
    async def test_element_lines(self, monkeypatch):
        payload = {
            "title": "Home",
            "url": "https://x.com",
            "elements": [
                {"index": 3, "tag": "a", "role": "link", "text": "Pricing", "href": "/pricing"},
                {"index": 7, "tag": "input", "role": None, "text": "Search", "inputType": "text"},
                {"index": 12, "tag": "button", "role": "button", "text": "Sign in"},
            ],
        }
        _patch_bridge(monkeypatch, lambda m, p, params, body: (payload, 200))
        result = await browser_tools._browser_snapshot(session=None, tc_input={})
        assert result.startswith('<untrusted-page-content source="https://x.com">')
        lines = result.splitlines()
        assert "Home" in lines
        assert "[3] link 'Pricing' -> /pricing" in lines
        assert "[7] input(text) 'Search'" in lines
        assert "[12] button 'Sign in'" in lines

    @pytest.mark.asyncio
    async def test_header_includes_version_token(self, monkeypatch):
        payload = {
            "title": "Home",
            "url": "https://x.com",
            "v": 3,
            "elements": [{"index": 3, "tag": "a", "role": "link", "text": "Pricing"}],
        }
        _patch_bridge(monkeypatch, lambda m, p, params, body: (payload, 200))
        result = await browser_tools._browser_snapshot(session=None, tc_input={})
        assert "snapshot v=3 — 1 interactive elements" in result
        assert "snapshot_v=3" in result


class TestMutations:
    @pytest.mark.asyncio
    async def test_click_sends_index_and_slow_timeout(self, monkeypatch):
        calls = _patch_bridge(monkeypatch, lambda m, p, params, body: ({"ok": True}, 200))
        result = await browser_tools._browser_click(
            session=None, tc_input={"index": 0, "tab_id": "t1"}
        )
        assert result.startswith("Clicked element [0].")
        # The supervised-mode gate pre-fetches a snapshot for classification —
        # the action call is no longer calls[0]; select it by path.
        click_call = next(c for c in calls if c["path"] == "/click")
        assert click_call["json"] == {"tabId": "t1", "index": 0}
        assert click_call["timeout"] == 30.0

    @pytest.mark.asyncio
    async def test_click_passes_snapshot_v_as_v(self, monkeypatch):
        calls = _patch_bridge(monkeypatch, lambda m, p, params, body: ({"ok": True}, 200))
        result = await browser_tools._browser_click(
            session=None, tc_input={"index": 0, "tab_id": "t1", "snapshot_v": 3}
        )
        assert result.startswith("Clicked element [0].")
        click_call = next(c for c in calls if c["path"] == "/click")
        assert click_call["json"] == {"tabId": "t1", "index": 0, "v": 3}

    @pytest.mark.asyncio
    async def test_click_missing_index_is_arg_error(self, monkeypatch):
        calls = _patch_bridge(monkeypatch, lambda *a: ({}, 200))
        result = await browser_tools._browser_click(session=None, tc_input={})
        assert result.startswith("browser_click: `index`")
        assert calls == []

    @pytest.mark.asyncio
    async def test_type_payload(self, monkeypatch):
        calls = _patch_bridge(monkeypatch, lambda m, p, params, body: ({"ok": True}, 200))
        result = await browser_tools._browser_type(
            session=None, tc_input={"index": 4, "text": "hello", "submit": True}
        )
        assert "Typed into element [4] and submitted (Enter)." in result
        assert calls[0]["json"] == {"index": 4, "text": "hello", "submit": True}

    @pytest.mark.asyncio
    async def test_type_passes_snapshot_v_as_v(self, monkeypatch):
        calls = _patch_bridge(monkeypatch, lambda m, p, params, body: ({"ok": True}, 200))
        result = await browser_tools._browser_type(
            session=None, tc_input={"index": 4, "text": "hello", "snapshot_v": 2}
        )
        assert "Typed into element [4]." in result
        assert calls[0]["json"]["v"] == 2

    @pytest.mark.asyncio
    async def test_scroll_validates_direction(self, monkeypatch):
        calls = _patch_bridge(monkeypatch, lambda *a: ({}, 200))
        result = await browser_tools._browser_scroll(
            session=None, tc_input={"direction": "sideways"}
        )
        assert result.startswith("browser_scroll: `direction` must be one of")
        assert calls == []

    @pytest.mark.asyncio
    async def test_screenshot_returns_path(self, monkeypatch):
        _patch_bridge(monkeypatch, lambda m, p, params, body: ({"path": "/tmp/shot.png"}, 200))
        result = await browser_tools._browser_screenshot(session=None, tc_input={})
        assert result == (
            "Screenshot saved to: /tmp/shot.png\n"
            "View it with the read_image tool using this exact path."
        )

    @pytest.mark.asyncio
    async def test_close_tab_defaults_to_active(self, monkeypatch):
        calls = _patch_bridge(monkeypatch, lambda m, p, params, body: ({"ok": True}, 200))
        result = await browser_tools._browser_close_tab(session=None, tc_input={})
        assert result == "Closed the active tab."
        # no tabId key at all — None values are stripped from the payload
        assert calls[0]["json"] == {}


class TestBack:
    @pytest.mark.asyncio
    async def test_plain_ok_reports_went_back(self, monkeypatch):
        _patch_bridge(monkeypatch, lambda m, p, params, body: ({"ok": True}, 200))
        result = await browser_tools._browser_back(session=None, tc_input={})
        assert result.startswith("Went back to the previous page.")

    @pytest.mark.asyncio
    async def test_moved_false_reports_earliest_page(self, monkeypatch):
        _patch_bridge(
            monkeypatch,
            lambda m, p, params, body: ({"ok": True, "moved": False}, 200),
        )
        result = await browser_tools._browser_back(session=None, tc_input={})
        assert result == "Already at the earliest page in this tab's history."


# ---------------------------------------------------------------------------
# Error paths — never raise into the agent loop
# ---------------------------------------------------------------------------


class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_no_discovery_returns_unavailable_string(self, monkeypatch):
        monkeypatch.setattr(browser_tools, "_resolve_bridge_candidates", lambda: [])
        for handler, tc_input in [
            (browser_tools._browser_navigate, {"url": "https://x.com"}),
            (browser_tools._browser_tabs, {}),
            (browser_tools._browser_read, {}),
            (browser_tools._browser_snapshot, {}),
            (browser_tools._browser_click, {"index": 1}),
            (browser_tools._browser_type, {"index": 1, "text": "x"}),
            (browser_tools._browser_scroll, {"direction": "down"}),
            (browser_tools._browser_back, {}),
            (browser_tools._browser_screenshot, {}),
            (browser_tools._browser_close_tab, {}),
        ]:
            result = await handler(session=None, tc_input=tc_input)
            assert "desktop browser is unavailable" in result, handler.__name__
            assert "don't retry in a loop" in result, handler.__name__

    @pytest.mark.asyncio
    async def test_connection_error_returns_unavailable_string(self, monkeypatch):
        _patch_bridge(monkeypatch, lambda *a: httpx.ConnectError("connection refused"))
        result = await browser_tools._browser_tabs(session=None, tc_input={})
        assert "desktop browser is unavailable" in result

    @pytest.mark.asyncio
    async def test_timeout_returns_unavailable_string(self, monkeypatch):
        _patch_bridge(monkeypatch, lambda *a: httpx.ReadTimeout("too slow"))
        result = await browser_tools._browser_read(session=None, tc_input={})
        assert "desktop browser is unavailable" in result

    @pytest.mark.asyncio
    async def test_bridge_error_payload(self, monkeypatch):
        _patch_bridge(monkeypatch, lambda *a: ({"error": "no such tab"}, 404))
        result = await browser_tools._browser_snapshot(session=None, tc_input={"tab_id": "nope"})
        assert result == "Browser error: no such tab"

    @pytest.mark.asyncio
    async def test_bridge_error_without_payload(self, monkeypatch):
        _patch_bridge(monkeypatch, lambda *a: ({}, 500))
        result = await browser_tools._browser_back(session=None, tc_input={})
        assert result == "Browser error: HTTP 500"


# ---------------------------------------------------------------------------
# Per-turn browser context (Browser Agent dock)
# ---------------------------------------------------------------------------


def _state_payload(tabs, active_id=None):
    return {"tabs": tabs, "activeTabId": active_id, "viewVisible": True}


class TestBuildBrowserTurnContext:
    @pytest.mark.asyncio
    async def test_context_has_guidance_and_live_tabs(self, monkeypatch):
        tabs = [
            {"id": "t1", "title": "BBC Sport - Scores, Fixtures, News", "url": "https://www.bbc.co.uk/sport", "isLoading": False},
            {"id": "t2", "title": "Shareable Online Calendar", "url": "https://cal.example.com", "isLoading": False},
        ]
        _patch_bridge(monkeypatch, lambda *a: (_state_payload(tabs, "t1"), 200))
        ctx = await browser_tools.build_browser_turn_context()
        # Copilot guidance: act on the live page, don't answer from memory.
        assert "browsing copilot" in ctx
        assert "browser_* tools" in ctx
        assert "LIVE page" in ctx
        # Live tab state with the active marker on the right tab.
        assert "1. [active] BBC Sport - Scores, Fixtures, News — https://www.bbc.co.uk/sport" in ctx
        assert "2. Shareable Online Calendar — https://cal.example.com" in ctx
        assert "tabId" not in ctx  # prompt-facing list stays human

    @pytest.mark.asyncio
    async def test_context_uses_short_timeout(self, monkeypatch):
        captured = _patch_bridge(monkeypatch, lambda *a: (_state_payload([]), 200))
        await browser_tools.build_browser_turn_context()
        # Request-path code must not stall a turn on a wedged bridge.
        assert captured[0]["timeout"] == browser_tools._CONTEXT_TIMEOUT
        assert captured[0]["path"] == "/state"

    @pytest.mark.asyncio
    async def test_context_with_no_tabs_offers_to_open_one(self, monkeypatch):
        _patch_bridge(monkeypatch, lambda *a: (_state_payload([]), 200))
        ctx = await browser_tools.build_browser_turn_context()
        assert "No tabs are open" in ctx
        assert "browser_navigate" in ctx

    @pytest.mark.asyncio
    async def test_context_truncates_long_tab_lists(self, monkeypatch):
        tabs = [
            {"id": f"t{i}", "title": f"Page {i}", "url": f"https://x{i}.example.com", "isLoading": False}
            for i in range(15)
        ]
        _patch_bridge(monkeypatch, lambda *a: (_state_payload(tabs, "t0"), 200))
        ctx = await browser_tools.build_browser_turn_context()
        assert "5 more" in ctx
        assert "Page 14" not in ctx

    @pytest.mark.asyncio
    async def test_context_empty_when_bridge_unreachable(self, monkeypatch):
        _patch_bridge(monkeypatch, lambda *a: httpx.ConnectError("connection refused"))
        assert await browser_tools.build_browser_turn_context() == ""

    @pytest.mark.asyncio
    async def test_context_empty_without_discovery(self, monkeypatch):
        monkeypatch.setattr(browser_tools, "_resolve_bridge_candidates", lambda: [])
        assert await browser_tools.build_browser_turn_context() == ""

    @pytest.mark.asyncio
    async def test_context_empty_on_bridge_error(self, monkeypatch):
        _patch_bridge(monkeypatch, lambda *a: ({"error": "boom"}, 500))
        assert await browser_tools.build_browser_turn_context() == ""


# ---------------------------------------------------------------------------
# Trusted-input tools (click_at / press_key / insert_text / paste)
# ---------------------------------------------------------------------------


class TestTrustedInputTools:
    @pytest.mark.asyncio
    async def test_click_at_payload_and_ok_text(self, monkeypatch):
        captured = _patch_bridge(monkeypatch, lambda *a: ({"ok": True}, 200))
        result = await browser_tools._browser_click_at(None, {"x": 120, "y": 340, "tab_id": "t1"})
        # The gate probes /inspect-point first; select the action call by path.
        at_call = next(c for c in captured if c["path"] == "/click-at")
        assert at_call["json"] == {"tabId": "t1", "x": 120, "y": 340}
        assert "trusted click" in result
        assert at_call["timeout"] == browser_tools._SLOW_TIMEOUT

    @pytest.mark.asyncio
    async def test_click_at_validates_coordinates(self):
        for bad in ({"x": "10", "y": 5}, {"x": 10}, {"y": 5}, {"x": True, "y": 5}, {}):
            result = await browser_tools._browser_click_at(None, bad)
            assert "x` and `y`" in result, bad

    @pytest.mark.asyncio
    async def test_press_key_payload_with_modifiers(self, monkeypatch):
        captured = _patch_bridge(monkeypatch, lambda *a: ({"ok": True}, 200))
        result = await browser_tools._browser_press_key(
            None, {"key": "a", "modifiers": ["cmd", "hyper"], "tab_id": "t1"}
        )
        # unknown modifiers are filtered out, cmd passes through
        assert captured[0]["path"] == "/press"
        assert captured[0]["json"] == {"tabId": "t1", "key": "a", "modifiers": ["cmd"]}
        assert "Pressed a with cmd" in result

    @pytest.mark.asyncio
    async def test_press_key_requires_key(self):
        assert "`key` is required" in await browser_tools._browser_press_key(None, {})
        assert "`key` is required" in await browser_tools._browser_press_key(None, {"key": "  "})

    @pytest.mark.asyncio
    async def test_insert_text_payload(self, monkeypatch):
        captured = _patch_bridge(monkeypatch, lambda *a: ({"ok": True}, 200))
        result = await browser_tools._browser_insert_text(None, {"text": "hello sheets"})
        assert captured[0]["path"] == "/insert-text"
        assert captured[0]["json"] == {"text": "hello sheets"}
        assert "12 characters" in result
        # insert-text is fast-timeout (no load settle)
        assert captured[0]["timeout"] == browser_tools._FAST_TIMEOUT

    @pytest.mark.asyncio
    async def test_insert_text_requires_text(self):
        assert "`text` is required" in await browser_tools._browser_insert_text(None, {})

    @pytest.mark.asyncio
    async def test_paste_payload_and_tsv_guidance(self, monkeypatch):
        captured = _patch_bridge(monkeypatch, lambda *a: ({"ok": True}, 200))
        result = await browser_tools._browser_paste(None, {"text": "a\tb\n1\t2", "tab_id": "t9"})
        assert captured[0]["path"] == "/paste"
        assert captured[0]["json"] == {"tabId": "t9", "text": "a\tb\n1\t2"}
        assert "cell range" in result
        assert captured[0]["timeout"] == browser_tools._SLOW_TIMEOUT

    @pytest.mark.asyncio
    async def test_paste_requires_text(self):
        assert "`text` is required" in await browser_tools._browser_paste(None, {})

    @pytest.mark.asyncio
    async def test_trusted_tools_degrade_gracefully(self, monkeypatch):
        monkeypatch.setattr(browser_tools, "_resolve_bridge_candidates", lambda: [])
        for handler, tc_input in [
            (browser_tools._browser_click_at, {"x": 1, "y": 1}),
            (browser_tools._browser_press_key, {"key": "enter"}),
            (browser_tools._browser_insert_text, {"text": "x"}),
            (browser_tools._browser_paste, {"text": "x"}),
        ]:
            result = await handler(None, tc_input)
            assert "desktop browser is unavailable" in result, handler.__name__


# ---------------------------------------------------------------------------
# browser_open_app + app annotations on browser_tabs
# ---------------------------------------------------------------------------

_APPS = [
    {"id": "app-mail.google.com", "name": "Gmail", "origin": "https://mail.google.com", "createdAt": 1},
    {"id": "app-linear.app", "name": "Linear", "origin": "https://linear.app", "createdAt": 2},
]


def _patch_apps_and_state(monkeypatch, tabs=None, open_result=None, apps_payload=None):
    """GET /apps returns the registry; everything else routes by path."""

    def responder(method, path, params, json_body):
        if path == "/apps":
            return (apps_payload if apps_payload is not None else _APPS, 200)
        if path == "/apps/open":
            return (open_result or {"tabId": "t-1", "created": False}, 200)
        if path == "/state":
            return ({"tabs": tabs or [], "activeTabId": None, "viewVisible": True}, 200)
        return ({"ok": True}, 200)

    return _patch_bridge(monkeypatch, responder)


class TestOpenApp:
    @pytest.mark.asyncio
    async def test_resolves_name_and_opens(self, monkeypatch):
        captured = _patch_apps_and_state(monkeypatch)
        result = await browser_tools._browser_open_app(None, {"name": "gmail"})
        posts = [c for c in captured if c["path"] == "/apps/open"]
        assert posts and posts[0]["json"] == {"appId": "app-mail.google.com"}
        assert "Gmail" in result
        assert "existing tab" in result

    @pytest.mark.asyncio
    async def test_created_variant_reports_fresh_pinned_tab(self, monkeypatch):
        _patch_apps_and_state(monkeypatch, open_result={"tabId": "t-9", "created": True})
        result = await browser_tools._browser_open_app(None, {"name": "Linear"})
        assert "fresh pinned tab" in result

    @pytest.mark.asyncio
    async def test_prefix_and_substring_matching(self, monkeypatch):
        _patch_apps_and_state(monkeypatch)
        for query in ("gma", "mail.google"):
            result = await browser_tools._browser_open_app(None, {"name": query})
            assert "Gmail" in result, query

    @pytest.mark.asyncio
    async def test_no_match_lists_known_apps(self, monkeypatch):
        _patch_apps_and_state(monkeypatch)
        result = await browser_tools._browser_open_app(None, {"name": "figma"})
        assert "No app matches 'figma'" in result
        assert "Gmail" in result and "Linear" in result

    @pytest.mark.asyncio
    async def test_name_required(self):
        assert "`name` is required" in await browser_tools._browser_open_app(None, {})

    @pytest.mark.asyncio
    async def test_apps_list_failure_returns_unavailable(self, monkeypatch):
        _patch_bridge(monkeypatch, lambda *a: httpx.ConnectError("down"))
        result = await browser_tools._browser_open_app(None, {"name": "gmail"})
        assert "desktop browser is unavailable" in result


class TestTabsAppAnnotation:
    @pytest.mark.asyncio
    async def test_matching_tabs_get_app_labels(self, monkeypatch):
        tabs = [
            {"id": "t1", "title": "Inbox", "url": "https://mail.google.com/mail/u/0/#inbox", "isLoading": False},
            {"id": "t2", "title": "News", "url": "https://www.bbc.co.uk/sport", "isLoading": False},
        ]
        async def fake_bridge_call(*a, **k):
            return {"tabs": tabs, "activeTabId": None, "apps": _APPS}

        monkeypatch.setattr(browser_tools, "_bridge_call", fake_bridge_call)
        result = await browser_tools._browser_tabs(None, {})
        assert "[app: Gmail]" in result
        assert "bbc.co.uk/sport (tabId: t2)" in result
        assert result.count("[app:") == 1

    @pytest.mark.asyncio
    async def test_tabs_without_apps_field_still_list(self, monkeypatch):
        async def fake_bridge_call(*a, **k):
            return {"tabs": [{"id": "t1", "title": "N", "url": "https://x.com", "isLoading": False}], "activeTabId": None}

        monkeypatch.setattr(browser_tools, "_bridge_call", fake_bridge_call)
        result = await browser_tools._browser_tabs(None, {})
        assert "Open tabs:" in result
        assert "[app:" not in result


class TestMatchAppRobustness:
    def test_null_fields_never_raise(self):
        apps = [
            {"id": None, "name": None, "origin": None},
            {"id": "app-x", "name": "X", "origin": "https://x.com"},
        ]
        assert browser_tools._match_app(apps, "anything") == apps[1] or browser_tools._match_app(apps, "anything") is None
        assert browser_tools._match_app(apps, "x") == apps[1]
        assert browser_tools._match_app(apps, "") is None
