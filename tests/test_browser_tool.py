"""Tests for the WS3/WS5 browser-control agent tool.

Covers `build_cowork_browser_tool` (the ToolDef contract), the total,
non-raising `_cowork_browser_control` handler (input validation, dispatch,
WS4 result_code → canonical BrowserErrorKind mapping, the observed-result
"no false success" guard, the JSON envelope + ERROR: prefix), the
browser-aware stream classifier, and the content-free Langfuse span (WS5-T3).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from cowork.harnesses.anton_harness import tools as tools_mod
from cowork.harnesses.anton_harness import browser_telemetry as bt
from cowork.harnesses.anton_harness.stream_formatter import classify_browser_status
from cowork.schemas.browser import (
    BrowserActionType,
    BrowserErrorKind,
    BrowserToolVerdict,
    ResultCode,
)
from cowork.services.browser.client import OPEN_URL_NOT_ANCHORED_DETAIL


# ── fakes ────────────────────────────────────────────────────────────
class _FakeSession:
    def __init__(self, conversation_id="conv-123"):
        self._session_id = conversation_id


def _run(coro):
    return asyncio.run(coro)


def _install_fake_bridge(monkeypatch, verdict=None, *, capture=None, raises=None):
    async def _fake_send(
        conversation_id, action, *, href=None, direction=None, user_texts=None
    ):
        if capture is not None:
            capture.append(
                {
                    "conversation_id": conversation_id,
                    "action": action,
                    "href": href,
                    "direction": direction,
                    "user_texts": user_texts,
                }
            )
        if raises is not None:
            raise raises
        return verdict

    monkeypatch.setattr(tools_mod, "_get_bridge_client", lambda: _fake_send)


def _handle(tc_input, session=None):
    session = session or _FakeSession()
    return _run(tools_mod._cowork_browser_control(session, tc_input))


# ── ToolDef contract (WS3-T2) ─────────────────────────────────────────
class TestToolDef:
    def test_name_and_required(self):
        td = tools_mod.build_cowork_browser_tool()
        assert td.name == "browser_control"
        assert td.input_schema["required"] == [
            "action",
            "reason",
            "progress_message",
        ]

    def test_action_enum_is_read_only(self):
        td = tools_mod.build_cowork_browser_tool()
        enum = td.input_schema["properties"]["action"]["enum"]
        assert enum == ["inspect", "follow_link", "scroll", "wait", "open_url"]
        for banned in ("click", "type", "submit", "download", "upload"):
            assert banned not in enum

    def test_prompt_biases_connector_first(self):
        td = tools_mod.build_cowork_browser_tool()
        assert "lookup_connector" in td.prompt
        assert "READ-ONLY" in td.prompt

    def test_open_url_schema_and_prompt_user_directed_rule(self):
        # open_url is in the enum, the `url` field exists and pins the
        # user-directed rule, and the prompt states the user's instruction
        # IS the approval (never on the agent's own initiative).
        td = tools_mod.build_cowork_browser_tool()
        props = td.input_schema["properties"]
        assert "open_url" in props["action"]["enum"]
        assert "url" in props
        assert "explicitly asked" in props["url"]["description"]
        assert "never on your own initiative" in props["url"]["description"]
        assert "open_url" in td.prompt
        assert "explicitly asks" in td.prompt
        assert "ask them instead of guessing" in td.prompt
        assert "same-site only" in td.prompt

    def test_prompt_states_only_setup_path_no_extension(self):
        # A1: the description must pin the ONLY setup path (the shared
        # in-app connect-flow steps) and explicitly rule out a Chrome
        # extension / toolbar icon, so the model relays the real steps
        # instead of hallucinating a nonexistent extension flow.
        from cowork.services.browser import BROWSER_CONNECT_FLOW_STEPS

        td = tools_mod.build_cowork_browser_tool()
        assert BROWSER_CONNECT_FLOW_STEPS in td.prompt
        assert "NO Chrome extension" in td.prompt
        assert "NO toolbar icon" in td.prompt


# ── session tool gate (WS3-T2) ─────────────────────────────────────────
class TestSessionToolGate:
    def _base(self):
        # Sentinel base tools — the gate only appends the browser tool.
        return ["A", "B"]

    def test_absent_when_disabled(self):
        from cowork.harnesses.anton_harness.harness import AntonHarness

        tools = AntonHarness._select_session_tools(
            self._base(), browser_enabled=False
        )
        names = [getattr(t, "name", t) for t in tools]
        assert "browser_control" not in names
        assert tools == ["A", "B"]

    def test_present_when_enabled(self):
        from cowork.harnesses.anton_harness.harness import AntonHarness

        tools = AntonHarness._select_session_tools(
            self._base(), browser_enabled=True
        )
        names = [getattr(t, "name", t) for t in tools]
        assert "browser_control" in names

    def test_base_tools_not_mutated(self):
        from cowork.harnesses.anton_harness.harness import AntonHarness

        base = self._base()
        AntonHarness._select_session_tools(base, browser_enabled=True)
        assert base == ["A", "B"]


# ── input validation (bridge NOT touched) ─────────────────────────────
class TestValidation:
    def test_missing_action(self, monkeypatch):
        cap = []
        _install_fake_bridge(monkeypatch, capture=cap)
        out = _handle({"reason": "r", "progress_message": "p"})
        env = json.loads(out)
        assert env["status"] == BrowserErrorKind.unsupported_action.value
        assert cap == []  # bridge never called

    def test_bad_action(self, monkeypatch):
        cap = []
        _install_fake_bridge(monkeypatch, capture=cap)
        out = _handle({"action": "click", "reason": "r", "progress_message": "p"})
        env = json.loads(out)
        assert env["status"] == BrowserErrorKind.unsupported_action.value
        assert cap == []

    def test_missing_reason(self, monkeypatch):
        cap = []
        _install_fake_bridge(monkeypatch, capture=cap)
        out = _handle({"action": "inspect", "progress_message": "p"})
        env = json.loads(out)
        assert env["status"] == BrowserErrorKind.unsupported_action.value
        assert cap == []

    def test_missing_progress_message(self, monkeypatch):
        cap = []
        _install_fake_bridge(monkeypatch, capture=cap)
        out = _handle({"action": "inspect", "reason": "r"})
        env = json.loads(out)
        assert env["status"] == BrowserErrorKind.unsupported_action.value
        assert cap == []

    def test_follow_link_requires_href(self, monkeypatch):
        cap = []
        _install_fake_bridge(monkeypatch, capture=cap)
        out = _handle(
            {"action": "follow_link", "reason": "r", "progress_message": "p"}
        )
        env = json.loads(out)
        assert env["status"] == BrowserErrorKind.navigation_failed.value
        assert cap == []

    def test_open_url_requires_url(self, monkeypatch):
        cap = []
        _install_fake_bridge(monkeypatch, capture=cap)
        out = _handle(
            {"action": "open_url", "reason": "r", "progress_message": "p"}
        )
        env = json.loads(out)
        assert env["status"] == BrowserErrorKind.navigation_failed.value
        assert cap == []

    def test_open_url_rejects_non_http_url(self, monkeypatch):
        cap = []
        _install_fake_bridge(monkeypatch, capture=cap)
        for bad in ("ftp://example.com", "javascript:alert(1)", "not a url"):
            out = _handle(
                {
                    "action": "open_url",
                    "url": bad,
                    "reason": "r",
                    "progress_message": "p",
                }
            )
            env = json.loads(out)
            assert env["status"] == BrowserErrorKind.navigation_failed.value
        assert cap == []

    def test_no_conversation(self, monkeypatch):
        cap = []
        _install_fake_bridge(monkeypatch, capture=cap)
        out = _handle(
            {"action": "inspect", "reason": "r", "progress_message": "p"},
            session=_FakeSession(conversation_id=None),
        )
        assert out.startswith("ERROR:")
        env = json.loads(out[len("ERROR:"):].strip())
        assert env["status"] == BrowserErrorKind.bridge_disconnected.value
        assert cap == []


# ── happy path + observed guard ────────────────────────────────────────
class TestDispatch:
    def test_ok_with_observed(self, monkeypatch):
        verdict = BrowserToolVerdict(
            result_code=ResultCode.ok,
            action_type=BrowserActionType.inspect,
            observed={"http_status": 200, "text": "hi"},
            citations=[{"n": 1}],
            domain="example.com",
            action_id="cmd-1",
        )
        _install_fake_bridge(monkeypatch, verdict=verdict)
        out = _handle(
            {"action": "inspect", "reason": "r", "progress_message": "p"}
        )
        env = json.loads(out)
        assert env["status"] == "ok"
        assert env["observed"]["http_status"] == 200
        assert env["citations"] == [{"n": 1}]
        assert env["domain"] == "example.com"

    def test_ok_without_observed_downgraded_inspect(self, monkeypatch):
        verdict = BrowserToolVerdict(
            result_code=ResultCode.ok,
            action_type=BrowserActionType.inspect,
            observed=None,
        )
        _install_fake_bridge(monkeypatch, verdict=verdict)
        out = _handle(
            {"action": "inspect", "reason": "r", "progress_message": "p"}
        )
        env = json.loads(out)
        # unobserved read → bridge_disconnected (never success)
        assert env["status"] == BrowserErrorKind.bridge_disconnected.value

    def test_ok_without_observed_downgraded_navigate(self, monkeypatch):
        verdict = BrowserToolVerdict(
            result_code=ResultCode.ok,
            action_type=BrowserActionType.navigate,
            observed=None,
        )
        _install_fake_bridge(monkeypatch, verdict=verdict)
        out = _handle(
            {
                "action": "follow_link",
                "href": "https://example.com/x",
                "reason": "r",
                "progress_message": "p",
            }
        )
        env = json.loads(out)
        assert env["status"] == BrowserErrorKind.navigation_failed.value

    def test_follow_link_passes_href(self, monkeypatch):
        cap = []
        verdict = BrowserToolVerdict(
            result_code=ResultCode.ok,
            action_type=BrowserActionType.navigate,
            observed={"final_domain": "example.com"},
        )
        _install_fake_bridge(monkeypatch, verdict=verdict, capture=cap)
        _handle(
            {
                "action": "follow_link",
                "href": "https://example.com/y",
                "reason": "r",
                "progress_message": "p",
            }
        )
        assert cap[0]["href"] == "https://example.com/y"
        assert cap[0]["direction"] is None

    def test_scroll_passes_direction_only(self, monkeypatch):
        cap = []
        verdict = BrowserToolVerdict(
            result_code=ResultCode.ok,
            action_type=BrowserActionType.scroll,
            observed={"settled": True},
        )
        _install_fake_bridge(monkeypatch, verdict=verdict, capture=cap)
        _handle(
            {
                "action": "scroll",
                "direction": "down",
                "reason": "r",
                "progress_message": "p",
            }
        )
        assert cap[0]["direction"] == "down"
        assert cap[0]["href"] is None

    def test_open_url_passes_url_as_href_with_user_texts(self, monkeypatch):
        cap = []
        verdict = BrowserToolVerdict(
            result_code=ResultCode.ok,
            action_type=BrowserActionType.open_url,
            observed={"final_domain": "bbc.co.uk"},
        )
        _install_fake_bridge(monkeypatch, verdict=verdict, capture=cap)
        session = _FakeSession()
        session._history = [
            {"role": "user", "content": "go to bbc.co.uk please"},
            {"role": "assistant", "content": "ok"},
        ]
        out = _handle(
            {
                "action": "open_url",
                "url": "https://bbc.co.uk/news",
                "reason": "r",
                "progress_message": "p",
            },
            session=session,
        )
        env = json.loads(out)
        assert env["status"] == "ok"
        assert cap[0]["action"] == "open_url"
        assert cap[0]["href"] == "https://bbc.co.uk/news"
        assert cap[0]["direction"] is None
        assert cap[0]["user_texts"] == ["go to bbc.co.uk please"]

    def test_open_url_denied_maps_to_permission_denied(self, monkeypatch):
        verdict = BrowserToolVerdict(
            result_code=ResultCode.permission_denied,
            action_type=BrowserActionType.open_url,
            detail=OPEN_URL_NOT_ANCHORED_DETAIL,
        )
        _install_fake_bridge(monkeypatch, verdict=verdict)
        out = _handle(
            {
                "action": "open_url",
                "url": "https://evil.com",
                "reason": "r",
                "progress_message": "p",
            }
        )
        env = json.loads(out)
        assert env["status"] == BrowserErrorKind.permission_denied.value
        assert "explicitly asked" in env["error"]

    def test_bridge_raises_is_caught(self, monkeypatch):
        _install_fake_bridge(monkeypatch, raises=RuntimeError("boom"))
        out = _handle(
            {"action": "inspect", "reason": "r", "progress_message": "p"}
        )
        assert out.startswith("ERROR:")
        env = json.loads(out[len("ERROR:"):].strip())
        assert env["status"] == BrowserErrorKind.bridge_disconnected.value

    def test_stopped_surfaced_as_distinct_control_state(self, monkeypatch):
        # A blocked (stopped) session is permission_denied at the error-kind
        # level, but the envelope carries the DISTINCT control_state so the UI
        # can render a stopped terminal state, not a generic denial.
        from cowork.schemas.browser import ControlState

        verdict = BrowserToolVerdict(
            result_code=ResultCode.permission_denied,
            action_type=BrowserActionType.inspect,
            detail="session stopped",
            control_state=ControlState.stopped,
        )
        _install_fake_bridge(monkeypatch, verdict=verdict)
        out = _handle({"action": "inspect", "reason": "r", "progress_message": "p"})
        env = json.loads(out)
        assert env["status"] == BrowserErrorKind.permission_denied.value
        assert env["control_state"] == "stopped"

    def test_takeover_surfaced_as_distinct_control_state(self, monkeypatch):
        from cowork.schemas.browser import ControlState

        verdict = BrowserToolVerdict(
            result_code=ResultCode.permission_denied,
            action_type=BrowserActionType.inspect,
            detail="session taken_over",
            control_state=ControlState.taken_over,
        )
        _install_fake_bridge(monkeypatch, verdict=verdict)
        out = _handle({"action": "inspect", "reason": "r", "progress_message": "p"})
        env = json.loads(out)
        assert env["control_state"] == "taken_over"

    def test_no_session_verdict_maps_to_bridge_disconnected(self, monkeypatch):
        # A1: a no-session verdict (ResultCode.error carrying
        # NO_SESSION_DETAIL) surfaces as bridge_disconnected with the detail
        # passed through untouched. Wording is asserted once on the constant
        # (test_browser_broker_control.test_no_session_detail_wording).
        from cowork.services.browser.client import NO_SESSION_DETAIL

        verdict = BrowserToolVerdict(
            result_code=ResultCode.error,
            action_type=BrowserActionType.inspect,
            detail=NO_SESSION_DETAIL,
        )
        _install_fake_bridge(monkeypatch, verdict=verdict)
        out = _handle({"action": "inspect", "reason": "r", "progress_message": "p"})
        env = json.loads(out)
        assert env["status"] == BrowserErrorKind.bridge_disconnected.value
        assert env["error"] == NO_SESSION_DETAIL

    def test_error_without_control_state_omits_field(self, monkeypatch):
        verdict = BrowserToolVerdict(
            result_code=ResultCode.timeout,
            action_type=BrowserActionType.inspect,
            detail="d",
        )
        _install_fake_bridge(monkeypatch, verdict=verdict)
        out = _handle({"action": "inspect", "reason": "r", "progress_message": "p"})
        env = json.loads(out)
        assert "control_state" not in env


# ── result_code → canonical kind mapping (through the handler) ─────────
@pytest.mark.parametrize(
    "code,action,expected",
    [
        (ResultCode.timeout, "inspect", BrowserErrorKind.bridge_disconnected),
        (ResultCode.target_lost, "inspect", BrowserErrorKind.tab_closed),
        (ResultCode.unapproved_tab, "inspect", BrowserErrorKind.permission_denied),
        (ResultCode.permission_denied, "inspect", BrowserErrorKind.permission_denied),
        (ResultCode.error, "follow_link", BrowserErrorKind.navigation_failed),
        (ResultCode.error, "inspect", BrowserErrorKind.bridge_disconnected),
    ],
)
def test_result_code_mapping(monkeypatch, code, action, expected):
    at = (
        BrowserActionType.navigate
        if action == "follow_link"
        else BrowserActionType(action)
    )
    verdict = BrowserToolVerdict(result_code=code, action_type=at, detail="d")
    _install_fake_bridge(monkeypatch, verdict=verdict)
    tc = {"action": action, "reason": "r", "progress_message": "p"}
    if action == "follow_link":
        tc["href"] = "https://example.com/z"
    out = _handle(tc)
    env = json.loads(out)
    assert env["status"] == expected.value
    # raw internal codes are never surfaced
    assert env["status"] not in ("timeout", "target_lost", "unapproved_tab")


# ── stream classifier (WS3-T3) ────────────────────────────────────────
class TestClassifier:
    def test_ok_envelope(self):
        assert classify_browser_status('{"status": "ok"}') == "ok"

    def test_error_envelope(self):
        assert classify_browser_status('{"status": "tab_closed"}') == "error"

    def test_error_prefixed(self):
        assert (
            classify_browser_status('ERROR: {"status": "bridge_disconnected"}')
            == "error"
        )

    def test_unparseable_is_error(self):
        assert classify_browser_status("not json") == "error"

    def test_empty_is_error(self):
        assert classify_browser_status("") == "error"


# ── content-free span (WS5-T3) ────────────────────────────────────────
class TestSpan:
    def test_span_allowed_keys_only(self):
        span = bt.build_browser_span(
            command_type="inspect",
            result_code="ok",
            duration_ms=12,
            domain="https://example.com/some/path?q=1",
            installation_id="inst-1",
            session_id="sess-1",
            task_id="task-1",
            action_id="act-1",
        )
        assert set(span.keys()) <= bt.ALLOWED_SPAN_KEYS
        # host-only domain
        assert span["domain"] == "example.com"
        assert span["funnel"] == bt.BROWSER_FUNNEL
        assert span["installation_id"] == "inst-1"
        assert span["action_id"] == "act-1"

    def test_disallowed_span_key_rejected(self):
        with pytest.raises(bt.DisallowedSpanKeyError):
            bt.assert_content_free_span({"command_type": "inspect", "text": "hi"})

    def test_span_rejects_full_url_domain(self):
        # A full URL smuggled through the allowed `domain` key is rejected by
        # the VALUE guard, not silently emitted.
        with pytest.raises(bt.DisallowedSpanKeyError):
            bt.assert_content_free_span(
                {"command_type": "inspect", "domain": "https://x.com/a?token=1"}
            )
        with pytest.raises(bt.DisallowedSpanKeyError):
            bt.assert_content_free_span(
                {"command_type": "inspect", "domain": "example.com/path"}
            )
        # A bare host passes.
        bt.assert_content_free_span(
            {"command_type": "inspect", "domain": "example.com"}
        )

    def test_span_rejects_unknown_command_type_and_result_code(self):
        with pytest.raises(bt.DisallowedSpanKeyError):
            bt.assert_content_free_span({"command_type": "click"})
        with pytest.raises(bt.DisallowedSpanKeyError):
            bt.assert_content_free_span(
                {"command_type": "inspect", "result_code": "bogus"}
            )

    def test_span_uses_effective_code_for_unobserved_ok(self, monkeypatch):
        # An `ok` verdict with no observed → the emitted span records the
        # DOWNGRADED (non-ok) result code, never a false `ok`.
        captured = []
        token = bt.set_span_sink(lambda et, data: captured.append((et, data)))
        try:
            verdict = BrowserToolVerdict(
                result_code=ResultCode.ok,
                action_type=BrowserActionType.inspect,
                observed=None,
            )
            _install_fake_bridge(monkeypatch, verdict=verdict)
            _handle({"action": "inspect", "reason": "r", "progress_message": "p"})
        finally:
            bt.reset_span_sink(token)
        spans = [p for et, p in captured if et == "response.browser_tool_span"]
        assert len(spans) == 1
        assert spans[0]["result_code"] != "ok"
        assert spans[0]["result_code"] == BrowserErrorKind.bridge_disconnected.value

    def test_emit_through_sink(self):
        captured = []
        token = bt.set_span_sink(lambda et, data: captured.append((et, data)))
        try:
            bt.emit_browser_span(
                bt.build_browser_span(
                    command_type="scroll", result_code="ok", duration_ms=3
                )
            )
        finally:
            bt.reset_span_sink(token)
        assert len(captured) == 1
        et, payload = captured[0]
        assert et == "response.browser_tool_span"
        assert payload["command_type"] == "scroll"

    def test_emit_noop_without_sink(self):
        # No sink installed → no exception, no output.
        bt.emit_browser_span(
            bt.build_browser_span(
                command_type="wait", result_code="ok", duration_ms=1
            )
        )

    def test_handler_emits_span(self, monkeypatch):
        captured = []
        token = bt.set_span_sink(lambda et, data: captured.append((et, data)))
        id_token = bt.set_browser_ids(
            {"installation_id": "inst-9", "task_id": "task-9"}
        )
        try:
            verdict = BrowserToolVerdict(
                result_code=ResultCode.ok,
                action_type=BrowserActionType.inspect,
                observed={"http_status": 200},
                domain="example.com",
                action_id="cmd-9",
            )
            _install_fake_bridge(monkeypatch, verdict=verdict)
            _handle({"action": "inspect", "reason": "r", "progress_message": "p"})
        finally:
            bt.reset_span_sink(token)
            bt.reset_browser_ids(id_token)
        spans = [p for et, p in captured if et == "response.browser_tool_span"]
        assert len(spans) == 1
        span = spans[0]
        assert span["command_type"] == "inspect"
        assert span["result_code"] == "ok"
        assert span["domain"] == "example.com"
        assert span["installation_id"] == "inst-9"
        assert span["task_id"] == "task-9"
        assert span["action_id"] == "cmd-9"
        assert set(span.keys()) <= bt.ALLOWED_SPAN_KEYS

    def test_no_span_on_validation_failure(self, monkeypatch):
        captured = []
        token = bt.set_span_sink(lambda et, data: captured.append((et, data)))
        try:
            _install_fake_bridge(monkeypatch)
            _handle({"action": "click", "reason": "r", "progress_message": "p"})
        finally:
            bt.reset_span_sink(token)
        assert captured == []
