"""Headless Anton probe — runs as a server-side worker, not a chat turn.

Spins up a fresh Anton ChatSession with empty history, no persistence,
and a minimal toolbelt of 7 tools. Runs one turn with a credential-test
prompt and yields (kind, payload) tuples back to the caller.

The final yield is always ('verdict', ProbeOutcome) — even on timeout or
crash — so the caller can always rely on a terminal event.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from cowork.common.paths import cowork_home

logger = logging.getLogger(__name__)

_PROBE_TMP_DIR = cowork_home() / "tmp"


@dataclass
class ProbeOutcome:
    status: str = "unresolved"   # success | failure | needs_input | unresolved
    summary: str = ""
    error: str = ""
    extra_fields: list[dict] = field(default_factory=list)
    follow_up: str = ""
    method_id: str | None = None


class CredentialProbe:
    """Orchestrates one headless Anton turn to test a set of credentials."""

    def __init__(
        self,
        *,
        engine: str,
        credentials: dict,
        llm_client,
        workspace,
        form_spec: dict | None = None,
        skipped: list[str] | None = None,
        timeout_seconds: float = 90.0,
    ) -> None:
        self.engine = engine
        self.credentials = credentials
        self.llm_client = llm_client
        self.workspace = workspace
        self.form_spec = form_spec or {}
        self.skipped = list(skipped or [])
        self.timeout_seconds = timeout_seconds
        self._outcome = ProbeOutcome()
        self._pending: list[tuple[str, Any]] = []

    async def _set_status(self, _session, tc_input):
        text = (tc_input.get("text") or "").strip()
        if text:
            self._pending.append(("status", text))
        return "ok"

    async def _set_field_status(self, _session, tc_input):
        name = (tc_input.get("name") or "").strip()
        if not name:
            return "ignored: missing field name"
        status = tc_input.get("status")
        if isinstance(status, str):
            status = status.strip()
        method_id = (tc_input.get("method_id") or "").strip() or None
        self._pending.append(("field_status", {"name": name, "status": status, "method_id": method_id}))
        return "ok"

    async def _remove_field(self, _session, tc_input):
        name = (tc_input.get("name") or "").strip()
        if not name:
            return "ignored: missing field name"
        method_id = (tc_input.get("method_id") or "").strip() or None
        self._pending.append(("remove_field", {"name": name, "method_id": method_id}))
        return "ok"

    async def _switch_method(self, _session, tc_input):
        method_id = (tc_input.get("method_id") or "").strip()
        if not method_id:
            return "ignored: missing method_id"
        reason = (tc_input.get("reason") or "").strip()
        self._pending.append(("switch_method", {"method_id": method_id, "reason": reason}))
        return "ok"

    async def _report_success(self, _session, tc_input):
        self._outcome.status = "success"
        self._outcome.summary = (tc_input.get("summary") or "").strip()
        return "ok"

    async def _report_failure(self, _session, tc_input):
        self._outcome.status = "failure"
        self._outcome.error = (tc_input.get("error") or "").strip() or "Connection failed."
        self._outcome.follow_up = (tc_input.get("follow_up") or "").strip()
        return "ok"

    async def _request_extra_field(self, _session, tc_input):
        self._outcome.status = "needs_input"
        fields = tc_input.get("fields") or []
        if isinstance(fields, list):
            self._outcome.extra_fields = [f for f in fields if isinstance(f, dict) and f.get("name")]
        self._outcome.follow_up = (tc_input.get("reason") or "").strip()
        self._outcome.method_id = (tc_input.get("method_id") or "").strip() or None
        return "ok"
    
    def _summarize_field_list(self, fields: list, filled_names: set[str], skipped_set: set[str], indent: str = "  ") -> str:
        if not fields:
            return f"{indent}(no fields)"
        lines: list[str] = []
        for f in fields:
            if not isinstance(f, dict):
                continue
            name = f.get("name") or ""
            if not name:
                continue
            ftype = f.get("type") or "text"
            label = f.get("label") or name
            if name in skipped_set:
                state = "skipped"
            elif name in filled_names:
                state = "filled"
            else:
                state = "empty"
            lines.append(f"{indent}• `{name}` ({ftype}, {state}) — {label}")
        return "\n".join(lines) if lines else f"{indent}(no fields)"


    def _summarize_form(self, form_spec: dict, filled_names: set[str], skipped: list[str]) -> str:
        skipped_set = set(skipped or [])
        methods = (form_spec or {}).get("methods") or []
        if methods:
            selected = (form_spec or {}).get("selected_method")
            lines: list[str] = []
            lines.append("This is a MULTI-METHOD form. The user picks ONE method "
                        "before submitting. Each method has its own field list.")
            if selected:
                lines.append(f"Currently selected method: `{selected}`")
            else:
                lines.append("No method selected yet — the user is on the picker.")
            lines.append("")
            for m in methods:
                if not isinstance(m, dict):
                    continue
                mid = m.get("id") or ""
                mlabel = m.get("label") or mid
                recommended = " (recommended)" if m.get("recommended") else ""
                sel_marker = " ← selected" if mid == selected else ""
                lines.append(f"Method `{mid}` — {mlabel}{recommended}{sel_marker}")
                if m.get("description"):
                    lines.append(f"  {m['description']}")
                lines.append(self._summarize_field_list(
                    m.get("fields") or [], filled_names, skipped_set, indent="    ",
                ))
                lines.append("")
            return "\n".join(lines).rstrip()

        fields = (form_spec or {}).get("fields") or []
        return self._summarize_field_list(fields, filled_names, skipped_set)

    def _write_credentials_env(self) -> tuple[str, list[str]]:
        var_names: list[str] = []
        lines: list[str] = []
        for key, value in (self.credentials or {}).items():
            if not key:
                continue
            var = f"DS_{str(key).upper()}"
            var_names.append(var)
            escaped = (
                str(value)
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
            )
            lines.append(f'{var}="{escaped}"')

        _PROBE_TMP_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"cowork-vault-{uuid.uuid4().hex[:16]}.env"
        path = _PROBE_TMP_DIR / filename
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path), var_names

    def _build_prompt(self, env_path: str, var_names: list[str]) -> str:
        filled_names = {v.replace("DS_", "", 1).lower() for v in var_names}
        field_names_lower = set()
        for f in (self.form_spec or {}).get("fields") or []:
            if isinstance(f, dict) and f.get("name"):
                field_names_lower.add(str(f["name"]).lower())
        filled_names = {n for n in filled_names if n in field_names_lower}
        filled_original = {
            f["name"] for f in ((self.form_spec or {}).get("fields") or [])
            if isinstance(f, dict) and f.get("name")
            and str(f["name"]).lower() in filled_names
        }
        roster = self._summarize_form(self.form_spec, filled_original, self.skipped)
        selected_method = (self.form_spec or {}).get("selected_method") or (self.form_spec or {}).get("auth_method")
        method_hint = (
            f"\nThe user picked method `{selected_method}` — focus your "
            f"probe on whatever auth flow that implies (e.g. app_password "
            f"→ IMAP/SMTP; service_account → impersonation; oauth_paste → "
            f"refresh-token exchange). If you decide a different method "
            f"would clearly work better, you can call `switch_method` with "
            f"a one-line reason.\n"
        ) if selected_method else (
            "\nNo method has been picked. Probe what the user submitted; "
            "if there's no usable info, request_extra_field with the "
            "minimum needed.\n"
        )
        return (
            f"You are a connection prober for `{self.engine}`. Your only job is "
            f"to determine if the credentials we just collected actually "
            f"work, and report back via your tools.\n\n"
            f"The user-submitted credentials are in a temporary `.env` file:\n"
            f"  Path: `{env_path}`\n"
            f"  Variable names: {', '.join(var_names) or '(none)'}\n"
            f"{method_hint}\n"
            f"——— CURRENT FORM ROSTER ———\n"
            f"These are the fields ALREADY in the form. Do NOT call "
            f"`request_extra_field` for any of these — they're already "
            f"there (even if empty or skipped). Use exact names when you "
            f"reference them via `set_field_status` / `remove_field`. For "
            f"multi-method forms, ALL field-edit tools (set_field_status, "
            f"remove_field, request_extra_field) take a `method_id` "
            f"parameter — pass the method whose fields you're touching.\n\n"
            f"{roster}\n\n"
            f"——— STEPS (follow in order) ———\n"
            f"1. Call `set_status` with a short message like \"Loading credentials…\".\n"
            f"2. In the scratchpad, parse the .env file (e.g. `dotenv_values('{env_path}')`). "
            f"NEVER print the values. NEVER echo them back in any tool input.\n"
            f"3. Call `set_status` with \"Installing <pkg>…\" if you need a client library, "
            f"then install it via the scratchpad's `packages` array.\n"
            f"4. Call `set_status` with \"Probing {self.engine}…\" and run a tiny test query "
            f"(e.g. `SELECT 1` for a database, `/me` for an API, list-buckets for storage).\n"
            f"5. **TRY HARD before reporting failure.** A successful auth that "
            f"can't access the resource the user actually needs IS NOT a success — but "
            f"it's also not a reason to immediately give up. Before calling "
            f"`report_failure` or `request_extra_field`:\n"
            f"   a) Try the engine's discovery endpoints first.\n"
            f"   b) Try multiple fallbacks. A 401 means broken auth; a 403 / 404 / "
            f"\"scope\" error means the credential works but is narrowly scoped.\n"
            f"   c) Probe the actual resource the user cares about.\n"
            f"   d) Only after exhausting (a)–(c) without finding a working path, "
            f"call `request_extra_field` for the missing piece.\n"
            f"6. Call EXACTLY ONE of:\n"
            f"   • `report_success(summary=...)` — connection works AND a real data "
            f"endpoint returned data.\n"
            f"   • `report_failure(error=..., follow_up=...)` — definitively broken.\n"
            f"   • `request_extra_field(fields=[...])` — credentials insufficient "
            f"after exhausting step (5).\n\n"
            f"——— RULES ———\n"
            f"• Keep prose to one sentence at most.\n"
            f"• NEVER print credential values. NEVER include them in tool inputs.\n"
            f"• You MUST call exactly one verdict tool before stopping.\n"
            f"• Don't ask follow-up questions in prose — use `request_extra_field`.\n"
        )

    # ── Main entry point ─────────────────────────────────────────────

    async def run(self) -> AsyncIterator[tuple[str, Any]]:
        from anton.core.session import ChatSession, ChatSessionConfig, SystemPromptContext
        from anton.core.llm.provider import (
            StreamTextDelta, StreamToolResult,
            StreamToolUseStart, StreamToolUseEnd, StreamToolUseDelta, StreamComplete,
        )
        from anton.core.tools.tool_defs import ToolDef

        env_path, var_names = self._write_credentials_env()

        SET_STATUS_TOOL = ToolDef(
            name="set_status",
            description=(
                "Update the form's live status line. Call before every "
                "scratchpad step so the user sees the probe progressing. "
                "Use 3-6 word phrases (e.g. 'Loading credentials', "
                "'Installing posthog', 'Probing /api/me'). Never include "
                "credential values."
            ),
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            handler=self._set_status,
        )

        SET_FIELD_STATUS_TOOL = ToolDef(
            name="set_field_status",
            description=(
                "Update the small status line under a SPECIFIC field in "
                "the form. Pass `status=null` to clear. For multi-method "
                "forms, include `method_id`."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "status": {"type": ["string", "null"]},
                    "method_id": {"type": "string"},
                },
                "required": ["name"],
            },
            handler=self._set_field_status,
        )

        REMOVE_FIELD_TOOL = ToolDef(
            name="remove_field",
            description="Permanently delete a field from the form.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "method_id": {"type": "string"},
                },
                "required": ["name"],
            },
            handler=self._remove_field,
        )

        SWITCH_METHOD_TOOL = ToolDef(
            name="switch_method",
            description="Flip the form to a different method (multi-method forms only).",
            input_schema={
                "type": "object",
                "properties": {
                    "method_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["method_id"],
            },
            handler=self._switch_method,
        )

        REPORT_SUCCESS_TOOL = ToolDef(
            name="report_success",
            description="Verdict: the connection works. Call AT MOST ONCE.",
            input_schema={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
            handler=self._report_success,
        )

        REPORT_FAILURE_TOOL = ToolDef(
            name="report_failure",
            description="Verdict: the connection does not work. Call AT MOST ONCE.",
            input_schema={
                "type": "object",
                "properties": {
                    "error": {"type": "string"},
                    "follow_up": {"type": "string"},
                },
                "required": ["error"],
            },
            handler=self._report_failure,
        )

        REQUEST_EXTRA_FIELD_TOOL = ToolDef(
            name="request_extra_field",
            description="Verdict: need more fields from the user before we can connect.",
            input_schema={
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "label": {"type": "string"},
                                "type": {"type": "string"},
                                "help": {"type": "string"},
                                "placeholder": {"type": "string"},
                                "required": {"type": "boolean"},
                            },
                            "required": ["name"],
                        },
                    },
                    "reason": {"type": "string"},
                    "method_id": {"type": "string"},
                },
                "required": ["fields"],
            },
            handler=self._request_extra_field,
        )

        config = ChatSessionConfig(
            llm_client=self.llm_client,
            system_prompt_context=SystemPromptContext(
                runtime_context="",
                suffix=(
                    "You are a connection prober. You are NOT in a user-facing "
                    "chat — your job is to call your tools to verify a "
                    "credential set, then exit. Don't narrate. Don't ask "
                    "the user questions in prose."
                ),
            ),
            workspace=self.workspace,
            tools=[
                SET_STATUS_TOOL,
                SET_FIELD_STATUS_TOOL,
                REMOVE_FIELD_TOOL,
                SWITCH_METHOD_TOOL,
                REPORT_SUCCESS_TOOL,
                REPORT_FAILURE_TOOL,
                REQUEST_EXTRA_FIELD_TOOL,
            ],
        )

        try:
            probe_session = ChatSession(config)
        except Exception as exc:
            logger.exception("Could not build probe session")
            self._outcome.status = "failure"
            self._outcome.error = f"Could not start probe: {exc}"
            try:
                os.unlink(env_path)
            except Exception:
                pass
            yield ("verdict", self._outcome)
            return

        prompt = self._build_prompt(env_path, var_names)
        current_tool_name: dict[str, str] = {}
        current_tool_input_json: dict[str, str] = {}

        try:
            async def _drive():
                async for event in probe_session.turn_stream(prompt):
                    while self._pending:
                        yield self._pending.pop(0)

                    if isinstance(event, StreamTextDelta):
                        if event.text:
                            yield ("text", event.text)
                    elif isinstance(event, StreamToolUseStart):
                        current_tool_name[event.id] = event.name
                        current_tool_input_json[event.id] = ""
                        if event.name == "scratchpad":
                            yield ("scratchpad", {"action": "start"})
                    elif isinstance(event, StreamToolUseDelta):
                        current_tool_input_json[event.id] = (
                            current_tool_input_json.get(event.id, "") + (event.json_delta or "")
                        )
                    elif isinstance(event, StreamToolUseEnd):
                        name = current_tool_name.pop(event.id, "")
                        raw = current_tool_input_json.pop(event.id, "")
                        if name == "scratchpad":
                            import json as _json
                            try:
                                parsed = _json.loads(raw or "{}")
                            except Exception:
                                parsed = {}
                            if (parsed.get("action") or "") == "exec":
                                yield ("scratchpad", {
                                    "action": "end",
                                    "name": parsed.get("name", ""),
                                    "code": parsed.get("code", ""),
                                    "one_line_description": parsed.get("one_line_description", ""),
                                })
                    elif isinstance(event, StreamToolResult):
                        if event.name == "scratchpad":
                            yield ("scratchpad", {
                                "action": "result",
                                "content": event.content or "",
                            })
                    elif isinstance(event, StreamComplete):
                        pass

                while self._pending:
                    yield self._pending.pop(0)

            gen = _drive().__aiter__()
            deadline = asyncio.get_event_loop().time() + self.timeout_seconds
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    self._outcome.status = "failure"
                    self._outcome.error = f"Probe timed out after {int(self.timeout_seconds)}s."
                    self._outcome.follow_up = "Try again, or check that the service is reachable."
                    break
                try:
                    evt = await asyncio.wait_for(gen.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    self._outcome.status = "failure"
                    self._outcome.error = f"Probe timed out after {int(self.timeout_seconds)}s."
                    self._outcome.follow_up = "Try again, or check that the service is reachable."
                    break
                yield evt

        except Exception as exc:
            logger.exception("Probe session crashed")
            self._outcome.status = "failure"
            self._outcome.error = f"Probe crashed: {exc}"
        finally:
            try:
                os.unlink(env_path)
            except Exception:
                logger.debug("Could not delete probe env file %s", env_path, exc_info=True)

        if self._outcome.status == "unresolved":
            self._outcome.status = "failure"
            self._outcome.error = "Probe ended without a verdict."
            self._outcome.follow_up = "Try resubmitting; if it persists, check that the engine name is correct."

        yield ("verdict", self._outcome)
