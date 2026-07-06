"""Claude Code coworker — the CLI's own stream-json event shape mapped
onto NormalizedEvent. All subprocess/lifecycle/retry plumbing lives in
BaseCliHarness; this class only knows Claude Code's specific flags and
wire format.
"""
from __future__ import annotations

import json
import subprocess

from cowork.harnesses.base import register
from cowork.harnesses.cli_agents.base import BaseCliHarness
from cowork.harnesses.cli_agents.config import CliConfig
from cowork.harnesses.cli_agents.events import ConversationRequest, NormalizedEvent

CLAUDE_CONFIG = CliConfig(
    executable="claude",
    print_flag="-p",
    model_flag="--model",
    resume_flag="--resume",
    session_flag="--session-id",
    skip_permissions_flag=None,  # uses --permission-mode instead — see build_arguments
    supports_resume=True,
    supports_images=False,  # pass file paths in the prompt instead
    supports_mcp=True,
)


@register
class ClaudeCodeHarness(BaseCliHarness):
    id = "claude-code"
    label = "Claude Code"
    config = CLAUDE_CONFIG

    category = "CLI"
    priority = 5  # fast, no per-token metering against a free API tier
    tags = ("subscription", "fast", "coding", "mcp")

    @classmethod
    def available_models(cls) -> tuple[str, ...]:
        # The aliases the CLI documents for --model (verified against
        # claude 2.1.198: `--model sonnet` / `--model haiku` both work).
        # Aliases track "latest of that family" so they never go stale
        # the way full model ids (claude-fable-5, …) would.
        return ("fable", "opus", "sonnet", "haiku")

    def check_status(self) -> dict:
        base = super().check_status()
        if not base["installed"]:
            return base
        try:
            result = subprocess.run(
                [*self.spawn_argv(base["path"]), "auth", "status"],
                capture_output=True, text=True, timeout=15,
            )
            status = json.loads(result.stdout)
        except Exception as exc:
            base["detail"] = f"Could not read auth status: {exc}"
            return base
        base["loggedIn"] = bool(status.get("loggedIn"))
        base["account"] = status.get("email")
        base["plan"] = status.get("subscriptionType")
        base["detail"] = (
            f"Logged in as {status.get('email')} ({status.get('subscriptionType')})"
            if base["loggedIn"] else "Not logged in — run `claude auth login`."
        )
        return base

    def build_arguments(self, request: ConversationRequest, *, resume: bool) -> list[str]:
        args = super().build_arguments(request, resume=resume)
        # stream-json in -p mode requires --verbose; --permission-mode
        # acceptEdits auto-approves in-project file edits per the user's
        # decision (grill-me, 2026-07-03) rather than hanging on a
        # prompt no terminal exists to answer.
        args += ["--output-format", "stream-json", "--verbose", "--permission-mode", "acceptEdits"]
        return args

    def env_removals(self) -> list[str]:
        # Never let a stray API key hijack billing away from the
        # user's subscription login.
        return ["ANTHROPIC_API_KEY"]

    def parse_line(self, line: str) -> NormalizedEvent | None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None
        etype = event.get("type")

        if etype == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                btype = block.get("type")
                if btype == "text" and block.get("text"):
                    return NormalizedEvent(type="text_chunk", text=block["text"])
                if btype == "tool_use":
                    return NormalizedEvent(
                        type="tool_call",
                        tool_call_id=block.get("id", ""),
                        tool_name=block.get("name", "tool"),
                        tool_args=block.get("input"),
                    )
            return None

        if etype == "user":
            for block in (event.get("message") or {}).get("content") or []:
                if block.get("type") == "tool_result":
                    content = block.get("content")
                    if isinstance(content, list):
                        content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                    return NormalizedEvent(
                        type="tool_result",
                        tool_call_id=block.get("tool_use_id", ""),
                        tool_result=str(content or "")[:65536],
                    )
            return None

        if etype == "result":
            return NormalizedEvent(type="completed", final_text=event.get("result") or "")

        return None
