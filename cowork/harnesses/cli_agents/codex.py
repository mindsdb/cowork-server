"""OpenAI Codex coworker — headless via `codex exec`.

Verified end-to-end against codex v0.142.5 (npm-installed, ChatGPT-
subscription login) on 2026-07-06: `codex exec [PROMPT]` non-interactive
turns, `codex login status` for auth state, and the --model overrides in
available_models(). On Windows the npm install resolves to a `codex.cmd`
shim — BaseCliHarness.spawn_argv routes it through cmd.exe (a bare
Popen would fail with WinError 193).

supports_resume=False until resume behavior can be verified (codex has
`codex exec resume`; enabling it untested risks the same silent-hang
failure mode Antigravity's resume showed).
"""
from __future__ import annotations

import subprocess

from cowork.harnesses.base import register
from cowork.harnesses.cli_agents.base import BaseCliHarness
from cowork.harnesses.cli_agents.config import CliConfig
from cowork.harnesses.cli_agents.events import ConversationRequest, NormalizedEvent

CODEX_CONFIG = CliConfig(
    executable="codex",
    print_flag="exec",  # subcommand, not a flag — prepended before the prompt just the same
    model_flag="--model",
    resume_flag=None,
    session_flag=None,
    skip_permissions_flag=None,  # handled via --full-auto in build_arguments
    supports_resume=False,
    supports_images=False,
    supports_mcp=True,
)


@register
class CodexHarness(BaseCliHarness):
    id = "codex"
    label = "OpenAI Codex"
    config = CODEX_CONFIG

    category = "CLI"
    priority = 7
    tags = ("subscription", "coding")

    @classmethod
    def available_models(cls) -> tuple[str, ...]:
        # Static catalog — codex has no `models` listing command. On a
        # ChatGPT-subscription login (this project's whole premise) the
        # backend rejects every other name tried against codex v0.142.5
        # ("model is not supported when using Codex with a ChatGPT
        # account" for gpt-5.5-codex / gpt-5.5-codex-mini); only the
        # CLI's own default works. Re-probe after codex CLI updates.
        return ("gpt-5.5",)

    def build_arguments(self, request: ConversationRequest, *, resume: bool) -> list[str]:
        args = super().build_arguments(request, resume=resume)
        # workspace-write sandbox — the codex equivalent of Claude Code's
        # acceptEdits (and safer than
        # --dangerously-bypass-approvals-and-sandbox). exec mode never
        # prompts for approval, so the sandbox flag is the whole policy;
        # its --full-auto shorthand is deprecated as of codex v0.142 and
        # exec rejects the interactive command's --ask-for-approval.
        # --skip-git-repo-check: cowork project folders aren't always
        # git repos, and codex refuses to run in non-repos by default.
        args += ["--sandbox", "workspace-write", "--skip-git-repo-check"]
        return args

    def env_removals(self) -> list[str]:
        # Same rule as Claude Code: never let a stray API key hijack
        # billing away from the ChatGPT-subscription login.
        return ["OPENAI_API_KEY"]

    def parse_line(self, line: str) -> NormalizedEvent | None:
        # codex exec writes human-readable progress + the final answer to
        # stdout (no stable machine format without --json, whose schema
        # can't be verified against a real install yet). Pass everything
        # through as text chunks — the downstream formatter accumulates
        # them into the saved assistant message, same as Antigravity's
        # plain-text path (verified working end-to-end).
        return NormalizedEvent(type="text_chunk", text=line)

    def check_status(self) -> dict:
        base = super().check_status()
        if not base["installed"]:
            base["detail"] = ("Not installed — run `npm i -g @openai/codex`, then `codex login` "
                               "(requires a paid ChatGPT plan).")
            return base
        try:
            result = subprocess.run(
                [*self.spawn_argv(base["path"]), "login", "status"],
                capture_output=True, text=True, timeout=30,
            )
        except Exception as exc:
            base["detail"] = f"Could not check login status: {exc}"
            return base
        out = (result.stdout + result.stderr).strip()
        base["loggedIn"] = result.returncode == 0 and "not logged in" not in out.lower()
        base["detail"] = out[:200] if out else ("Logged in" if base["loggedIn"] else "Not logged in — run `codex login`.")
        return base
