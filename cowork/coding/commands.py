from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cowork.coding.context import goal_directive, goal_status_text, slash_command
from cowork.coding.contracts import CodingEvent, CodingSession, EventType
from cowork.coding.engines.base import EngineCredentials, EngineSession
from cowork.coding.engines.registry import CodingEngineRegistry
from cowork.coding.runtime import RuntimeManager

EmitEvent = Callable[[str, CodingEvent], CodingEvent]

_ATTACHMENTLESS_COMMANDS = {"compact", "status", "goal", "review"}
_IMMEDIATE_GOAL_ACTIONS = {"view", "edit", "pause", "clear"}


@dataclass(frozen=True)
class CommandIntent:
    """Canonical interpretation of one prompt's optional slash command."""

    name: str
    argument: str
    goal_action: str = ""
    goal_objective: str | None = None

    @classmethod
    def parse(cls, prompt: str) -> CommandIntent:
        name, argument = slash_command(prompt)
        action, objective = goal_directive(argument) if name == "goal" else ("", None)
        return cls(name=name, argument=argument, goal_action=action, goal_objective=objective)

    @property
    def runs_immediately(self) -> bool:
        return self.name in {"compact", "status"} or self.goal_action in _IMMEDIATE_GOAL_ACTIONS

    @property
    def resumes_goal(self) -> bool:
        return self.goal_action == "resume"

    @property
    def is_review(self) -> bool:
        return self.name == "review"

    def engine_prompt(self, display_prompt: str) -> str:
        if self.name != "init":
            return display_prompt
        return (
            "Create or improve AGENTS.md for this repository. Inspect the project first, preserve useful "
            "existing instructions, and document concise commands and conventions for future coding agents."
        )

    def runtime_payload(self, display_prompt: str) -> dict[str, str | None]:
        """Serialize the parsed intent once for an agent-neutral remote worker."""

        return {
            "prompt": display_prompt,
            "engine_prompt": self.engine_prompt(display_prompt),
            "command": self.name,
            "goal_action": self.goal_action,
            "goal_objective": self.goal_objective,
        }

    def validate_attachments(self, has_attachments: bool) -> None:
        if has_attachments and self.name in _ATTACHMENTLESS_COMMANDS:
            raise ValueError(f"/{self.name} does not accept file attachments")

    def validate_arguments(self) -> None:
        if self.argument and self.name in {"compact", "status", "review", "init"}:
            raise ValueError(f"/{self.name} does not accept an argument")

    def validate_while_running(self) -> None:
        if self.name == "compact":
            raise RuntimeError("Queue /compact for after the active turn finishes")
        if self.name in {"review", "init"}:
            raise RuntimeError(f"Queue /{self.name} for after the active turn finishes")
        if self.goal_action in {"set", "resume"}:
            raise RuntimeError("Queue this goal command for after the active turn finishes")


class CodingCommandHandler:
    """Execute task-local slash commands that do not start a model turn."""

    def __init__(
        self,
        registry: CodingEngineRegistry,
        runtimes: RuntimeManager,
        get_session: Callable[[str], CodingSession],
        emit: EmitEvent,
        state_lock: Any,
        is_running: Callable[[str], bool],
    ) -> None:
        self._registry = registry
        self._runtimes = runtimes
        self._get_session = get_session
        self._emit = emit
        self._state_lock = state_lock
        self._is_running = is_running

    def validate(self, session: CodingSession, command: str) -> None:
        if not command:
            return
        capabilities = self._registry.get(session.engine_id).capabilities()
        supported = {item.name: item for item in capabilities.commands}
        registered = supported.get(command)
        if registered is None:
            raise ValueError(
                f"/{command} is not supported by {capabilities.label}. Type / to see available commands"
            )
        if registered.action == "client":
            raise ValueError(f"/{command} opens a Code workspace control and cannot be sent to the agent")

    def run_immediate(
        self,
        session_id: str,
        intent: CommandIntent,
        display_prompt: str,
        credentials: EngineCredentials,
    ) -> CodingSession:
        with self._runtimes.session_lock(session_id):
            with self._state_lock:
                session = self._get_session(session_id)
                if intent.name == "compact" and self._is_running(session_id):
                    raise RuntimeError("Wait for the active turn to finish before compacting this task")
                self._emit(
                    session_id,
                    CodingEvent(
                        type=EventType.user_message,
                        title="You",
                        text=display_prompt,
                        phase="completed",
                    ),
                )
            runtime = self._runtimes.open_locked(session, credentials)
            if intent.name == "compact":
                runtime.compact()
                self._emit(
                    session_id,
                    CodingEvent(
                        type=EventType.session,
                        title="Compaction started",
                        text="Codex is compacting the task context. Future turns will continue from the compacted history.",
                        phase="completed",
                        data={"command": "compact"},
                    ),
                )
            elif intent.name == "goal":
                self.emit_goal_result(
                    session_id,
                    runtime,
                    intent.goal_action,
                    intent.goal_objective,
                )
            else:
                self.emit_status(session, runtime)
            return self._get_session(session_id)

    def emit_goal_result(
        self,
        session_id: str,
        runtime: EngineSession,
        action: str,
        objective: str | None,
    ) -> None:
        goal = runtime.goal_status() if action == "view" else runtime.update_goal(action, objective)
        labels = {
            "view": "Goal status",
            "edit": "Goal updated",
            "pause": "Goal paused",
            "clear": "Goal cleared",
        }
        if goal is None:
            text = "No goal is active for this coding task." if action == "view" else "The task goal has been cleared."
        else:
            text = goal_status_text(goal)
        self._emit(
            session_id,
            CodingEvent(
                type=EventType.session,
                title=labels[action],
                text=text,
                phase="completed",
                data={"command": "goal", "goal": goal or {}},
            ),
        )

    def emit_status(self, session: CodingSession, runtime: EngineSession) -> None:
        goal = runtime.goal_status()
        goal_line = goal_status_text(goal) if goal else "Goal: none"
        text = "\n".join(
            (
                f"Status: {session.status.value}",
                f"Model: {session.model}",
                f"Reasoning: {session.reasoning_effort or 'model default'}",
                f"Fast: {'on' if session.service_tier == 'priority' else 'off'}",
                f"Permissions: {session.permission_mode.value}",
                f"Network: {'on' if session.network_access else 'off'} · Web search: {'on' if session.web_search else 'off'}",
                f"Folder: {session.workspace_path}",
                f"Additional folders: {len(session.additional_dirs)}",
                f"Queued instructions: {len(session.queued_instructions)}",
                f"Terminal: {self._runtimes.terminal_status(session.id)}",
                goal_line,
            )
        )
        self._emit(
            session.id,
            CodingEvent(
                type=EventType.session,
                title="Task status",
                text=text,
                phase="completed",
                data={"command": "status", "goal": goal or {}},
            ),
        )
