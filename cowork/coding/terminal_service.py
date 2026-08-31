from __future__ import annotations

import uuid
from collections.abc import Callable

from cowork.coding.context import safe_engine_error
from cowork.coding.contracts import (
    CodingSession,
    TerminalPage,
    TerminalShellPreference,
    TerminalTab,
    TerminalTabPage,
    TerminalTabState,
)
from cowork.coding.engines.base import EngineCredentials
from cowork.coding.remote_execution import RemoteExecutionCoordinator
from cowork.coding.runtime import RuntimeManager
from cowork.coding.store import CodingStore


class TaskTerminalService:
    """Keep task terminal lifecycle independent from agent turn orchestration."""

    def __init__(
        self,
        store: CodingStore,
        runtimes: RuntimeManager,
        remote: RemoteExecutionCoordinator,
        get_session: Callable[[str], CodingSession],
    ) -> None:
        self.store = store
        self.runtimes = runtimes
        self.remote = remote
        self.get_session = get_session

    def list(self, session_id: str) -> TerminalTabPage:
        session = self.get_session(session_id)
        return TerminalTabPage(items=[self._state(session_id, tab) for tab in session.terminal_tabs])

    def create(
        self,
        session_id: str,
        label: str | None = None,
        *,
        project_action_id: str | None = None,
        project_resource_id: str | None = None,
    ) -> TerminalTabState:
        with self.runtimes.session_lock(session_id):
            session = self.get_session(session_id)
            if len(session.terminal_tabs) >= 12:
                raise RuntimeError("This coding task already has 12 terminals")
            tab = TerminalTab(
                id=str(uuid.uuid4()),
                label=label or self._next_label(session.terminal_tabs),
                project_action_id=project_action_id,
                project_resource_id=project_resource_id,
            )
            self.store.update_session(
                session_id,
                lambda current: current.terminal_tabs.append(tab),
                touch_updated_at=False,
            )
            return self._state(session_id, tab)

    def rename(self, session_id: str, terminal_id: str, label: str) -> TerminalTabState:
        with self.runtimes.session_lock(session_id):
            session = self.get_session(session_id)
            original = self._require_tab(session, terminal_id)
            updated = original.model_copy(update={"label": label})

            def apply(current: CodingSession) -> None:
                for index, item in enumerate(current.terminal_tabs):
                    if item.id == terminal_id:
                        current.terminal_tabs[index] = updated
                        return
                raise KeyError("terminal not found")

            self.store.update_session(session_id, apply, touch_updated_at=False)
            return self._state(session_id, updated)

    def delete(self, session_id: str, terminal_id: str) -> None:
        with self.runtimes.session_lock(session_id):
            session = self.get_session(session_id)
            self._require_tab(session, terminal_id)
            if self.remote.is_remote(session):
                self.remote.operation(session, "terminal_remove", {"terminal_id": terminal_id})
            else:
                self.runtimes.remove_terminal(session_id, terminal_id)
            self.store.update_session(
                session_id,
                lambda current: setattr(
                    current,
                    "terminal_tabs",
                    [item for item in current.terminal_tabs if item.id != terminal_id],
                ),
                touch_updated_at=False,
            )

    def page(self, session_id: str, terminal_id: str, after: int = 0) -> TerminalPage:
        session = self.get_session(session_id)
        self._require_tab(session, terminal_id)
        if self.remote.is_remote(session):
            return self._remote_page(session, terminal_id, after)
        return self.runtimes.terminal_page(session_id, terminal_id, after)

    def wait(
        self,
        session_id: str,
        terminal_id: str,
        after: int,
        timeout: float = 15.0,
    ) -> TerminalPage:
        session = self.get_session(session_id)
        self._require_tab(session, terminal_id)
        if self.remote.is_remote(session):
            return self._remote_page(session, terminal_id, after, wait=timeout)
        return self.runtimes.wait_for_terminal(session_id, terminal_id, after, timeout)

    def start(
        self,
        session_id: str,
        terminal_id: str,
        credentials: EngineCredentials,
        cols: int,
        rows: int,
        shell: TerminalShellPreference = TerminalShellPreference.auto,
    ) -> TerminalPage:
        with self.runtimes.session_lock(session_id):
            session = self.get_session(session_id)
            self._require_tab(session, terminal_id)
            if self.remote.is_remote(session):
                return TerminalPage.model_validate(self.remote.operation(
                    session,
                    "terminal_start",
                    {"terminal_id": terminal_id, "cols": cols, "rows": rows, "shell": shell.value},
                ))
            try:
                return self.runtimes.start_terminal(session, credentials, terminal_id, cols, rows, shell)
            except Exception as exc:
                raise RuntimeError(safe_engine_error(str(exc), credentials)) from exc

    def write(self, session_id: str, terminal_id: str, data_base64: str) -> TerminalPage:
        return self._mutate(session_id, terminal_id, "terminal_input", {"data_base64": data_base64})

    def resize(self, session_id: str, terminal_id: str, cols: int, rows: int) -> TerminalPage:
        return self._mutate(session_id, terminal_id, "terminal_resize", {"cols": cols, "rows": rows})

    def stop(self, session_id: str, terminal_id: str) -> TerminalPage:
        return self._mutate(session_id, terminal_id, "terminal_stop")

    def legacy_page(self, session_id: str, after: int = 0) -> TerminalPage:
        terminal_id = self._first_id(self.get_session(session_id))
        return self.page(session_id, terminal_id, after) if terminal_id else TerminalPage()

    def legacy_wait(self, session_id: str, after: int, timeout: float = 15.0) -> TerminalPage:
        terminal_id = self._first_id(self.get_session(session_id))
        return self.wait(session_id, terminal_id, after, timeout) if terminal_id else TerminalPage()

    def legacy_start(
        self,
        session_id: str,
        credentials: EngineCredentials,
        cols: int,
        rows: int,
        shell: TerminalShellPreference = TerminalShellPreference.auto,
    ) -> TerminalPage:
        session = self.get_session(session_id)
        tab = session.terminal_tabs[0] if session.terminal_tabs else self.create(session_id)
        return self.start(session_id, tab.id, credentials, cols, rows, shell)

    def legacy_write(self, session_id: str, data_base64: str) -> TerminalPage:
        return self.write(session_id, self._required_first_id(session_id), data_base64)

    def legacy_resize(self, session_id: str, cols: int, rows: int) -> TerminalPage:
        return self.resize(session_id, self._required_first_id(session_id), cols, rows)

    def legacy_stop(self, session_id: str) -> TerminalPage:
        return self.stop(session_id, self._required_first_id(session_id))

    def _state(self, session_id: str, tab: TerminalTab) -> TerminalTabState:
        page = self.page(session_id, tab.id)
        return TerminalTabState(
            **tab.model_dump(),
            status=page.status,
            exit_code=page.exit_code,
            error=page.error,
        )

    def _remote_page(
        self,
        session: CodingSession,
        terminal_id: str,
        after: int,
        *,
        wait: float | None = None,
    ) -> TerminalPage:
        payload: dict[str, object] = {"terminal_id": terminal_id, "after": after}
        if wait is not None:
            payload["wait"] = wait
        return TerminalPage.model_validate(self.remote.operation(
            session,
            "terminal_page",
            payload,
            timeout=(wait + 5) if wait is not None else 20,
        ))

    def _mutate(
        self,
        session_id: str,
        terminal_id: str,
        operation: str,
        payload: dict[str, object] | None = None,
    ) -> TerminalPage:
        session = self.get_session(session_id)
        self._require_tab(session, terminal_id)
        if self.remote.is_remote(session):
            return TerminalPage.model_validate(self.remote.operation(
                session,
                operation,
                {"terminal_id": terminal_id, **(payload or {})},
            ))
        if operation == "terminal_input":
            return self.runtimes.write_terminal(session_id, terminal_id, str((payload or {})["data_base64"]))
        if operation == "terminal_resize":
            return self.runtimes.resize_terminal(
                session_id,
                terminal_id,
                int((payload or {})["cols"]),
                int((payload or {})["rows"]),
            )
        return self.runtimes.stop_terminal(session_id, terminal_id)

    def _required_first_id(self, session_id: str) -> str:
        terminal_id = self._first_id(self.get_session(session_id))
        if terminal_id is None:
            raise RuntimeError("There is no terminal for this coding task")
        return terminal_id

    @staticmethod
    def _first_id(session: CodingSession) -> str | None:
        return session.terminal_tabs[0].id if session.terminal_tabs else None

    @staticmethod
    def _require_tab(session: CodingSession, terminal_id: str) -> TerminalTab:
        try:
            return next(item for item in session.terminal_tabs if item.id == terminal_id)
        except StopIteration as exc:
            raise KeyError("terminal not found") from exc

    @staticmethod
    def _next_label(terminals: list[TerminalTab]) -> str:
        existing = {item.label.casefold() for item in terminals}
        number = 1
        while f"terminal {number}".casefold() in existing:
            number += 1
        return f"Terminal {number}"
