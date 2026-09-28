"""The one sanctioned way for cowork-server to construct anton's ChatSession.

`anton.core.session.ChatSession` registers the `scratchpad` tool
unconditionally and, with no `runtime_factory` override, binds it to
`local_scratchpad_runtime_factory`: a `LocalScratchpadRuntime` that spawns a
subprocess of THIS process and pipes LLM-written Python into it, with `cwd`
set to the workspace's project directory. In org mode that directory sits on
the shared EFS tree, whose files are written by every organization's agent, so
any ChatSession built inside cowork-server is arbitrary code execution in the
process that holds every organization's data and the deployment's credentials.
`noexec` on the mount does not help: it blocks `./script`, not
`python script.py`.

The guard lives here, on construction, rather than on each caller, because two
independent constructors already existed (the anton harness's turn path and the
credential probe reached from POST /connectors/submissions/), each of which
would have needed, and one of which did not get, its own check. Construction is
also the point a static test can see: `tests/test_no_subprocess_static.py`
treats a `ChatSession(...)` call anywhere under `cowork/` other than the one
below as a new execution site and fails, so a third caller cannot be added
without either routing through this function or being reviewed as an exception.
Guarding `turn_stream` instead would be invisible to that test, since it is a
method call on whatever the caller named its session object.

Both callers also end their session here: `close_session_scratchpads` stops
the scratchpad processes a session started, which nothing else does.
"""

from __future__ import annotations

import asyncio
import functools
import logging

from cowork.common.settings.app_settings import get_app_settings

logger = logging.getLogger(__name__)

NO_IN_PROCESS_TURN_DETAIL = (
    "Agent turns do not run inside this deployment's server process; "
    "they are dispatched to a worker."
)


def in_process_agent_allowed() -> bool:
    """False on the multi-tenant (org) deployment, where the workspace an
    agent would run against is shared-EFS storage written by every
    organization."""
    return get_app_settings().tenancy_mode != "org"


def build_chat_session(config):
    """Construct an anton ChatSession, or refuse in org mode.

    `config` is an `anton.core.session.ChatSessionConfig`. It is not annotated
    because anton is imported lazily below: importing `anton.core.session` at
    cowork import time pulls the whole agent runtime into every process that
    touches this module, including ones that only ever refuse.
    """
    if not in_process_agent_allowed():
        raise RuntimeError(NO_IN_PROCESS_TURN_DETAIL)
    from anton.core.session import ChatSession

    return ChatSession(config)


# Strong references for the detached closes below: asyncio holds only a weak
# reference to a task once nothing else does, so a dropped `create_task` result
# can be garbage-collected mid-flight. Discarded on completion.
_scratchpad_closes: set[asyncio.Task[None]] = set()


def close_session_scratchpads(session, *, owner: str) -> None:
    """Kill the scratchpad processes a finished ChatSession started.

    Every pad is a child process that exits only when told to. Dropping the
    session does not stop it, because asyncio's child watcher keeps the pad's
    transport referenced, so without this call each turn that ran Python
    leaves a process behind until the server exits.

    Pads only. `ChatSession.close()` also reaps the full-stack backends the
    session launched, and those must keep serving after the turn. anton has no
    public pads-only close, so this calls `session._scratchpads.close_all()`,
    as anton's own CLI does on Ctrl-C. `close_all()` keeps each pad's
    namespace snapshot on disk, and the next turn's fresh pad restores it.

    Scheduled, never awaited, so a caller's `finally` can use it without an
    `await`. A second cancel while the caller unwinds (Stop pressed twice, or
    Stop then shutdown) would cut an awaited close off after its first pad;
    the task runs to completion either way. It never raises, so the caller's
    later cleanup steps still run. `drain_scratchpad_closes` waits for any
    still running at shutdown.
    """
    manager = getattr(session, "_scratchpads", None)
    if manager is None:
        return
    try:
        task = asyncio.get_running_loop().create_task(manager.close_all())
    except Exception:
        logger.exception("Could not schedule closing the scratchpads of %s", owner)
        return
    _scratchpad_closes.add(task)
    task.add_done_callback(functools.partial(_log_scratchpad_close, owner=owner))


def _log_scratchpad_close(task: asyncio.Task[None], *, owner: str) -> None:
    _scratchpad_closes.discard(task)
    if task.cancelled():
        logger.warning("Closing the scratchpads of %s was cancelled; some may still be running", owner)
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("Closing the scratchpads of %s failed", owner, exc_info=exc)


async def drain_scratchpad_closes(*, timeout: float = 5.0) -> None:
    """Wait for scheduled scratchpad closes, for shutdown.

    The turns shutdown cancels schedule their closes as they unwind, and a
    task still pending when the event loop stops is destroyed with its pads
    alive. `asyncio.wait` does not cancel on timeout, so a slow close keeps
    running for as long as the loop does.
    """
    tasks = list(_scratchpad_closes)
    if not tasks:
        return
    _done, pending = await asyncio.wait(tasks, timeout=timeout)
    if pending:
        logger.warning("%d scratchpad close(s) did not finish within %.1fs", len(pending), timeout)
