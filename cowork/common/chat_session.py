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
"""

from __future__ import annotations

from cowork.common.settings.app_settings import get_app_settings

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
