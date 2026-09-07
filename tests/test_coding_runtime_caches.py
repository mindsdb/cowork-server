from __future__ import annotations

import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

from cowork.coding.contracts import PermissionMode
from cowork.coding.control_models import CodeTask, RunStatus, RuntimeCommand, TaskRun
from cowork.coding.engines.registry import CodingEngineRegistry
from cowork.coding.project_models import CodeProject, RepositoryResource
from cowork.coding.runtime import RuntimeManager
from cowork.coding.runtime_operations import (
    _MAX_COMPLETED_RESULTS,
    RuntimeWorkspaceOperations,
)
from cowork.coding.runtime_protocol import RuntimeExecutionConfig, RuntimeLease


def lease() -> RuntimeLease:
    run = TaskRun(
        id="run-caches",
        task_id="task-caches",
        computer_id="remote-computer",
        status=RunStatus.running,
        lease_id="lease-caches",
    )
    return RuntimeLease(
        task=CodeTask(id="task-caches", title="Cache task", prompt="Build"),
        run=run,
        lease_id="lease-caches",
        agent_token="agent-token-that-is-long-enough-for-runtime",
        project=CodeProject(
            id="caches-project",
            name="Caches project",
            resources=[RepositoryResource(id="repo", name="Repo", source_url="https://example.invalid/repo.git")],
        ),
        execution=RuntimeExecutionConfig(
            engine_id="fake",
            model="fake-model",
            permission_mode=PermissionMode.workspace,
        ),
    )


def operation(index: int) -> RuntimeCommand:
    return RuntimeCommand(
        id=f"operation-{index}",
        run_id="run-caches",
        epoch=1,
        kind="operation",
        payload={"operation": "git_states"},
    )


def test_completed_operation_results_evict_the_oldest_entries() -> None:
    calls: list[int] = []
    manager = SimpleNamespace(git_states=lambda workspaces: calls.append(len(workspaces)) or [])
    operations = RuntimeWorkspaceOperations(lease(), manager, SimpleNamespace(workspaces=[]), object())
    first = operation(0)

    assert operations.execute(first) == operations.execute(first) == ({"items": []}, None)
    assert len(calls) == 1
    for index in range(1, _MAX_COMPLETED_RESULTS + 1):
        operations.execute(operation(index))
    assert len(calls) == _MAX_COMPLETED_RESULTS + 1

    operations.execute(operation(_MAX_COMPLETED_RESULTS))
    assert len(calls) == _MAX_COMPLETED_RESULTS + 1
    operations.execute(first)
    assert len(calls) == _MAX_COMPLETED_RESULTS + 2


def test_session_locks_are_evicted_once_no_caller_holds_them(tmp_path: Path) -> None:
    manager = RuntimeManager(tmp_path, CodingEngineRegistry(), lambda *_args: {})
    lock = manager.session_lock("session-1")
    with lock:
        assert manager.session_lock("session-1") is lock
    other = manager.session_lock("session-2")
    reference = weakref.ref(lock)

    del lock
    gc.collect()

    assert reference() is None
    assert manager.session_lock("session-2") is other
