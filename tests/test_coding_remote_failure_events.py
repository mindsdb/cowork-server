from __future__ import annotations

from cowork.coding.control_models import RunStatus, RuntimeEvent, TaskRun
from cowork.coding.remote_execution import RemoteExecutionCoordinator


def _failed_run() -> TaskRun:
    return TaskRun(
        id="run-1",
        task_id="task-1",
        computer_id="remote-computer",
        status=RunStatus.failed,
        lease_id="lease-1",
        last_error="This model needs credits. Add credits or choose another model.",
    )


def _error_event(payload: dict[str, object]) -> RuntimeEvent:
    return RuntimeEvent(run_id="run-1", computer_id="remote-computer", lease_id="lease-1", epoch=1, seq=3, kind="error", payload=payload)


def test_a_classified_remote_failure_projects_its_code_detail_and_model() -> None:
    _, event = RemoteExecutionCoordinator._coding_event(_failed_run(), _error_event({
        "message": "This model needs credits. Add credits or choose another model.",
        "detail": "unexpected status 402 Payment Required: wallet empty Authorization: Bearer secret-token",
        "code": "insufficient_credits",
        "model": "gpt",
    }))

    assert event.type.value == "error"
    assert event.title == "Task failed"
    assert event.text == "This model needs credits. Add credits or choose another model."
    assert event.data["code"] == "insufficient_credits"
    assert event.data["model"] == "gpt"
    assert event.data["detail"] == "unexpected status 402 Payment Required: wallet empty Authorization: Bearer [redacted]"
    assert event.data["runStatus"] == "failed"


def test_an_unclassified_remote_failure_keeps_the_legacy_detail_shape() -> None:
    _, event = RemoteExecutionCoordinator._coding_event(_failed_run(), _error_event({"detail": "adapter stream disconnected"}))

    assert event.text == "adapter stream disconnected"
    assert "code" not in event.data and "model" not in event.data
    assert event.data["detail"] == "adapter stream disconnected"


def test_a_remote_budget_stop_keeps_its_scope_for_the_app() -> None:
    _, event = RemoteExecutionCoordinator._coding_event(_failed_run(), _error_event({
        "message": "This workspace's spending budget is used up.",
        "code": "budget_exceeded",
        "budget_scope": "workspace",
    }))

    assert event.data["code"] == "budget_exceeded"
    assert event.data["budget_scope"] == "workspace"


def test_a_remote_failure_drops_a_malformed_or_misplaced_scope() -> None:
    _, bad = RemoteExecutionCoordinator._coding_event(_failed_run(), _error_event({
        "code": "budget_exceeded", "budget_scope": "Bad Scope",
    }))
    _, other = RemoteExecutionCoordinator._coding_event(_failed_run(), _error_event({
        "code": "insufficient_credits", "budget_scope": "workspace",
    }))

    assert "budget_scope" not in bad.data
    assert "budget_scope" not in other.data


def test_a_remote_budget_stop_keeps_its_period_for_the_app() -> None:
    _, event = RemoteExecutionCoordinator._coding_event(_failed_run(), _error_event({
        "code": "budget_exceeded", "budget_scope": "team", "budget_period": "month",
    }))
    _, bad = RemoteExecutionCoordinator._coding_event(_failed_run(), _error_event({
        "code": "budget_exceeded", "budget_period": "week",
    }))

    assert event.data["budget_period"] == "month"
    assert "budget_period" not in bad.data


def test_a_remote_budget_stop_keeps_its_model_for_the_app() -> None:
    _, event = RemoteExecutionCoordinator._coding_event(_failed_run(), _error_event({
        "code": "budget_exceeded", "budget_scope": "team", "budget_model": "haiku-4-5",
    }))
    _, bad = RemoteExecutionCoordinator._coding_event(_failed_run(), _error_event({
        "code": "budget_exceeded", "budget_model": "bad model",
    }))
    _, other = RemoteExecutionCoordinator._coding_event(_failed_run(), _error_event({
        "code": "insufficient_credits", "budget_model": "haiku-4-5",
    }))

    assert event.data["budget_model"] == "haiku-4-5"
    assert "budget_model" not in bad.data
    assert "budget_model" not in other.data
