from __future__ import annotations

import threading
import time

import pytest

from cowork.coding.approvals import ApprovalBroker
from cowork.coding.contracts import ApprovalDecision


def request_in_thread(broker: ApprovalBroker, method: str, params: dict):
    result: list[dict[str, str]] = []
    thread = threading.Thread(target=lambda: result.append(broker.request("session-1", method, params)))
    thread.start()
    return thread, result


def wait_for_opened(opened: list) -> None:
    deadline = time.monotonic() + 1
    while not opened and time.monotonic() < deadline:
        time.sleep(0.005)
    assert opened


def test_approval_blocks_until_explicit_one_time_decision() -> None:
    opened = []
    closed = []
    broker = ApprovalBroker(lambda session_id, pending: opened.append((session_id, pending)), lambda *args: closed.append(args))
    thread, result = request_in_thread(broker, "item/commandExecution/requestApproval", {"command": "git status"})
    wait_for_opened(opened)
    assert len(opened) == 1 and thread.is_alive()

    pending = opened[0][1]
    broker.resolve("session-1", pending.id, ApprovalDecision.approve_once)
    assert closed[0][2] == ApprovalDecision.approve_once
    with pytest.raises(KeyError):
        broker.resolve("session-1", pending.id, ApprovalDecision.approve_once)
    thread.join(timeout=1)

    assert result == [{"decision": "accept"}]
    assert len(closed) == 1


def test_session_approval_is_only_offered_for_narrow_policy_amendment() -> None:
    opened = []
    broker = ApprovalBroker(lambda _session, pending: opened.append(pending), lambda *_args: None)
    thread, result = request_in_thread(
        broker,
        "item/commandExecution/requestApproval",
        {"command": "npm test", "proposedExecPolicyAmendment": {"prefix": ["npm", "test"]}},
    )
    wait_for_opened(opened)
    pending = opened[0]
    assert pending.allow_session is True
    broker.resolve("session-1", pending.id, ApprovalDecision.approve_session)
    thread.join(timeout=1)
    assert result == [{"decision": "acceptForSession"}]


def test_cancel_and_wrong_session_fail_closed() -> None:
    opened = []
    broker = ApprovalBroker(lambda _session, pending: opened.append(pending), lambda *_args: None)
    thread, result = request_in_thread(broker, "permission/request", {"reason": "network"})
    wait_for_opened(opened)
    pending = opened[0]
    with pytest.raises(KeyError):
        broker.resolve("another-session", pending.id, ApprovalDecision.approve_once)
    broker.cancel_session("session-1")
    thread.join(timeout=1)
    assert result == [{"decision": "decline"}]


def test_approval_details_redact_inline_credentials() -> None:
    pending = ApprovalBroker._describe(
        "item/commandExecution/requestApproval",
        {"command": "curl -H 'Authorization: Bearer top-secret' https://example.invalid?api_key=also-secret"},
    )
    assert "top-secret" not in pending.detail
    assert "also-secret" not in pending.detail
    assert pending.detail.count("[redacted]") == 2


def test_persistence_failure_turns_an_approval_into_a_denial() -> None:
    opened = []

    def fail_close(*_args) -> None:
        raise OSError("disk unavailable")

    broker = ApprovalBroker(lambda _session, pending: opened.append(pending), fail_close)
    thread, result = request_in_thread(broker, "item/commandExecution/requestApproval", {"command": "npm test"})
    wait_for_opened(opened)

    with pytest.raises(OSError, match="disk unavailable"):
        broker.resolve("session-1", opened[0].id, ApprovalDecision.approve_once)

    thread.join(timeout=1)
    assert result == [{"decision": "decline"}]


def test_open_persistence_failure_does_not_leave_a_resolvable_waiter() -> None:
    broker = ApprovalBroker(
        lambda *_args: (_ for _ in ()).throw(OSError("disk unavailable")),
        lambda *_args: None,
    )

    with pytest.raises(OSError, match="disk unavailable"):
        broker.request("session-1", "permission/request", {"reason": "network"})

    assert broker._waiters == {}


def test_concurrent_approval_for_one_task_is_denied_instead_of_hiding_the_visible_request() -> None:
    opened = []
    broker = ApprovalBroker(lambda _session, pending: opened.append(pending), lambda *_args: None)
    first_thread, first_result = request_in_thread(broker, "permission/request", {"reason": "network"})
    wait_for_opened(opened)

    second_result = broker.request("session-1", "permission/request", {"reason": "filesystem"})

    assert second_result == {"decision": "decline"}
    assert len(opened) == 1
    broker.cancel_session("session-1")
    first_thread.join(timeout=1)
    assert first_result == [{"decision": "decline"}]
