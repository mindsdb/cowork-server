from __future__ import annotations

from pathlib import Path

import pytest

from coding_service_fakes import CREDS, FakeEngine, repository, service_with, wait_for_status
from cowork.coding.approvals import ApprovalBroker
from cowork.coding.contracts import ApprovalDecision, PermissionMode, SessionCreateRequest, SessionStatus, SessionUpdateRequest
from cowork.coding.store import CodingStore
from test_coding_command_approvals import METHOD, request


class ApprovalEngine(FakeEngine):
    """Exercise the real turn -> approval -> store -> resume lifecycle."""

    next_request: dict | None = None

    def __init__(self):
        super().__init__()
        self.decisions: list[dict[str, str]] = []

    def open_session(self, **kwargs):
        runtime = super().open_session(**kwargs)
        original = runtime.events

        def events(turn_id):
            params, self.next_request = self.next_request, None
            if params:
                self.decisions.append(kwargs["approval_handler"](METHOD, params))
            yield from original(turn_id)

        runtime.events = events
        return runtime


def make_task(tmp_path: Path):
    repo = repository(tmp_path)
    engine = ApprovalEngine()
    service = service_with(tmp_path, engine)
    task = service.create_session(SessionCreateRequest(path=str(repo), prompt="Approval test"), CREDS, "fake", "fake-model")
    wait_for_status(service, task.id, SessionStatus.completed)
    return service, engine, task


def approve(service, engine, task_id, decision=ApprovalDecision.approve_session):
    engine.next_request = request()
    service.submit_turn(task_id, "Run the command", CREDS)
    wait_for_status(service, task_id, SessionStatus.awaiting_approval)
    pending = service.get_session(task_id).pending_approval
    assert pending and pending.allow_session
    service.resolve_approval(task_id, pending.id, decision)
    wait_for_status(service, task_id, SessionStatus.completed)
    assert service.get_session(task_id).pending_approval is None
    return engine.decisions[-1]


def test_similar_command_completes_without_another_approval_and_survives_reload(tmp_path):
    service, engine, task = make_task(tmp_path)
    assert approve(service, engine, task.id) == {"decision": "accept"}
    assert len(service.get_session(task.id).command_approval_grants) == 1
    engine.next_request = request("npm test -- --reporter=verbose")
    service.submit_turn(task.id, "Run the similar command", CREDS)
    wait_for_status(service, task.id, SessionStatus.completed)
    assert engine.decisions == [{"decision": "accept"}, {"decision": "accept"}]

    reloaded = CodingStore(service.root)
    broker = ApprovalBroker(
        lambda *_: pytest.fail("matching command should not prompt again"), lambda *_: None,
        lambda task_id: reloaded.load_session(task_id).command_approval_grants,
    )
    assert broker.request(task.id, METHOD, request("npm test -- --reporter=verbose")) == {"decision": "accept"}
    assert not (service.root / "codex-home" / "rules").exists()


@pytest.mark.parametrize("decision", [ApprovalDecision.approve_once, ApprovalDecision.deny])
def test_once_and_deny_never_persist_a_rule(tmp_path, decision):
    service, engine, task = make_task(tmp_path)
    assert approve(service, engine, task.id, decision) == {"decision": "accept" if decision == ApprovalDecision.approve_once else "decline"}
    assert service.get_session(task.id).command_approval_grants == []
    approve(service, engine, task.id, ApprovalDecision.deny)


def test_forks_do_not_inherit_grants_even_if_the_command_is_identical(tmp_path):
    service, engine, task = make_task(tmp_path)
    approve(service, engine, task.id)
    child = service.fork_session(task.id, CREDS)
    assert child.command_approval_grants == []
    assert service.get_session(task.id).command_approval_grants
    assert approve(service, engine, child.id) == {"decision": "accept"}


@pytest.mark.parametrize("change", [
    {"permission_mode": PermissionMode.read_only}, {"network_access": True}, {"additional_dirs": []},
])
def test_permission_changes_revoke_grants_but_model_changes_do_not(tmp_path, change):
    service, engine, task = make_task(tmp_path)
    if "additional_dirs" in change:
        folder = tmp_path / "extra"
        folder.mkdir()
        change = {"additional_dirs": [str(folder)]}
    approve(service, engine, task.id)
    updated = service.update_session_config(task.id, SessionUpdateRequest(model="another-model"))
    assert updated.command_approval_grants
    updated = service.update_session_config(task.id, SessionUpdateRequest(**change))
    assert updated.command_approval_grants == []
    assert approve(service, engine, task.id) == {"decision": "accept"}


def test_failed_grant_save_does_not_authorize_or_remember_command(tmp_path, monkeypatch):
    service, engine, task = make_task(tmp_path)
    engine.next_request = request()
    service.submit_turn(task.id, "Run the command", CREDS)
    wait_for_status(service, task.id, SessionStatus.awaiting_approval)
    pending = service.get_session(task.id).pending_approval
    original = service.store.save_session

    def fail_grant_save(session, **kwargs):
        if session.command_approval_grants:
            raise OSError("disk full")
        return original(session, **kwargs)

    monkeypatch.setattr(service.store, "save_session", fail_grant_save)
    with pytest.raises(OSError, match="disk full"):
        service.resolve_approval(task.id, pending.id, ApprovalDecision.approve_session)
    wait_for_status(service, task.id, SessionStatus.completed)
    assert engine.decisions == [{"decision": "decline"}]
    assert service.get_session(task.id).command_approval_grants == []
