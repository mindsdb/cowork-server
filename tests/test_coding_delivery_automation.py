from pathlib import Path

from cowork.coding.contracts import (
    CodingSession,
    DeliveryAutomationClaimRequest,
    DeliveryAutomationPolicy,
    WorkspaceKind,
)
from cowork.coding.delivery_automation import DeliveryAutomationService
from cowork.coding.store import CodingStore


def session() -> CodingSession:
    return CodingSession(
        id="session-1",
        title="Task",
        engine_id="codex",
        engine_adapter_version="1",
        model="gpt",
        source_path="/source",
        workspace_path="/workspace",
        workspace_kind=WorkspaceKind.git_worktree,
    )


def test_fix_attempts_are_disabled_by_default_and_bounded_when_enabled(tmp_path: Path) -> None:
    store = CodingStore(tmp_path)
    store.save_session(session())
    automation = DeliveryAutomationService(store)
    request = DeliveryAutomationClaimRequest(fingerprint="pr-42:typecheck:sha-1")

    assert store.load_session("session-1").delivery_policy.complete_source_after_merge is False
    assert automation.claim_fix("session-1", request).claimed is False
    automation.update_policy("session-1", DeliveryAutomationPolicy(
        fix_failing_checks=True,
        max_fix_attempts=2,
    ))

    assert automation.claim_fix("session-1", request).attempts == 1
    assert automation.claim_fix("session-1", request).attempts == 2
    exhausted = automation.claim_fix("session-1", request)
    assert exhausted.claimed is False
    assert exhausted.attempts == 2


def test_a_new_failure_fingerprint_gets_its_own_attempt_budget(tmp_path: Path) -> None:
    store = CodingStore(tmp_path)
    item = session()
    item.delivery_policy = DeliveryAutomationPolicy(fix_failing_checks=True, max_fix_attempts=1)
    store.save_session(item)
    automation = DeliveryAutomationService(store)

    first = automation.claim_fix("session-1", DeliveryAutomationClaimRequest(fingerprint="check:sha-1"))
    second = automation.claim_fix("session-1", DeliveryAutomationClaimRequest(fingerprint="check:sha-2"))

    assert first.claimed is True
    assert second.claimed is True
    assert store.load_session("session-1").delivery_automation.fix_attempts == {
        "check:sha-1": 1,
        "check:sha-2": 1,
    }


def test_fix_attempt_history_discards_the_oldest_fingerprint(tmp_path: Path) -> None:
    store = CodingStore(tmp_path)
    item = session()
    item.delivery_policy = DeliveryAutomationPolicy(fix_failing_checks=True, max_fix_attempts=1)
    item.delivery_automation.fix_attempts = {f"old:{index}": 1 for index in range(128)}
    store.save_session(item)
    automation = DeliveryAutomationService(store)

    claim = automation.claim_fix(
        "session-1",
        DeliveryAutomationClaimRequest(fingerprint="new:failure"),
    )

    attempts = store.load_session("session-1").delivery_automation.fix_attempts
    assert claim.claimed is True
    assert len(attempts) == 128
    assert "old:0" not in attempts
    assert attempts["new:failure"] == 1
