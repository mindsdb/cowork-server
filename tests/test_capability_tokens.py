"""Capability tokens (cowork.capability_tokens) and the per-session
``inference`` token each Code mode engine session gets from RuntimeManager."""
from __future__ import annotations

import pytest

from cowork import capability_tokens
from cowork.coding.contracts import SessionCreateRequest, SessionStatus
from tests.support.coding.service_fakes import CREDS, FakeEngine, repository, service_with, wait_for_status


@pytest.fixture(autouse=True)
def empty_store(monkeypatch):
    monkeypatch.setattr(capability_tokens, "_records", {})


class TestStore:
    def test_a_minted_token_verifies_for_its_purpose_and_owner(self):
        token = capability_tokens.mint(purpose="inference", session_id="session-1")
        record = capability_tokens.verify(token=token, purpose="inference")
        assert record == capability_tokens.CapabilityRecord(purpose="inference", session_id="session-1")

    def test_each_mint_is_a_new_token(self):
        first = capability_tokens.mint(purpose="inference", session_id="session-1")
        second = capability_tokens.mint(purpose="inference", session_id="session-1")
        assert first != second
        assert len(first) >= 32

    @pytest.mark.parametrize("token", [None, "", "not-a-token"])
    def test_an_unknown_token_verifies_as_nothing(self, token):
        capability_tokens.mint(purpose="inference", session_id="session-1")
        assert capability_tokens.verify(token=token, purpose="inference") is None

    def test_a_token_of_another_purpose_is_refused(self):
        # A purpose a later change adds; the Literal does not name it yet.
        token = capability_tokens.mint(purpose="hooks", session_id="session-1")  # type: ignore[arg-type]
        assert capability_tokens.verify(token=token, purpose="inference") is None

    def test_a_revoked_token_is_refused(self):
        token = capability_tokens.mint(purpose="inference", session_id="session-1")
        assert capability_tokens.revoke(token=token) is True
        assert capability_tokens.verify(token=token, purpose="inference") is None
        assert capability_tokens.revoke(token=token) is False

    def test_revoking_a_session_revokes_only_its_tokens(self):
        first = capability_tokens.mint(purpose="inference", session_id="session-1")
        second = capability_tokens.mint(purpose="inference", session_id="session-1")
        other = capability_tokens.mint(purpose="inference", session_id="session-2")

        assert capability_tokens.revoke_session(session_id="session-1") == 2
        assert capability_tokens.verify(token=first, purpose="inference") is None
        assert capability_tokens.verify(token=second, purpose="inference") is None
        assert capability_tokens.verify(token=other, purpose="inference") is not None
        assert capability_tokens.revoke_session(session_id="session-1") == 0

    def test_the_store_holds_no_raw_token(self):
        token = capability_tokens.mint(purpose="inference", session_id="session-1")
        stored = repr(capability_tokens._records)
        assert token not in stored
        assert token.encode() not in capability_tokens._records
        assert all(len(digest) == 32 for digest in capability_tokens._records)


class _FailingEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.fail_open = False

    def open_session(self, **kwargs):
        runtime = super().open_session(**kwargs)
        if self.fail_open:
            raise RuntimeError("app-server failed to start")
        return runtime


def _completed_task(tmp_path, engine):
    service = service_with(tmp_path, engine)
    created = service.create_session(
        SessionCreateRequest(path=str(repository(tmp_path)), prompt="Fix the timer"),
        CREDS, "fake", "fake-model",
    )
    wait_for_status(service, created.id, SessionStatus.completed)
    return service, created


def _live(token: str) -> capability_tokens.CapabilityRecord | None:
    return capability_tokens.verify(token=token, purpose="inference")


class TestEngineSessions:
    def test_each_engine_session_gets_its_own_inference_token(self, tmp_path):
        engine = FakeEngine()
        service, created = _completed_task(tmp_path, engine)

        token = engine.configs[-1].inference_api_key
        assert _live(token) == capability_tokens.CapabilityRecord(purpose="inference", session_id=created.id)
        # The user's MindsHub key never stands in for it.
        assert token != CREDS.minds_api_key

    def test_closing_the_session_revokes_its_token(self, tmp_path):
        engine = FakeEngine()
        service, created = _completed_task(tmp_path, engine)
        token = engine.configs[-1].inference_api_key

        service.runtimes.close_session(created.id)

        assert _live(token) is None

    def test_a_reopened_session_gets_a_new_token_and_the_old_one_dies(self, tmp_path):
        engine = FakeEngine()
        service, created = _completed_task(tmp_path, engine)
        first = engine.configs[-1].inference_api_key

        service.runtimes.close_session(created.id)
        service.runtimes.open(service.get_session(created.id), CREDS)
        second = engine.configs[-1].inference_api_key

        assert second != first
        assert _live(first) is None
        assert _live(second) is not None

    def test_a_runtime_that_closed_itself_loses_its_token_when_discarded(self, tmp_path):
        engine = FakeEngine()
        service, created = _completed_task(tmp_path, engine)
        token = engine.configs[-1].inference_api_key
        runtime = service.runtimes.open(service.get_session(created.id), CREDS)

        runtime.close()
        service.runtimes.discard_if_closed(created.id, runtime)

        assert _live(token) is None

    def test_a_failed_open_revokes_the_token_it_minted(self, tmp_path):
        engine = _FailingEngine()
        service, created = _completed_task(tmp_path, engine)
        service.runtimes.close_session(created.id)
        engine.fail_open = True

        with pytest.raises(RuntimeError, match="app-server failed to start"):
            service.runtimes.open(service.get_session(created.id), CREDS)

        assert _live(engine.configs[-1].inference_api_key) is None
        assert capability_tokens._records == {}

    def test_shutting_down_revokes_every_token(self, tmp_path):
        engine = FakeEngine()
        service, created = _completed_task(tmp_path, engine)
        token = engine.configs[-1].inference_api_key

        service.runtimes.close_all()

        assert _live(token) is None
