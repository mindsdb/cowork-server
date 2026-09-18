import pytest

from coding_service_fakes import CREDS, FakeEngine, repository, service_with, wait_for_status
from cowork.coding.contracts import ModeTurnRequest, PermissionMode, SessionCreateRequest, SessionStatus
from cowork.coding.control_errors import StateConflict
from cowork.coding.engines.base import EngineSessionConfig
from cowork.coding.engines.codex_config import prepare_launch
from cowork.coding.questions import QuestionResponse
from cowork.coding.store import CodingStore
from test_coding_questions import PARAMS


class QuestionEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.next_question = False
        self.answers = []
        self.configs = []

    def open_session(self, **kwargs):
        runtime = super().open_session(**kwargs)
        self.configs.append(kwargs['config'])
        original = runtime.events
        def events(turn_id):
            if self.next_question:
                self.next_question = False
                self.answers.append(kwargs['approval_handler']('item/tool/requestUserInput', PARAMS))
            yield from original(turn_id)
        runtime.events = events
        return runtime


def task(tmp_path):
    engine = QuestionEngine()
    service = service_with(tmp_path, engine)
    created = service.create_session(SessionCreateRequest(path=str(repository(tmp_path)), prompt='Plan a timer', task_mode='plan', permission_mode='full_access'), CREDS, 'fake', 'fake-model')
    wait_for_status(service, created.id, SessionStatus.completed)
    return service, engine, service.get_session(created.id)


def test_question_persists_until_answer_then_continues_without_permission_grant(tmp_path):
    service, engine, created = task(tmp_path)
    engine.next_question = True
    service.submit_turn(created.id, 'Ask about layout', CREDS)
    wait_for_status(service, created.id, SessionStatus.awaiting_approval)
    pending = service.get_session(created.id).pending_question
    assert pending
    with pytest.raises(StateConflict, match='question'):
        service.steer(created.id, 'Do something else')
    service.answer_question(created.id, pending.id, QuestionResponse(answers={'layout': ['Compact']}))
    wait_for_status(service, created.id, SessionStatus.completed)
    current = service.get_session(created.id)
    assert current.pending_question is None
    assert current.command_approval_grants == []
    assert current.task_mode == 'plan'
    assert engine.answers == [{'answers': {'layout': {'answers': ['Compact']}}}]
    assert any('Layout: Compact' in event.text for event in service.events(created.id).items)


def test_stop_while_question_pending_and_stale_answer(tmp_path):
    service, engine, created = task(tmp_path)
    engine.next_question = True
    service.submit_turn(created.id, 'Ask', CREDS)
    wait_for_status(service, created.id, SessionStatus.awaiting_approval)
    pending = service.get_session(created.id).pending_question
    service.cancel(created.id)
    wait_for_status(service, created.id, SessionStatus.cancelled)
    assert service.get_session(created.id).pending_question is None
    with pytest.raises(KeyError):
        service.answer_question(created.id, pending.id, QuestionResponse(answers={'layout': ['Compact']}))
    assert engine.answers == [{'answers': {}}]


def test_restart_clears_questions_that_no_longer_have_a_live_reader(tmp_path):
    service, engine, created = task(tmp_path)
    engine.next_question = True
    service.submit_turn(created.id, 'Ask', CREDS)
    wait_for_status(service, created.id, SessionStatus.awaiting_approval)
    service.prepare_shutdown()
    reloaded = CodingStore(service.root).load_session(created.id)
    assert reloaded.pending_question is None
    assert reloaded.status == SessionStatus.interrupted


def test_approve_plan_is_fenced_and_keeps_execution_permissions(tmp_path):
    service, engine, created = task(tmp_path)
    assert engine.configs[0].task_mode == 'plan'
    with pytest.raises(StateConflict, match='changed'):
        service.submit_mode_turn(created.id, ModeTurnRequest(prompt='Build', task_mode='build', expected_event_count=0), CREDS)
    assert service.get_session(created.id).task_mode == 'plan'
    service.submit_mode_turn(created.id, ModeTurnRequest(prompt='Build the plan', task_mode='build', expected_event_count=created.event_count), CREDS)
    wait_for_status(service, created.id, SessionStatus.completed)
    assert service.get_session(created.id).task_mode == 'build'
    assert engine.configs[-1].permission_mode == PermissionMode.full_access
    assert engine.configs[-1].task_mode == 'build'


def test_failed_start_restores_plan_mode(tmp_path, monkeypatch):
    service, _, created = task(tmp_path)
    submit = service._submit_turn
    def fail(*_, **__): raise RuntimeError('cannot start')
    monkeypatch.setattr(service, '_submit_turn', fail)
    with pytest.raises(RuntimeError, match='cannot start'):
        service.submit_mode_turn(created.id, ModeTurnRequest(prompt='Build', task_mode='build', expected_event_count=created.event_count), CREDS)
    restored = service.get_session(created.id)
    assert restored.task_mode == 'plan'
    assert restored.status == SessionStatus.completed
    assert restored.run_status == 'completed'
    monkeypatch.setattr(service, '_submit_turn', submit)
    service.submit_mode_turn(created.id, ModeTurnRequest(prompt='Build', task_mode='build', expected_event_count=restored.event_count), CREDS)
    wait_for_status(service, created.id, SessionStatus.completed)
    assert service.get_session(created.id).task_mode == 'build'


def test_plan_launch_cannot_write_or_escalate_even_with_full_access_default(tmp_path):
    launch = prepare_launch(EngineSessionConfig(model='gpt', permission_mode=PermissionMode.full_access, task_mode='plan', network_access=True), tmp_path, 'http://127.0.0.1:1234')
    assert launch.approval_policy == 'never'
    assert launch.sandbox_policy['type'] == 'readOnly'
    assert launch.thread_params['sandbox'] == 'read-only'


def test_immediate_command_cannot_change_mode_or_reserve_a_run(tmp_path):
    service, _, created = task(tmp_path)
    with pytest.raises(ValueError, match='run this command without changing mode'):
        service.submit_mode_turn(created.id, ModeTurnRequest(
            prompt='/status', task_mode='build', expected_event_count=created.event_count,
        ), CREDS)
    assert service.get_session(created.id) == created


def test_mode_turn_validates_the_target_mode_before_changing_state(tmp_path):
    service, _, created = task(tmp_path)
    service.submit_mode_turn(created.id, ModeTurnRequest(
        prompt='Build', task_mode='build', expected_event_count=created.event_count,
    ), CREDS)
    wait_for_status(service, created.id, SessionStatus.completed)
    built = service.get_session(created.id)
    with pytest.raises(ValueError, match='Finish planning'):
        service.submit_mode_turn(created.id, ModeTurnRequest(
            prompt='/review', task_mode='plan', expected_event_count=built.event_count,
        ), CREDS)
    assert service.get_session(created.id) == built


def test_question_after_turn_ends_does_not_strand_a_reader(tmp_path):
    service, _, created = task(tmp_path)
    with pytest.raises(RuntimeError, match='no longer accepting'):
        service._engine_request(created.id, 'item/tool/requestUserInput', PARAMS)
    assert service.get_session(created.id).pending_question is None


def test_private_answers_never_enter_history(tmp_path):
    service, engine, created = task(tmp_path)
    private = {**PARAMS, 'questions': [{**PARAMS['questions'][0], 'isSecret': True}]}
    import threading
    engine.block_until_release = True
    service.submit_turn(created.id, 'Ask privately', CREDS)
    result = []
    reader = threading.Thread(target=lambda: result.append(service._engine_request(created.id, 'item/tool/requestUserInput', private)))
    reader.start()
    wait_for_status(service, created.id, SessionStatus.awaiting_approval)
    pending = service.get_session(created.id).pending_question
    service.answer_question(created.id, pending.id, QuestionResponse(answers={'layout': ['private-value-123']}))
    reader.join(timeout=1)
    engine.release_events.set()
    wait_for_status(service, created.id, SessionStatus.completed)
    assert result == [{'answers': {'layout': {'answers': ['private-value-123']}}}]
    assert 'private-value-123' not in service.events(created.id).model_dump_json()
    assert '[private answer]' in service.events(created.id).model_dump_json()
