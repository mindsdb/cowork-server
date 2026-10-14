"""Plan text uses the bounded stream buffer without swallowing plan state."""

from coding_service_fakes import CREDS, FakeEngine, FakeSession, repository, service_with, wait_for_status
from cowork.coding.contracts import CodingEvent, EventType, SessionCreateRequest, SessionStatus
from cowork.coding.engines.codex_events import map_codex_notification
from cowork.coding.turns import EventBuffer


def delta(text, item='plan-1', turn='turn-1'):
    return map_codex_notification('item/plan/delta', {
        'delta': text, 'itemId': item, 'turnId': turn,
    })


def test_plan_deltas_are_coalesced_with_a_size_bound(monkeypatch):
    monkeypatch.setattr('cowork.coding.turns.time.monotonic', lambda: 0.0)
    emitted = []
    buffer = EventBuffer(emitted.append)
    for _ in range(1000):
        buffer.add(delta('next step '))
    buffer.flush()
    assert len(emitted) == 3
    assert ''.join(event.text for event in emitted) == 'next step ' * 1000
    assert all(len(event.text) <= 4000 for event in emitted)


def test_plan_deltas_flush_at_the_existing_time_bound(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('cowork.coding.turns.time.monotonic', lambda: clock[0])
    emitted = []
    buffer = EventBuffer(emitted.append)
    buffer.add(delta('first'))
    assert emitted == []
    clock[0] = 0.16
    buffer.add(delta('second'))
    assert [event.text for event in emitted] == ['first']
    buffer.flush()
    assert [event.text for event in emitted] == ['first', 'second']


def test_plan_updates_and_completion_stay_ordered_and_separate(monkeypatch):
    monkeypatch.setattr('cowork.coding.turns.time.monotonic', lambda: 0.0)
    emitted = []
    buffer = EventBuffer(emitted.append)
    update = map_codex_notification('turn/plan/updated', {
        'turnId': 'turn-1', 'plan': [{'step': 'Build', 'status': 'inProgress'}],
    })
    completed = map_codex_notification('item/completed', {
        'turnId': 'turn-1', 'item': {'id': 'plan-1', 'type': 'plan', 'text': 'whole plan'},
    })
    buffer.add(delta('first'))
    buffer.add(update)
    buffer.add(delta('second'))
    buffer.add(completed)
    assert len(emitted) == 4
    assert emitted[1] == update
    assert emitted[3] == completed
    assert [event.text for event in emitted] == ['first', '', 'second', '']
    assert completed.data['text'] == 'whole plan'
    buffer.flush()
    assert len(emitted) == 4


def test_plan_buffer_does_not_merge_different_items_turns_or_event_types(monkeypatch):
    monkeypatch.setattr('cowork.coding.turns.time.monotonic', lambda: 0.0)
    emitted = []
    buffer = EventBuffer(emitted.append)
    for event in (
        delta('one'), delta('two', item='plan-2'), delta('three', turn='turn-2'),
        CodingEvent(type=EventType.agent_message, text='response', item_id='plan-1', turn_id='turn-2'),
    ):
        buffer.add(event)
    buffer.flush()
    assert [event.text for event in emitted] == ['one', 'two', 'three', 'response']


def test_streamed_plan_is_coalesced_before_session_persistence(tmp_path, monkeypatch):
    def events(_session, turn_id):
        for _ in range(1000):
            yield delta('next step ', turn=turn_id)
        yield CodingEvent(type=EventType.session, data={'status': 'completed'})

    monkeypatch.setattr(FakeSession, 'events', events)
    service = service_with(tmp_path, FakeEngine())
    created = service.create_session(SessionCreateRequest(
        path=str(repository(tmp_path)), prompt='Plan a timer', task_mode='plan',
    ), CREDS, 'fake', 'fake-model')
    wait_for_status(service, created.id, SessionStatus.completed)
    persisted = [event for event in service.events(created.id).items if event.type == EventType.plan]
    assert 3 <= len(persisted) < 1000
    assert ''.join(event.text for event in persisted) == 'next step ' * 1000
    assert all(len(event.text) <= 4000 for event in persisted)
