import threading

import pytest
from pydantic import ValidationError

from cowork.coding.questions import PendingQuestion, QuestionBroker, QuestionResponse


PARAMS = {"questions": [{"id": "layout", "header": "Layout", "question": "Which layout?", "isOther": True,
    "options": [{"label": "Compact", "description": "Keep it simple"}, {"label": "Spacious", "description": "More room"}]}]}


def start(broker, opened):
    result = []
    thread = threading.Thread(target=lambda: result.append(broker.request("task", PARAMS)))
    thread.start()
    assert opened.wait(1)
    return thread, result


def test_answer_is_explicit_validated_and_not_an_approval():
    pending, opened = [], threading.Event()
    broker = QuestionBroker(lambda _, item: (pending.append(item), opened.set()), lambda *_: None)
    thread, result = start(broker, opened)
    assert thread.is_alive()
    with pytest.raises(ValueError, match="each question"):
        broker.resolve("task", pending[0].id, QuestionResponse(answers={}))
    with pytest.raises(KeyError):
        broker.resolve("wrong-task", pending[0].id, QuestionResponse(answers={"layout": ["Compact"]}))
    broker.resolve("task", pending[0].id, QuestionResponse(answers={"layout": ["My own layout"]}))
    thread.join(1)
    assert result == [{"answers": {"layout": {"answers": ["My own layout"]}}}]
    with pytest.raises(KeyError):
        broker.resolve("task", pending[0].id, QuestionResponse(answers={"layout": ["Compact"]}))


@pytest.mark.parametrize("timeout", [0.01, 60])
def test_timeout_and_cancel_release_the_reader_without_an_answer(timeout):
    opened, closed = threading.Event(), []
    broker = QuestionBroker(lambda *_: opened.set(), lambda *args: closed.append(args), timeout=timeout)
    thread, result = start(broker, opened)
    if timeout == 60:
        broker.cancel_session("task")
    thread.join(1)
    assert not thread.is_alive()
    assert result == [{"answers": {}}]
    assert len(closed) == 1


def test_failed_persistence_does_not_deliver_answer():
    opened, pending = threading.Event(), []
    def fail(*_):
        raise OSError("store unavailable")
    broker = QuestionBroker(lambda _, item: (pending.append(item), opened.set()), fail)
    thread, result = start(broker, opened)
    with pytest.raises(OSError):
        broker.resolve("task", pending[0].id, QuestionResponse(answers={"layout": ["Compact"]}))
    thread.join(1)
    assert result == [{"answers": {}}]


def test_duplicate_questions_and_empty_answers_are_rejected():
    with pytest.raises(ValidationError):
        PendingQuestion(questions=PARAMS["questions"] * 2)
    with pytest.raises(ValidationError):
        QuestionResponse(answers={"layout": ["  "]})
    pending = PendingQuestion(questions=[{**PARAMS["questions"][0], "isOther": False}])
    with pytest.raises(ValueError, match="offered"):
        QuestionResponse(answers={"layout": ["Unknown"]}).validate_for(pending)


def test_concurrent_request_cannot_replace_pending_question():
    opened = threading.Event()
    broker = QuestionBroker(lambda *_: opened.set(), lambda *_: None)
    thread, result = start(broker, opened)
    assert broker.request("task", PARAMS) == {"answers": {}}
    broker.cancel_session("task")
    thread.join(1)
    assert result == [{"answers": {}}]


def test_secret_question_accepts_private_text_instead_of_an_offered_label():
    pending = PendingQuestion(questions=[{**PARAMS["questions"][0], "isOther": False, "isSecret": True}])
    QuestionResponse(answers={"layout": ["private response"]}).validate_for(pending)
