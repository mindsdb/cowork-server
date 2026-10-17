"""get_messages fetched every message's MessageEvents in its own
query (an N+1). _hydrate_message_items batches them into one
`WHERE message_id IN (...)` query, grouped by message_id in Python — this
only regression-tests that grouping preserves each message's own event
order, since the batched query drops the per-message ORDER BY the old
per-message query relied on.

The reads a turn makes at its start (the row count, the gate's tail and the
harness's replay history) are bounded in SQL rather than in Python. The tests
after the batching ones pin that each returns what the unbounded read did.
The last ones pin the read of the earlier answers whose figures a turn's
figures are checked against.
"""
from __future__ import annotations

import asyncio
import json
import logging
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event
from sqlalchemy.dialects import postgresql, sqlite
from sqlmodel import Session, SQLModel, create_engine, select

from cowork.db.scoped import LOCAL_SCOPE, ScopedSession, TenantScope
from cowork.db.units import DatabaseBusy
from cowork.harnesses.anton_harness.scratchpad_cell_replay import (
    SCRATCHPAD_REPLAY_ROLES,
    extract_scratchpad_cells,
)
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.message_event import MessageEvent
from cowork.models.project import Project
from cowork.schemas.responses import Role
from cowork.services.conversations import ConversationService

from _fakes import inline_run_db


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    with Session(engine) as raw:
        yield ScopedSession(raw, LOCAL_SCOPE)


@pytest.fixture
def conversation(tmp_path, session):
    project = Project(name="P", path=str(tmp_path / "p"))
    session.add(project)
    session.commit()
    session.refresh(project)
    conv = Conversation(project_id=project.id, topic="t")
    session.add(conv)
    session.commit()
    session.refresh(conv)
    return conv


def _add_message(session, conversation, *, role, content, seq, events, pending=False, message_id=None):
    message = Message(
        conversation_id=conversation.id, role=role, content=content, seq=seq, pending=pending
    )
    if message_id is not None:
        message.id = message_id
    session.add(message)
    session.commit()
    session.refresh(message)
    for i, event_data in enumerate(events):
        session.add(
            MessageEvent(message_id=message.id, sequence_number=i, event_data=event_data)
        )
    session.commit()
    return message


def test_batched_events_stay_in_order_per_message(session, conversation):
    # Two messages, each with several events, inserted so the events are NOT
    # already grouped/ordered by (message_id, sequence_number) accidentally —
    # message A's events are added out of natural id order relative to B's.
    a = _add_message(
        session, conversation, role=Role.assistant, content="a", seq=0,
        events=[{"step": 0}, {"step": 1}, {"step": 2}],
    )
    b = _add_message(
        session, conversation, role=Role.assistant, content="b", seq=1,
        events=[{"step": 0}, {"step": 1}],
    )

    items = ConversationService(session).get_messages(conversation.id)

    by_id = {item["id"]: item for item in items}
    assert [e["step"] for e in by_id[a.id]["events"]] == [0, 1, 2]
    assert [e["step"] for e in by_id[b.id]["events"]] == [0, 1]


def test_message_with_no_events_gets_empty_list(session, conversation):
    m = _add_message(
        session, conversation, role=Role.user, content="hi", seq=0, events=[]
    )
    items = ConversationService(session).get_messages(conversation.id)
    assert items[0]["id"] == m.id
    assert items[0]["events"] == []


def test_events_batch_correctly_across_the_in_clause_chunk_boundary(session, conversation):
    # A 1,000-message conversation (the ticket's own largest verification
    # size) binds one parameter per visible message in the events IN(...)
    # query if it isn't chunked — past some SQLite builds' bound-parameter
    # ceiling. Insert more messages than one chunk holds and confirm every
    # message's events still resolve correctly, not just the first chunk's.
    from cowork.services.conversations import _EVENTS_IN_CHUNK_SIZE

    count = _EVENTS_IN_CHUNK_SIZE + 5
    messages = [
        _add_message(
            session, conversation, role=Role.assistant, content=f"m{i}", seq=i,
            events=[{"step": i}],
        )
        for i in range(count)
    ]

    items = ConversationService(session).get_messages(conversation.id)

    by_id = {item["id"]: item for item in items}
    assert len(items) == count
    for i, m in enumerate(messages):
        assert by_id[m.id]["events"] == [{"step": i}]


def _scratchpad_event(role: str, content) -> dict:
    return {
        "type": "response.in_progress",
        "thought_role": f"thought.scratchpad.{role}",
        "content": content if isinstance(content, str) else json.dumps(content),
    }


def _exec_end() -> dict:
    return _scratchpad_event("end", {"action": "exec", "code": "print(1)"})


def _result(code: str) -> dict:
    return _scratchpad_event("result", {"code": code, "stdout": "1\n", "stderr": "", "error": None})


_TOOL_USE = [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]
_TOOL_RESULT = [{"type": "tool_result", "tool_use_id": "t1", "content": "r"}]


def test_message_count_matches_the_relationship_including_pending_and_tool_rows(
    engine, session, conversation,
):
    _add_message(session, conversation, role=Role.user, content="q", seq=0, events=[])
    _add_message(session, conversation, role=Role.assistant, content=_TOOL_USE, seq=1, events=[])
    _add_message(session, conversation, role=Role.user, content=_TOOL_RESULT, seq=2, events=[])
    _add_message(
        session, conversation, role=Role.assistant, content="a", seq=3,
        events=[{"type": "response.completed"}],
    )
    _add_message(session, conversation, role=Role.user, content="stranded", seq=4, events=[], pending=True)

    with Session(engine) as raw:
        service = ConversationService(ScopedSession(raw, LOCAL_SCOPE))
        loaded = service.get_conversation(conversation.id)

        assert service.message_count(loaded) == 5
        assert len(loaded.messages) == 5


def test_recent_messages_are_the_tail_of_the_ordered_history(session, conversation):
    seq = 0
    for turn in range(8):
        _add_message(session, conversation, role=Role.user, content=f"q{turn}", seq=seq, events=[])
        seq += 1
        if turn % 3 == 0:
            _add_message(session, conversation, role=Role.assistant, content=_TOOL_USE, seq=seq, events=[])
            _add_message(session, conversation, role=Role.user, content=_TOOL_RESULT, seq=seq + 1, events=[])
            seq += 2
        _add_message(session, conversation, role=Role.assistant, content=f"a{turn}", seq=seq, events=[])
        seq += 1
    _add_message(session, conversation, role=Role.system, content="system", seq=seq, events=[])
    # Rows sharing one seq: the role breaks the tie, and between two answers
    # the id does.
    for role, content in ((Role.assistant, "late answer"), (Role.user, "late question"),
                          (Role.assistant, "second late answer")):
        _add_message(session, conversation, role=role, content=content, seq=seq + 1, events=[])
    _add_message(session, conversation, role=Role.user, content="in flight", seq=seq + 2, events=[], pending=True)
    service = ConversationService(session)
    roles = (Role.user, Role.assistant)
    ordered = [m.id for m in service.get_ordered_messages(conversation.id) if m.role in set(roles)]

    recent = service.get_recent_messages(conversation.id, roles=roles, limit=16)
    everything = service.get_recent_messages(conversation.id, roles=roles, limit=1000)

    assert len(ordered) > 16
    assert [m.id for m in recent] == ordered[-16:]
    assert [m.id for m in everything] == ordered


def test_replay_history_rebuilds_the_same_cells(engine, session, conversation):
    # Ids run opposite to seq, so a read ordered by message id rather than by
    # history would replay the second answer's reset after the first's cell.
    ids = [UUID(int=100 - i) for i in range(5)]
    _add_message(session, conversation, role=Role.user, content="q0", seq=0, events=[], message_id=ids[0])
    _add_message(
        session, conversation, role=Role.assistant, content="a0", seq=1, message_id=ids[1],
        events=[
            _scratchpad_event("start", "{}"),
            _exec_end(),
            _result("first"),
            {"type": "response.output_text.delta", "delta": "hi"},
            {"type": "response.in_progress", "thought_role": "thought.progress", "content": "p"},
        ],
    )
    _add_message(session, conversation, role=Role.user, content="q1", seq=2, events=[], message_id=ids[2])
    _add_message(
        session, conversation, role=Role.assistant, content="a1", seq=3, message_id=ids[3],
        events=[
            _scratchpad_event("end", {"action": "reset"}),
            _scratchpad_event("end", {"action": "view"}),
            _result("not executed"),
            "a scalar string",
            ["a", "list"],
            _exec_end(),
            _result("second"),
            {"type": "response.completed"},
        ],
    )
    # Pending rows are not history; a reset here would clear every cell.
    _add_message(
        session, conversation, role=Role.user, content="in flight", seq=4, pending=True,
        message_id=ids[4], events=[_scratchpad_event("end", {"action": "reset"})],
    )

    with Session(engine) as raw:
        history = ConversationService(ScopedSession(raw, LOCAL_SCOPE)).get_replay_history(
            conversation.id, event_roles=SCRATCHPAD_REPLAY_ROLES,
        )
    with Session(engine) as raw:
        ordered = ConversationService(ScopedSession(raw, LOCAL_SCOPE)).get_ordered_messages(conversation.id)
        every_event = [
            stored.event_data
            for message in ordered
            for stored in raw.exec(
                select(MessageEvent)
                .where(MessageEvent.message_id == message.id)
                .order_by(MessageEvent.sequence_number)
            ).all()
        ]

    assert [m.id for m in history.messages] == [m.id for m in ordered]
    assert all(e["thought_role"] in SCRATCHPAD_REPLAY_ROLES for e in history.events)
    assert extract_scratchpad_cells(history.events) == extract_scratchpad_cells(every_event)
    assert [c.code for c in extract_scratchpad_cells(history.events)] == ["second"]


@pytest.mark.parametrize("late_events", [
    [_scratchpad_event("end", {"action": "reset"})],
    [_exec_end(), _result("late")],
], ids=["reset", "exec"])
def test_replay_history_ignores_answer_committed_after_messages_read(engine, session, conversation, late_events):
    """A channel answer can commit while a UI turn reads its replay history.
    Its events must not enter the cells until its messages enter the history."""
    original_events = [_exec_end(), _result("first")]
    question = _add_message(session, conversation, role=Role.user, content="q", seq=0, events=[])
    answer = _add_message(
        session, conversation, role=Role.assistant, content="a", seq=1, events=original_events,
    )
    conversation_id = conversation.id
    expected_ids = [question.id, answer.id]
    late_ids: list[UUID] = []

    def commit_answer(_conn, _cursor, statement, _parameters, _context, _executemany):
        if late_ids or "FROM message_events" not in statement:
            return
        # The message rows have been materialized; commit a real answer on a
        # separate connection immediately before the event query executes.
        with Session(engine) as writer:
            saved = ConversationService(ScopedSession(writer, LOCAL_SCOPE)).save_assistant_turn(
                conversation_id, "late answer", late_events,
            )
            assert saved is not None
            late_ids.append(saved.id)

    event.listen(engine, "before_cursor_execute", commit_answer)
    try:
        with Session(engine) as reader:
            history = ConversationService(ScopedSession(reader, LOCAL_SCOPE)).get_replay_history(
                conversation_id, event_roles=SCRATCHPAD_REPLAY_ROLES,
            )
    finally:
        event.remove(engine, "before_cursor_execute", commit_answer)

    assert len(late_ids) == 1
    assert [message.id for message in history.messages] == expected_ids
    assert history.events == original_events
    assert [cell.code for cell in extract_scratchpad_cells(history.events)] == ["first"]


def test_replay_history_reads_an_event_with_a_non_finite_float(session, conversation):
    """json.dumps writes NaN bare and SQLite stores it as written. SQLite's
    JSON functions reject such a row before 3.42, so the replay read, which
    parses no JSON in SQL, reads it on any SQLite."""
    _add_message(session, conversation, role=Role.user, content="q", seq=0, events=[])
    _add_message(
        session, conversation, role=Role.assistant, content="a", seq=1,
        events=[{"type": "response.in_progress", "score": float("nan")}, _exec_end(), _result("x")],
    )

    history = ConversationService(session).get_replay_history(
        conversation.id, event_roles=SCRATCHPAD_REPLAY_ROLES,
    )

    assert [c.code for c in extract_scratchpad_cells(history.events)] == ["x"]


def test_replay_history_of_a_history_with_no_rows_issues_no_events_query(engine, session, conversation):
    _add_message(
        session, conversation, role=Role.user, content="in flight", seq=0, pending=True,
        events=[_exec_end(), _result("x")],
    )
    statements: list[str] = []

    def record(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        history = ConversationService(session).get_replay_history(
            conversation.id, event_roles=SCRATCHPAD_REPLAY_ROLES,
        )
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert history.messages == [] and history.events == []
    assert any("FROM messages" in s for s in statements)
    assert not any("message_events" in s for s in statements)


def test_replay_history_reads_events_postgres_cannot_decode(session, conversation):
    """A raw tool result is stored under thought.scratchpad.result as it came,
    NUL bytes included, and a progress line can name a file Python decoded
    with surrogateescape. json.dumps writes those as \\u0000 and \\udcff,
    which a Postgres json column stores as written but its ->> refuses to
    decode anywhere in the document, so one such row would fail every later
    turn. SQLite decodes both, so the next test pins the SQL itself."""
    raw_tool_result = {
        "type": "response.in_progress", "thought_role": "thought.scratchpad.result",
        "content": "col_a\x00col_b",
    }
    undecodable_filename = {
        "type": "response.in_progress", "thought_role": "thought.progress",
        "content": "reading caf\udcff.csv",
    }
    _add_message(session, conversation, role=Role.user, content="q", seq=0, events=[])
    _add_message(
        session, conversation, role=Role.assistant, content="a", seq=1,
        events=[undecodable_filename, _exec_end(), _result("x"), raw_tool_result],
    )

    history = ConversationService(session).get_replay_history(
        conversation.id, event_roles=SCRATCHPAD_REPLAY_ROLES,
    )

    assert history.events == [_exec_end(), _result("x"), raw_tool_result]
    assert [c.code for c in extract_scratchpad_cells(history.events)] == ["x"]


def test_replay_events_filter_parses_no_json_in_sql(session, conversation):
    """No unit lane runs Postgres, so the filter's SQL is checked per dialect.
    It reads each event's stored text: Postgres's ->> raises on a \\u0000 or
    a lone surrogate escape anywhere in an event, and SQLite's JSON functions
    reject a bare NaN before 3.42."""
    stmt = ConversationService(session)._replay_events(conversation.id, event_roles=SCRATCHPAD_REPLAY_ROLES)

    for dialect in (postgresql.dialect(), sqlite.dialect()):
        sql = str(stmt.compile(dialect=dialect))
        assert "->" not in sql and "JSON" not in sql.upper(), sql
        assert "CAST(message_events.event_data AS TEXT) LIKE" in sql, sql


def test_replay_history_keeps_only_events_whose_thought_role_matches(session, conversation):
    """The SQL filter finds a role anywhere in an event's text, so the read
    narrows what it returns to the events whose thought_role is the role."""
    _add_message(session, conversation, role=Role.user, content="q", seq=0, events=[])
    _add_message(
        session, conversation, role=Role.assistant, content="a", seq=1,
        events=[
            {"type": "response.in_progress", "thought_role": "thought.progress",
             "content": "waiting for thought.scratchpad.result"},
            "thought.scratchpad.end",
            _exec_end(),
            {"type": "response.output_text.delta", "delta": "thought.scratchpad.end"},
            _result("x"),
        ],
    )

    history = ConversationService(session).get_replay_history(
        conversation.id, event_roles=SCRATCHPAD_REPLAY_ROLES,
    )

    assert history.events == [_exec_end(), _result("x")]


def _text_delta(text: str, seq: int) -> dict:
    return {"type": "response.output_text.delta", "sequence_number": seq, "item_id": "msg-1", "delta": text,
            "at_ms": 1000 + seq}


def _joined_deltas(events: list[dict]) -> str:
    return "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")


def test_a_saved_turn_stores_each_run_of_text_deltas_as_one_row(session, conversation):
    events = [
        {"type": "response.created", "sequence_number": 0},
        _text_delta("Let", 1), _text_delta(" me", 2), _text_delta(" check.", 3),
        _scratchpad_event("start", "scratchpad"), _exec_end(), _result("x = 1"),
        _text_delta("Done", 7), _text_delta(".", 8),
        {"type": "response.completed", "sequence_number": 9},
    ]
    service = ConversationService(session)
    service.save_user_message(conversation.id, "q")

    saved = service.save_assistant_turn(conversation.id, "Let me check.Done.", events)

    rows = session.exec(
        session.select(MessageEvent).where(MessageEvent.message_id == saved.id).order_by(MessageEvent.sequence_number)
    ).all()
    assert [row.sequence_number for row in rows] == list(range(7))
    stored = [row.event_data for row in rows]
    assert [e.get("delta") for e in stored if e["type"] == "response.output_text.delta"] == ["Let me check.", "Done."]
    (answer,) = [item for item in service.get_messages(conversation.id) if item["role"] == "assistant"]
    page = service.get_messages_page(conversation.id, limit=10).items
    assert answer["events"] == stored == page[-1]["events"]
    assert _joined_deltas(answer["events"]) == _joined_deltas(events)


def test_a_history_mixing_per_delta_and_merged_answers_reads_back_in_order(session, conversation):
    """Answers saved before the merge keep a row per delta, and both kinds
    read back alike."""
    _add_message(session, conversation, role=Role.user, content="q0", seq=0, events=[])
    _add_message(
        session, conversation, role=Role.assistant, content="abc", seq=1,
        events=[_text_delta("a", 1), _text_delta("b", 2), _text_delta("c", 3)],
    )
    service = ConversationService(session)
    service.save_user_message(conversation.id, "q1")
    service.save_assistant_turn(
        conversation.id, "def", [_text_delta("d", 1), _text_delta("e", 2), _text_delta("f", 3), {"type": "response.completed"}],
    )

    for items in (service.get_messages(conversation.id), service.get_messages_page(conversation.id, limit=10).items):
        answers = [item for item in items if item["role"] == "assistant"]
        assert [a["content"] for a in answers] == ["abc", "def"]
        assert [len(a["events"]) for a in answers] == [3, 2]
        assert [_joined_deltas(a["events"]) for a in answers] == ["abc", "def"]


def test_merged_deltas_rebuild_the_same_scratchpad_cells(session, conversation):
    events = [
        _text_delta("a", 1), _exec_end(), _text_delta("b", 3), _text_delta("c", 4), _result("first"),
        _text_delta("d", 6), _scratchpad_event("end", {"action": "reset"}), _text_delta("e", 8),
        _exec_end(), _result("second"), _text_delta("f", 11), _text_delta("g", 12),
    ]
    service = ConversationService(session)
    service.save_user_message(conversation.id, "q")
    service.save_assistant_turn(conversation.id, "abcdefg", events)

    history = service.get_replay_history(conversation.id, event_roles=SCRATCHPAD_REPLAY_ROLES)

    assert extract_scratchpad_cells(history.events) == extract_scratchpad_cells(events)
    assert [c.code for c in extract_scratchpad_cells(history.events)] == ["second"]


# --- the earlier answers a turn checks its figures against --------------------

ORG_A = "11111111-1111-1111-1111-111111111111"
ORG_B = "22222222-2222-2222-2222-222222222222"
USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
# A second member of ORG_A: same tenant, different person.
USER_A2 = "a2a2a2a2-a2a2-a2a2-a2a2-a2a2a2a2a2a2"


def _figure(text: str, origin: str = "step") -> dict:
    return {"start": 0, "end": len(text), "text": text, "origin": origin, "sources": []}


def _figure_sources_event(*texts: str) -> dict:
    """A response.figure_sources event as the formatter saves it."""
    return {
        "type": "response.figure_sources", "sequence_number": 3, "item_id": "msg-1", "project_id": None,
        "steps": [], "figures": [_figure(text) for text in texts],
    }


def _save_turn(service: ConversationService, conversation, *, answer: str, events: list[dict]) -> None:
    """A question and its answer saved as a turn is, with the answer's tool
    rows ahead of it."""
    service.save_user_message(conversation.id, "q")
    service.save_assistant_turn(
        conversation.id, answer, [_text_delta(answer, 1), *events, {"type": "response.completed"}],
        harness="anton",
        tool_rows=[
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "scratchpad", "input": {}},
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "4600"}]},
        ],
    )


def _texts(answers) -> list[list[str]]:
    return [[figure.text for figure in answer.figures] for answer in answers]


def test_earlier_answers_come_from_the_last_answers_only(session, conversation):
    """The third answer has no figures and still takes one of the places; the
    tool rows saved ahead of each answer take none."""
    service = ConversationService(session)
    _save_turn(service, conversation, answer="one", events=[_figure_sources_event("100")])
    _save_turn(service, conversation, answer="two", events=[_figure_sources_event("200")])
    _save_turn(service, conversation, answer="three", events=[])
    _save_turn(service, conversation, answer="four", events=[_figure_sources_event("400")])

    answers = service.get_earlier_answers(conversation.id, answers=3)

    assert _texts(answers) == [["400"], ["200"]]


def test_earlier_answers_skip_other_conversations(session, conversation):
    other = Conversation(project_id=conversation.project_id, topic="other")
    session.add(other)
    session.commit()
    session.refresh(other)
    service = ConversationService(session)
    _save_turn(service, other, answer="theirs", events=[_figure_sources_event("900")])
    _save_turn(service, conversation, answer="mine", events=[_figure_sources_event("100")])

    assert _texts(service.get_earlier_answers(conversation.id, answers=20)) == [["100"]]


def test_earlier_answers_skip_a_pending_row(session, conversation):
    _add_message(
        session, conversation, role=Role.assistant, content="in flight", seq=0, pending=True,
        events=[_figure_sources_event("900")],
    )
    _add_message(
        session, conversation, role=Role.assistant, content="saved", seq=1, events=[_figure_sources_event("100")],
    )

    assert _texts(ConversationService(session).get_earlier_answers(conversation.id, answers=20)) == [["100"]]


def test_a_malformed_figure_sources_event_is_skipped_not_raised(session, conversation, caplog):
    service = ConversationService(session)
    _save_turn(service, conversation, answer="one", events=[_figure_sources_event("100")])
    _save_turn(
        service, conversation, answer="two",
        events=[{"type": "response.figure_sources", "figures": [{"text": 5, "origin": "sideways"}]}],
    )

    with caplog.at_level(logging.WARNING, logger="cowork.services.conversations"):
        answers = service.get_earlier_answers(conversation.id, answers=20)

    assert _texts(answers) == [["100"]]
    assert [r.levelname for r in caplog.records if r.name == "cowork.services.conversations"] == ["WARNING"]


@pytest.mark.parametrize("kind", ["memory", "unknown_kind"], ids=["removed", "newer"])
def test_an_answer_whose_step_kind_this_build_lacks_still_gives_its_figures(session, conversation, caplog, kind):
    """A saved step can carry a kind this build no longer has (memory, which
    earlier builds wrote) or does not have yet (a newer release's, read after
    a rollback). The read uses only the figures, so such a step does not
    skip the answer."""
    service = ConversationService(session)
    event = _figure_sources_event("4,600")
    event["steps"] = [{"tool_use_id": "toolu_1", "kind": kind, "label": "notes.md"}]
    event["figures"][0]["sources"] = [{"tool_use_id": "toolu_1"}]
    _save_turn(service, conversation, answer="4,600", events=[event])

    with caplog.at_level(logging.WARNING, logger="cowork.services.conversations"):
        answers = service.get_earlier_answers(conversation.id, answers=20)

    assert _texts(answers) == [["4,600"]]
    assert [r for r in caplog.records if r.name == "cowork.services.conversations"] == []


def test_a_text_delta_whose_whole_text_is_the_event_type_is_not_read(session, conversation):
    """The SQL filter finds the type, quotes and all, in a text delta that
    holds nothing else, and such a row reads as an event with no figures."""
    _add_message(
        session, conversation, role=Role.assistant, content="response.figure_sources", seq=0,
        events=[{"type": "response.output_text.delta", "delta": "response.figure_sources"}],
    )

    assert ConversationService(session).get_earlier_answers(conversation.id, answers=20) == []


def test_a_text_delta_quoting_the_event_type_is_not_read(session, conversation):
    _add_message(
        session, conversation, role=Role.assistant, content="x", seq=0,
        events=[{"type": "response.output_text.delta", "delta": 'the "response.figure_sources" event'}],
    )

    assert ConversationService(session).get_earlier_answers(conversation.id, answers=20) == []


@pytest.mark.parametrize(
    "stray", ["response.figure_sources", ["response.figure_sources"]], ids=["string", "list"],
)
def test_an_event_that_is_not_an_object_is_skipped_beside_the_ones_that_read(session, conversation, stray):
    """A stored event can be a string or a list, and its text holds the type
    the SQL filter finds. Such a row is skipped, so it can't fail the read
    and lose the conversation's other answers."""
    _add_message(
        session, conversation, role=Role.assistant, content="one", seq=0, events=[_figure_sources_event("100")],
    )
    _add_message(
        session, conversation, role=Role.assistant, content="two", seq=1,
        events=[stray, _figure_sources_event("200")],
    )

    answers = ConversationService(session).get_earlier_answers(conversation.id, answers=20)

    assert _texts(answers) == [["200"], ["100"]]


def test_earlier_answers_filter_parses_no_json_in_sql(session, conversation):
    """The type is found in each event's stored text, for the reason
    _replay_events gives, with the _ in it escaped so it matches only
    itself."""
    stmt = ConversationService(session)._earlier_answer_events(conversation.id, answers=20)

    for dialect in (postgresql.dialect(), sqlite.dialect()):
        compiled = stmt.compile(dialect=dialect)
        sql = str(compiled)
        assert "->" not in sql and "JSON" not in sql.upper(), sql
        assert "CAST(message_events.event_data AS TEXT) LIKE" in sql, sql
        assert "ESCAPE '/'" in sql, sql
        assert '"response.figure/_sources"' in compiled.params.values(), compiled.params


async def test_the_read_covers_the_last_20_answers(session, conversation, monkeypatch):
    import cowork.services.conversations as conversations_mod

    service = ConversationService(session)
    for number in range(21):
        _save_turn(service, conversation, answer=f"answer {number}", events=[_figure_sources_event(str(number))])
    monkeypatch.setattr(conversations_mod, "run_db", inline_run_db(session))

    answers = await conversations_mod.read_earlier_answers(conversation_id=conversation.id, scope=LOCAL_SCOPE)

    assert conversations_mod.MAX_EARLIER_ANSWERS == 20
    assert _texts(answers) == [[str(number)] for number in range(20, 0, -1)]


@pytest.mark.parametrize("error", [DatabaseBusy("no connection"), ValueError("Conversation not found")])
async def test_read_earlier_answers_gives_none_and_warns_when_the_read_fails(monkeypatch, caplog, error):
    import cowork.services.conversations as conversations_mod

    async def failing(fn, *, scope):
        raise error

    monkeypatch.setattr(conversations_mod, "run_db", failing)

    with caplog.at_level(logging.WARNING, logger="cowork.services.conversations"):
        answers = await conversations_mod.read_earlier_answers(conversation_id=uuid4(), scope=LOCAL_SCOPE)

    assert answers == ()
    assert [r.levelname for r in caplog.records if r.name == "cowork.services.conversations"] == ["WARNING"]


async def test_a_cancel_during_the_read_reaches_the_turn(monkeypatch):
    """A failed read finds no earlier answers, but a cancel is not a failure:
    it reaches the turn that awaits the read, so a stopped turn stops."""
    import cowork.services.conversations as conversations_mod

    async def cancelled(fn, *, scope):
        raise asyncio.CancelledError

    monkeypatch.setattr(conversations_mod, "run_db", cancelled)

    with pytest.raises(asyncio.CancelledError):
        await conversations_mod.read_earlier_answers(conversation_id=uuid4(), scope=LOCAL_SCOPE)


@pytest.mark.parametrize(
    ("org_id", "owner"), [(ORG_A, USER_A2), (ORG_B, USER_A)], ids=["another-member", "another-org"],
)
async def test_a_conversation_the_reader_does_not_own_gives_no_earlier_answers(engine, monkeypatch, org_id, owner):
    """In org mode a conversation is its owner's: another member of the same
    org, or anyone in another org, reads none of its answers."""
    import cowork.services.conversations as conversations_mod

    with Session(engine) as raw:
        writer = ScopedSession(raw, LOCAL_SCOPE)
        project = Project(name="P", path="/tmp/p", org_id=org_id)
        writer.add(project)
        writer.commit()
        writer.refresh(project)
        theirs = Conversation(project_id=project.id, topic="theirs", org_id=org_id, created_by=owner)
        writer.add(theirs)
        writer.commit()
        writer.refresh(theirs)
        _add_message(writer, theirs, role=Role.assistant, content="a", seq=0, events=[_figure_sources_event("100")])
        conversation_id = theirs.id

    reader_scope = TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A)
    with Session(engine) as raw:
        reader = ScopedSession(raw, reader_scope)
        with pytest.raises(ValueError):
            ConversationService(reader).get_earlier_answers(conversation_id, answers=20)
        monkeypatch.setattr(conversations_mod, "run_db", inline_run_db(reader))

        answers = await conversations_mod.read_earlier_answers(conversation_id=conversation_id, scope=reader_scope)

    assert answers == ()


async def test_the_owner_reads_their_own_conversations_earlier_answers(engine, monkeypatch):
    """The same org-mode read gives the owner their answers, so the refusals
    above are the owner rule's and not a read that never works."""
    import cowork.services.conversations as conversations_mod

    reader_scope = TenantScope(org_mode=True, org_id=ORG_A, user_id=USER_A)
    with Session(engine) as raw:
        writer = ScopedSession(raw, LOCAL_SCOPE)
        project = Project(name="P", path="/tmp/p", org_id=ORG_A)
        writer.add(project)
        writer.commit()
        writer.refresh(project)
        mine = Conversation(project_id=project.id, topic="mine", org_id=ORG_A, created_by=USER_A)
        writer.add(mine)
        writer.commit()
        writer.refresh(mine)
        _add_message(writer, mine, role=Role.assistant, content="a", seq=0, events=[_figure_sources_event("100")])
        conversation_id = mine.id

    with Session(engine) as raw:
        reader = ScopedSession(raw, reader_scope)
        monkeypatch.setattr(conversations_mod, "run_db", inline_run_db(reader))

        answers = await conversations_mod.read_earlier_answers(conversation_id=conversation_id, scope=reader_scope)

    assert _texts(answers) == [["100"]]
