"""Tool calls survive into conversation history (ENG-742).

A turn's tool_use / tool_result blocks are persisted as their own `messages`
rows (ordered by `seq`), hidden from the UI, and replayed verbatim into the
next turn's LLM history so the agent remembers what it did.
"""
import pytest
from sqlmodel import Session, SQLModel, create_engine

from cowork.harnesses.anton_harness.harness import (
    _sanitize_tool_result,
    _split_turn_into_rows,
)
from cowork.db.scoped import LOCAL_SCOPE, ScopedSession
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.project import Project
from cowork.services.conversations import ConversationService, _is_tool_row


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    # ConversationService expects a ScopedSession (get_messages/delete_turn use
    # its .select). LOCAL_SCOPE adds no org filter, matching desktop/local mode.
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


def _turn_slice():
    """A realistic anton history slice: narration + tool call, its result,
    then the final answer."""
    return [
        {"role": "assistant", "content": [
            {"type": "text", "text": "let me look it up"},
            {"type": "tool_use", "id": "t1", "name": "recall_skill", "input": {"name": "pdf"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "SKILL BODY"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "final answer"}]},
    ]


# --- pure helpers -----------------------------------------------------------

def test_split_strips_text_and_keeps_tool_blocks():
    rows = _split_turn_into_rows(_turn_slice())
    assert len(rows) == 2
    # tool_use row: text block dropped, tool_use kept verbatim
    assert rows[0]["role"] == "assistant"
    assert [b["type"] for b in rows[0]["content"]] == ["tool_use"]
    assert rows[0]["content"][0]["input"] == {"name": "pdf"}
    # tool_result row
    assert rows[1]["role"] == "user"
    assert rows[1]["content"][0]["type"] == "tool_result"
    assert rows[1]["content"][0]["tool_use_id"] == "t1"


def test_split_skips_pure_text_message():
    # The final "final answer" message carries no tool block -> no row for it.
    rows = _split_turn_into_rows(_turn_slice())
    assert all(
        b["type"] in ("tool_use", "tool_result")
        for row in rows for b in row["content"]
    )


def test_sanitize_replaces_image_with_replay_marker():
    block = {"type": "tool_result", "tool_use_id": "x", "content": [
        {"type": "image", "source": {"data": "AAAA"}},
        {"type": "text", "text": "kept"},
    ]}
    out = _sanitize_tool_result(block)
    assert out["content"][0]["type"] == "text"
    assert "omitted from replayed history" in out["content"][0]["text"]
    assert out["content"][1] == {"type": "text", "text": "kept"}


def test_sanitize_leaves_string_result_untouched():
    block = {"type": "tool_result", "tool_use_id": "x", "content": "plain string"}
    assert _sanitize_tool_result(block) == block


# --- passthrough ------------------------------------------------------------

def test_to_openai_message_passes_tool_blocks_verbatim():
    m = Message(role="assistant", content=[
        {"type": "tool_use", "id": "t1", "name": "recall_skill", "input": {"name": "pdf"}},
    ])
    dumped = m.to_openai_message().model_dump()
    assert isinstance(dumped["content"], list)
    assert dumped["content"][0]["id"] == "t1"
    assert dumped["content"][0]["input"] == {"name": "pdf"}


def test_to_openai_message_collapses_input_text():
    m = Message(role="user", content=[{"type": "input_text", "text": "hello"}])
    assert m.to_openai_message().model_dump()["content"] == "hello"


# --- persistence + ordering + UI filter -------------------------------------

def _persist_turn(session, conv):
    session.add(Message(conversation_id=conv.id, role="user", content="use recall_skill"))
    session.commit()
    tool_rows = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "recall_skill", "input": {"name": "pdf"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "BODY"}]},
    ]
    ConversationService(session).save_assistant_turn(
        conv.id, "done", [], harness="anton", tool_rows=tool_rows,
    )


def test_rows_ordered_by_seq_not_role(session, conversation):
    """tool_result (role=user) must NOT sort ahead of tool_use (role=assistant)
    despite the user-before-assistant tiebreak, because they share created_at."""
    _persist_turn(session, conversation)
    rows = ConversationService(session).get_ordered_messages(conversation.id)
    shapes = [
        (r.role.value if hasattr(r.role, "value") else r.role,
         "list" if isinstance(r.content, list) else "str")
        for r in rows
    ]
    assert shapes == [
        ("user", "str"),        # user input
        ("assistant", "list"),  # tool_use
        ("user", "list"),       # tool_result
        ("assistant", "str"),   # visible answer
    ]


def test_get_messages_hides_tool_rows(session, conversation):
    _persist_turn(session, conversation)
    ui = ConversationService(session).get_messages(conversation.id)
    assert [type(m["content"]).__name__ for m in ui] == ["str", "str"]
    assert all(not _is_tool_row(m["content"]) for m in ui)


def test_history_round_trip_is_valid_tool_sequence(session, conversation):
    _persist_turn(session, conversation)
    svc = ConversationService(session)
    history = [
        m.to_openai_message().model_dump()
        for m in svc.get_ordered_messages(conversation.id)
        if m.role in {"user", "assistant"}
    ]
    # user -> assistant(tool_use) -> user(tool_result) -> assistant(text)
    assert history[1]["content"][0]["type"] == "tool_use"
    assert history[2]["content"][0]["type"] == "tool_result"
    assert history[2]["content"][0]["tool_use_id"] == history[1]["content"][0]["id"]
    assert isinstance(history[3]["content"], str)


def test_is_tool_row_discriminates():
    assert _is_tool_row([{"type": "tool_use", "id": "x"}])
    assert _is_tool_row([{"type": "tool_result", "tool_use_id": "x"}])
    assert not _is_tool_row("plain text")
    assert not _is_tool_row([{"type": "input_text", "text": "hi"}])
    assert not _is_tool_row([])


# --- monotonic seq + delete_turn across turns -------------------------------

def _persist_named_turn(session, conv, user_text, answer):
    """One tool-using turn via the real service path (monotonic seq)."""
    svc = ConversationService(session)
    svc.save_user_message(conv.id, user_text)
    tool_rows = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t", "name": "x", "input": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": "B"}]},
    ]
    svc.save_assistant_turn(conv.id, answer, [], harness="anton", tool_rows=tool_rows)


def _visible(session, conv):
    return [
        (m["role"].value if hasattr(m["role"], "value") else m["role"], m["content"])
        for m in ConversationService(session).get_messages(conv.id)
    ]


def test_turns_stay_ordered_within_same_second(session, conversation):
    """Every row here shares one created_at (server now() is second-precision);
    monotonic seq alone must keep the turns from interleaving."""
    _persist_named_turn(session, conversation, "q1", "a1")
    _persist_named_turn(session, conversation, "q2", "a2")
    assert _visible(session, conversation) == [
        ("user", "q1"), ("assistant", "a1"),
        ("user", "q2"), ("assistant", "a2"),
    ]


def test_delete_second_turn_keeps_first(session, conversation):
    """Deleting the 2nd visible turn must NOT destroy the 1st turn's answer:
    hidden tool_use rows (role=assistant) must not inflate the turn count."""
    _persist_named_turn(session, conversation, "q1", "a1")
    _persist_named_turn(session, conversation, "q2", "a2")
    svc = ConversationService(session)

    deleted = svc.delete_turn(conversation.id, 1)  # UI's 2nd assistant turn

    assert _visible(session, conversation) == [("user", "q1"), ("assistant", "a1")]
    # Turn 2's four rows (user + tool_use + tool_result + answer) are gone.
    assert deleted == 4
    # Turn 1's tool rows survive for LLM replay.
    all_rows = svc.get_ordered_messages(conversation.id)
    assert sum(1 for m in all_rows if _is_tool_row(m.content)) == 2


def test_delete_first_turn_clears_all(session, conversation):
    _persist_named_turn(session, conversation, "q1", "a1")
    _persist_named_turn(session, conversation, "q2", "a2")
    svc = ConversationService(session)

    svc.delete_turn(conversation.id, 0)

    assert svc.get_ordered_messages(conversation.id) == []


def test_hermes_history_excludes_tool_rows(session, conversation):
    """Hermes builds OpenAI-format history and can't consume anton's Anthropic
    tool blocks, so tool rows are filtered out (mirrors HermesHarness)."""
    _persist_named_turn(session, conversation, "q1", "a1")
    ordered = ConversationService(session).get_ordered_messages(conversation.id)
    history = [
        m.to_openai_message().model_dump()
        for m in ordered
        if m.role in {"user", "assistant"} and not _is_tool_row(m.content)
    ]
    # Only the plain user question and assistant answer survive — no list
    # content (which is where the tool blocks would have leaked).
    assert [type(h["content"]).__name__ for h in history] == ["str", "str"]
