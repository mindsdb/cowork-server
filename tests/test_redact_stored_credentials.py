"""The cleanup script redacts tokens already stored in history, and nothing the user or model wrote."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.message_event import MessageEvent
from cowork.models.project import Project
from cowork.schemas.responses import Role
from scripts.redact_stored_credentials import redact_stored_credentials

TOKEN = "opq" + "Z9" * 15
CELL_CODE = f"headers = {{'Authorization': 'Bearer {TOKEN}'}}"


def _cell_event() -> dict:
    """A scratchpad result event as an older build stored it: the raw cell as JSON in `content`."""
    cell = {"code": CELL_CODE, "stdout": f'{{"access_token": "{TOKEN}"}}\n', "stderr": "", "error": None, "logs": ""}
    return {"type": "response.in_progress", "thought_role": "thought_scratchpad_result", "content": json.dumps(cell)}


@pytest.fixture()
def stored(tmp_path):
    """A database and streams directory holding a token in every place the script covers."""
    engine = create_engine(f"sqlite:///{tmp_path / 'history.db'}")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)
    project = Project(name="p", path=str(tmp_path / "p"))
    session.add(project)
    session.flush()
    conversation = Conversation(topic="t", project_id=project.id, history_summary=f"Fetched access_token={TOKEN} earlier.")
    session.add(conversation)
    session.flush()
    rows = {
        "user_text": Message(conversation_id=conversation.id, role=Role.user, content=f"my token is {TOKEN}", seq=0),
        "tool_use": Message(conversation_id=conversation.id, role=Role.assistant, seq=1, content=[
            {"type": "tool_use", "id": "t1", "name": "scratchpad", "input": {"code": CELL_CODE}},
        ]),
        "tool_result": Message(conversation_id=conversation.id, role=Role.user, seq=2, content=[
            {"type": "tool_result", "tool_use_id": "t1", "content": f'{{"refresh_token": "{TOKEN}"}}'},
        ]),
        "answer": Message(conversation_id=conversation.id, role=Role.assistant, seq=3, content=f"Bearer {TOKEN}"),
    }
    session.add_all(rows.values())
    session.flush()
    event = MessageEvent(message_id=rows["answer"].id, sequence_number=0, event_data=_cell_event())
    session.add(event)
    session.commit()

    streams = tmp_path / "streams"
    turn_file = streams / str(conversation.id) / "turn_000001.jsonl"
    turn_file.parent.mkdir(parents=True)
    turn_file.write_text(
        json.dumps({"seq": 1, "type": "response.in_progress", "data": _cell_event()}, ensure_ascii=False) + "\n"
        + '{"seq": 2, "type": "resp',
        encoding="utf-8",
    )
    yield session, streams, rows, event, conversation, turn_file
    session.close()


def test_a_dry_run_counts_and_changes_nothing(stored):
    """Without --apply every location is counted and nothing is written."""
    session, streams, rows, event, conversation, turn_file = stored
    before = turn_file.read_text(encoding="utf-8")

    report = redact_stored_credentials(session, streams, apply=False)

    assert (report.message_events, report.messages, report.history_summaries, report.turn_files) == (1, 2, 1, 1)
    session.expire_all()
    assert TOKEN in session.get(MessageEvent, event.id).event_data["content"]
    assert TOKEN in session.get(Message, rows["answer"].id).content
    assert turn_file.read_text(encoding="utf-8") == before


def test_apply_redacts_output_and_keeps_what_was_written_by_hand(stored):
    """Tool output, answers, summaries and turn files lose the token; user text and code keep theirs."""
    session, streams, rows, event, conversation, turn_file = stored

    redact_stored_credentials(session, streams, apply=True)
    session.expire_all()

    cell = json.loads(session.get(MessageEvent, event.id).event_data["content"])
    assert cell["stdout"] == '{"access_token": "[REDACTED_TOKEN]"}\n'
    assert cell["code"] == CELL_CODE
    assert session.get(Message, rows["tool_result"].id).content[0]["content"] == '{"refresh_token": "[REDACTED_TOKEN]"}'
    assert session.get(Message, rows["answer"].id).content == "Bearer [REDACTED_TOKEN]"
    assert session.get(Conversation, conversation.id).history_summary == "Fetched access_token=[REDACTED_TOKEN] earlier."
    assert session.get(Message, rows["user_text"].id).content == f"my token is {TOKEN}"
    assert session.get(Message, rows["tool_use"].id).content[0]["input"]["code"] == CELL_CODE

    lines = turn_file.read_text(encoding="utf-8").splitlines()
    record_cell = json.loads(json.loads(lines[0])["data"]["content"])
    assert record_cell["stdout"] == '{"access_token": "[REDACTED_TOKEN]"}\n'
    assert lines[1] == '{"seq": 2, "type": "resp'


def test_a_second_run_finds_nothing_left(stored):
    """Applying twice is safe: markers are not redacted again."""
    session, streams, *_ = stored
    redact_stored_credentials(session, streams, apply=True)

    report = redact_stored_credentials(session, streams, apply=False)

    assert (report.message_events, report.messages, report.history_summaries, report.turn_files) == (0, 0, 0, 0)


def test_a_missing_streams_directory_is_not_an_error(stored, tmp_path):
    """A deployment without file-backed turn buffers still cleans its database."""
    session, *_ = stored
    report = redact_stored_credentials(session, tmp_path / "no-such-dir", apply=False)
    assert report.turn_files == 0
    assert report.messages == 2
