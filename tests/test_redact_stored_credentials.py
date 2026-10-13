"""The cleanup script redacts tokens already stored in history, and nothing the user or model wrote."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

import scripts.redact_stored_credentials as redact_module
from cowork.models.conversation import Conversation
from cowork.models.message import Message
from cowork.models.message_event import MessageEvent
from cowork.models.project import Project
from cowork.schemas.responses import Role
from scripts.redact_stored_credentials import redact_stored_credentials, scrub_value

TOKEN = "opq" + "Z9" * 15
CELL_CODE = f"headers = {{'Authorization': 'Bearer {TOKEN}'}}"


@dataclass
class Stored:
    """The seeded database and streams directory, with the ids a test reads back."""

    session: Session
    streams: Path
    turn_file: Path
    messages: dict[str, UUID]
    cell_event: UUID
    call_event: UUID
    conversation: UUID


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
    cell_event = MessageEvent(message_id=rows["answer"].id, sequence_number=0, event_data=_cell_event())
    call_event = MessageEvent(message_id=rows["answer"].id, sequence_number=1, event_data={
        "type": "response.in_progress",
        "thought_role": "thought_tool_call_message",
        "content": json.dumps({"action": "exec", "name": "main", "code": CELL_CODE}),
    })
    session.add_all([cell_event, call_event])
    session.commit()

    streams = tmp_path / "streams"
    turn_file = streams / str(conversation.id) / "turn_000001.jsonl"
    turn_file.parent.mkdir(parents=True)
    turn_file.write_text(
        json.dumps({"seq": 1, "type": "response.in_progress", "data": _cell_event()}, ensure_ascii=False) + "\n"
        + '{"seq": 2, "type": "resp',
        encoding="utf-8",
    )
    yield Stored(
        session=session,
        streams=streams,
        turn_file=turn_file,
        messages={name: row.id for name, row in rows.items()},
        cell_event=cell_event.id,
        call_event=call_event.id,
        conversation=conversation.id,
    )
    session.close()


def _counts(report) -> tuple[int, int, int, int]:
    """The report's per-store counts, in a fixed order."""
    return report.message_events, report.messages, report.history_summaries, report.turn_files


def test_a_dry_run_counts_and_changes_nothing(stored):
    """Without --apply every location is counted and nothing is written."""
    before = stored.turn_file.read_text(encoding="utf-8")

    report = redact_stored_credentials(stored.session, stored.streams, apply=False)

    assert _counts(report) == (1, 2, 1, 1)
    assert TOKEN in stored.session.get(MessageEvent, stored.cell_event).event_data["content"]
    assert TOKEN in stored.session.get(Message, stored.messages["answer"]).content
    assert stored.turn_file.read_text(encoding="utf-8") == before


def test_apply_redacts_output_and_keeps_what_was_written_by_hand(stored):
    """Tool output, answers, summaries and turn files lose the token; user text and code keep theirs."""
    session = stored.session
    redact_stored_credentials(session, stored.streams, apply=True)

    cell = json.loads(session.get(MessageEvent, stored.cell_event).event_data["content"])
    assert cell["stdout"] == '{"access_token": "[REDACTED_TOKEN]"}\n'
    assert cell["code"] == CELL_CODE
    assert json.loads(session.get(MessageEvent, stored.call_event).event_data["content"])["code"] == CELL_CODE
    tool_result = session.get(Message, stored.messages["tool_result"]).content[0]["content"]
    assert tool_result == '{"refresh_token": "[REDACTED_TOKEN]"}'
    assert session.get(Message, stored.messages["answer"]).content == "Bearer [REDACTED_TOKEN]"
    assert session.get(Conversation, stored.conversation).history_summary == "Fetched access_token=[REDACTED_TOKEN] earlier."
    assert session.get(Message, stored.messages["user_text"]).content == f"my token is {TOKEN}"
    assert session.get(Message, stored.messages["tool_use"]).content[0]["input"]["code"] == CELL_CODE

    lines = stored.turn_file.read_text(encoding="utf-8").splitlines()
    record_cell = json.loads(json.loads(lines[0])["data"]["content"])
    assert record_cell["stdout"] == '{"access_token": "[REDACTED_TOKEN]"}\n'
    assert lines[1] == '{"seq": 2, "type": "resp'
    assert [p.name for p in stored.turn_file.parent.iterdir()] == [stored.turn_file.name]


def test_a_second_run_finds_nothing_left(stored):
    """Applying twice is safe: markers are not redacted again."""
    redact_stored_credentials(stored.session, stored.streams, apply=True)

    report = redact_stored_credentials(stored.session, stored.streams, apply=False)

    assert _counts(report) == (0, 0, 0, 0)


def test_a_missing_streams_directory_is_not_an_error(stored, tmp_path):
    """A deployment without file-backed turn buffers still cleans its database."""
    report = redact_stored_credentials(stored.session, tmp_path / "no-such-dir", apply=False)
    assert report.turn_files == 0
    assert report.messages == 2


def test_code_and_input_keys_in_printed_output_are_scrubbed():
    """Only a cell's code and a tool_use input are kept; the same key names in printed API output are not."""
    printed = {"stdout_obj": {"input": f"Bearer {TOKEN}", "code": f"access_token={TOKEN}"}}
    assert TOKEN not in json.dumps(scrub_value(printed))


def test_small_batches_still_reach_every_row(stored, monkeypatch):
    """Rows are read and committed in batches; a batch size of one still covers everything."""
    monkeypatch.setattr(redact_module, "BATCH_SIZE", 1)

    report = redact_stored_credentials(stored.session, stored.streams, apply=True)

    assert _counts(report)[:3] == (1, 2, 1)
    assert redact_stored_credentials(stored.session, stored.streams, apply=False).messages == 0


def test_a_turn_file_that_cannot_be_written_is_reported(stored):
    """A write failure is reported per file, the old file stays whole, and the database changes still land."""
    stored.turn_file.parent.chmod(0o500)
    try:
        report = redact_stored_credentials(stored.session, stored.streams, apply=True)
    finally:
        stored.turn_file.parent.chmod(0o700)

    assert report.skipped_turn_files == [f"{stored.turn_file.name}: PermissionError"]
    assert TOKEN not in stored.session.get(Message, stored.messages["answer"]).content
    assert TOKEN in stored.turn_file.read_text(encoding="utf-8")
