"""A failed attachment upload names its conversation on the log line, also
when the database filter replaces the message."""
from __future__ import annotations

import io
import logging
from uuid import uuid4

import pytest
from fastapi import HTTPException, UploadFile
from sqlalchemy.exc import OperationalError

from cowork.api.v1.endpoints.compat import stubs
from cowork.services import files

FILENAME = "q3-report.pdf"
STATEMENT = "INSERT INTO files (purpose) VALUES (%(purpose)s)"
DRIVER_TEXT = "server closed the connection unexpectedly"


def _file_service_raising(*, error: Exception) -> type:
    class FailingFileService:
        def __init__(self, scoped):
            pass

        async def create_file(self, *, upload, purpose):
            raise error

    return FailingFileService


async def _upload(*, monkeypatch, conversation_id: str, error: Exception) -> None:
    monkeypatch.setattr(files, "FileService", _file_service_raising(error=error))
    with pytest.raises(HTTPException) as failed:
        await stubs.upload_attachment(
            "proj", conversation_id, scoped=None,
            files=[UploadFile(io.BytesIO(b"bytes"), filename=FILENAME)],
        )
    assert failed.value.status_code == 500


def _error_record(*, caplog) -> logging.LogRecord:
    [record] = [
        record for record in caplog.records
        if record.name == stubs.logger.name and record.levelno == logging.ERROR
    ]
    return record


async def test_a_database_failure_keeps_the_conversation_on_the_line(monkeypatch, caplog, owned_logger):
    conversation_id = str(uuid4())
    logged = owned_logger(stubs.logger.name)

    await _upload(
        monkeypatch=monkeypatch, conversation_id=conversation_id,
        error=OperationalError(STATEMENT, {"purpose": f"attachment:{conversation_id}"}, Exception(DRIVER_TEXT)),
    )

    record = _error_record(caplog=caplog)
    assert record.getMessage().startswith("Database operation failed: error_type=OperationalError ")
    assert record.conversation_id == conversation_id
    output = logged.output()
    assert f"[Conversation:{conversation_id}]: Database operation failed" in output
    assert STATEMENT not in output
    assert DRIVER_TEXT not in output


async def test_any_other_failure_names_the_project_and_the_file(monkeypatch, caplog, owned_logger):
    conversation_id = str(uuid4())
    logged = owned_logger(stubs.logger.name)

    await _upload(monkeypatch=monkeypatch, conversation_id=conversation_id, error=ValueError("unreadable upload"))

    record = _error_record(caplog=caplog)
    assert record.getMessage() == f"Attachment upload failed (project=proj file={FILENAME})"
    assert record.conversation_id == conversation_id
    assert (
        f"[Conversation:{conversation_id}]: Attachment upload failed (project=proj file={FILENAME})"
        in logged.output()
    )
