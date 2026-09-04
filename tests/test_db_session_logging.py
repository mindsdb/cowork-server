from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi import HTTPException

from cowork.db import session as db_session


def _drive(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> tuple[mock.Mock, mock.Mock]:
    db = mock.Mock()
    monkeypatch.setattr(db_session, "get_engine", lambda db_uri=None: object())
    monkeypatch.setattr(db_session, "get_session_factory", lambda engine: (lambda: db))
    logger = mock.Mock()
    monkeypatch.setattr(db_session, "logger", logger)

    generator = db_session.get_session(db_uri="sqlite://")
    next(generator)
    with pytest.raises(type(exc)):
        generator.throw(exc)
    return db, logger


def test_a_4xx_answer_rolls_back_without_a_logged_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 409 "steer while an approval is pending" or a 404 is the endpoint's
    # intended answer; it must not fill the sidecar log with tracebacks.
    db, logger = _drive(monkeypatch, HTTPException(status_code=409, detail="Resolve the pending approval first"))

    db.rollback.assert_called_once()
    db.close.assert_called_once()
    logger.exception.assert_not_called()


def test_unexpected_failures_still_log_a_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    db, logger = _drive(monkeypatch, RuntimeError("database gone"))

    db.rollback.assert_called_once()
    db.close.assert_called_once()
    logger.exception.assert_called_once()


def test_a_5xx_http_exception_is_still_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    _, logger = _drive(monkeypatch, HTTPException(status_code=500, detail="Coding operation failed"))

    logger.exception.assert_called_once()
