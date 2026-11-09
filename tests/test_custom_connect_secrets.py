"""A submitted value never reaches a log record or a response, whatever the connect does."""

from __future__ import annotations

import gzip
import logging

import httpx
import pytest

from tests import test_custom_connect as connect

SECRET = connect.SECRET
_client = connect._client
_connect = connect._connect
engine = connect.engine
world = connect.world


class _Every(logging.Handler):
    """Collect every record any logger emits, formatted with its arguments."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())
        if record.exc_info:
            self.lines.append(logging.Formatter().formatException(record.exc_info))


@pytest.fixture()
def records():
    """Attach to the root and to every named logger, whatever their propagation."""
    handler = _Every()
    loggers = [logging.getLogger()] + [
        logger for logger in logging.root.manager.loggerDict.values() if isinstance(logger, logging.Logger)
    ]
    saved = [(logger, logger.level) for logger in loggers]
    for logger in loggers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    yield handler.lines
    for logger, level in saved:
        logger.removeHandler(handler)
        logger.setLevel(level)


def _echo(status: int) -> tuple[int, bytes]:
    return status, f'{{"error":"bad key {SECRET}","key":"{SECRET}"}}'.encode()


@pytest.mark.parametrize(
    ("setup", "body"),
    [
        ("pass", {}),
        ("401", {}),
        ("500", {}),
        ("unprotected", {}),
        ("gzip", {}),
        ("dropped", {}),
        ("refused-address", {"address": f"{SECRET}.mindshub.ai"}),
        ("trailing-newline", {"values": {"api_key": SECRET + "\n"}}),
        ("non-ascii", {"values": {"api_key": SECRET + "é"}}),
        ("unknown-field", {"values": {"api_key": SECRET, SECRET: "x"}}),
    ],
)
def test_a_planted_value_reaches_no_log_and_no_response(engine, world, monkeypatch, records, setup, body):
    if setup == "401":
        world["protected"] = _echo(401)
    elif setup == "500":
        world["protected"] = _echo(500)
    elif setup == "unprotected":
        world["unprotected"] = _echo(200)
    elif setup == "gzip":
        world["protected"] = (500, gzip.compress(_echo(500)[1]))
    elif setup == "dropped":
        async def drop(request):
            raise httpx.ReadError(f"connection reset while sending {request.headers['authorization']}")

        world["protected"] = drop

    res = _connect(_client(engine, monkeypatch), **body)

    assert res.status_code in (201, 409, 422)
    assert SECRET not in res.text
    assert records, "the capture saw nothing; the check would pass for nothing"
    assert not [line for line in records if SECRET in line]
