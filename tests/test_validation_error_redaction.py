"""Request validation errors must not repeat the submitted body.

Connector submissions carry passwords, DSNs and CA material. FastAPI's
default 422 echoes the offending value back in ``input``, and ``get_session``
logs the same error with ``logger.exception`` when its generator unwinds, so
a malformed body reaches both the response and the server log verbatim.
"""
from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from cowork.server import create_app

#: Stands in for a password or a DSN. ``values`` is typed ``dict[str, Any]``,
#: so a bare string fails validation and FastAPI puts it in ``input``.
SENTINEL = "s3ntinel-p4ssw0rd-do-not-echo"

MALFORMED_BODY = {
    "connector_id": "postgres",
    "method": "host-port",
    "values": SENTINEL,
}


@pytest.fixture(autouse=True)
def _reset_app_settings():
    from cowork.common.settings.app_settings import get_app_settings

    get_app_settings.cache_clear()
    yield
    get_app_settings.cache_clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture()
def session_log(client):
    """Capture the db session logger directly, not through caplog.

    Two things silence this logger for a test that only reads caplog, and
    either one turns "the secret is absent from the log" into an assertion
    over nothing: create_app() calls setup_logging(), whose
    logging.basicConfig(force=True) drops pytest's handler, and any earlier
    test that runs Alembic leaves the logger disabled, because
    cowork/db/alembic/env.py calls fileConfig() with its default
    disable_existing_loggers=True. So take the logger's state for the test.
    """
    from cowork.db import session as session_module

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture(logging.DEBUG)
    logger = session_module.logger
    previous_level, previously_disabled = logger.level, logger.disabled
    logger.setLevel(logging.DEBUG)
    logger.disabled = False
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.disabled = previously_disabled


def test_malformed_body_is_not_echoed_in_the_response(client):
    res = client.post("/api/v1/connectors/submissions/", json=MALFORMED_BODY)

    assert res.status_code == 422
    assert SENTINEL not in res.text


def test_malformed_body_is_not_echoed_in_the_session_log(client, session_log):
    res = client.post("/api/v1/connectors/submissions/", json=MALFORMED_BODY)

    assert res.status_code == 422
    # Pin that the line was emitted before asserting what it does not contain,
    # so the check below cannot pass on an empty log.
    rollback = [r for r in session_log if "rolled back after a client error" in r.getMessage()]
    assert len(rollback) == 1
    assert rollback[0].levelname == "DEBUG"
    assert all(SENTINEL not in record.getMessage() for record in session_log)


def test_validation_detail_still_names_the_location_and_reason(client):
    """Redaction drops the value, not the diagnosis."""
    res = client.post("/api/v1/connectors/submissions/", json=MALFORMED_BODY)

    detail = res.json()["detail"]
    assert detail, "a 422 still has to say what was wrong"
    for entry in detail:
        assert set(entry) == {"loc", "msg", "type"}
    assert any("values" in entry["loc"] for entry in detail)


def test_well_formed_body_still_reaches_the_handler(client):
    """The change is about what a 422 says, not about rejecting more requests."""
    res = client.post(
        "/api/v1/connectors/submissions/",
        json={"connector_id": "not-a-real-connector", "name": "x", "values": {}},
    )

    assert res.status_code == 404
