"""A remote turn's failure line names the pod's error by a fixed label and the
classified code. The pod's error text can quote the provider, so none of it
reaches the record."""
from __future__ import annotations

import logging

import pytest

from cowork.handlers import turn_errors as te
from cowork.turnqueue import producer as prod
from test_producer import _failed_turn, _stub_llm_mint  # noqa: F401  (_stub_llm_mint is an autouse fixture)

SENTINEL = "SECRET-PROMPT-TEXT"
[TURN_INTERRUPTED] = [text for text in te.SELF_AUTHORED_TURN_FAILURES if text.startswith("TurnInterrupted:")]
[TURN_WORKER_LOST] = [text for text in te.SELF_AUTHORED_TURN_FAILURES if text.startswith("TurnWorkerLost:")]


@pytest.mark.parametrize(
    ("error", "error_type", "error_code"),
    [
        (
            f"ContentTooLargeError: The image is too large. The provider said: {SENTINEL}",
            te.CONTENT_TOO_LARGE_TYPE_NAME,
            te.CONTENT_TOO_LARGE_CODE,
        ),
        (f"RequestRefusedError: The provider said: {SENTINEL}", "unmapped", te.GENERIC_TURN_ERROR_CODE),
        # No "TypeName:" prefix: the whole string would pass for a type name.
        (f"{SENTINEL} without a type prefix", "unmapped", te.GENERIC_TURN_ERROR_CODE),
        (
            f"{te.POD_STREAM_ENDED_PREFIX}; stderr tail: {SENTINEL}",
            "pod_stream_ended",
            te.GENERIC_TURN_ERROR_CODE,
        ),
    ],
    ids=["mapped", "unmapped", "no-prefix", "self-authored"],
)
async def test_a_failed_turn_logs_a_fixed_label_and_the_code_never_the_pods_text(
    monkeypatch, caplog, error, error_type, error_code
):
    # A logging config an earlier test loads can leave this logger disabled.
    monkeypatch.setattr(prod.logger, "disabled", False)
    with caplog.at_level(logging.WARNING, logger=prod.logger.name):
        failed = await _failed_turn(monkeypatch, {"error": error})

    assert failed["code"] == error_code
    [record] = [
        record for record in caplog.records
        if record.name == prod.logger.name and record.levelno == logging.WARNING
    ]
    assert record.getMessage() == (
        f"Remote turn failed conversation=conv-1 correlation_id=r "
        f"error_type={error_type} error_code={error_code}"
    )
    assert SENTINEL not in repr(record.__dict__)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TURN_INTERRUPTED, "TurnInterrupted"),
        (TURN_WORKER_LOST, "TurnWorkerLost"),
        (f"{te.POD_STREAM_ENDED_PREFIX}; stderr tail: {SENTINEL}", "pod_stream_ended"),
        (f"{te.TURN_ABORTED_TIMEOUT_PREFIX} after 900s {SENTINEL}", "turn_aborted_timeout"),
        (f"{te.TURN_ABORTED_STALL_PREFIX} of 300s {SENTINEL}", "turn_aborted_stall"),
        (f"job corr-{SENTINEL} {te.MISSING_ORGANIZATION_SUFFIX}", "missing_organization"),
        (f"live pod sp-{SENTINEL} did not reach Running within 120s", "live_pod_never_ran"),
        (f"pod sp-{SENTINEL} belongs to scratchpad 'a', not 'b'", "pod_identity_mismatch"),
        (te.REMOTE_CANCEL_LITERAL, "cancelled"),
        (te.REMOTE_CANCEL_VIA_FAIL_JOB, "cancelled"),
        (prod.UNRESPONSIVE_WORKER_ERROR, te.WORKER_UNRESPONSIVE_TYPE_NAME),
        ("ContentValidationError: Invalid value: 'image'.", "ContentValidationError"),
        ("ProviderAuthError", "ProviderAuthError"),
        (f"ConnectionError: Invalid API key {SENTINEL}", "ConnectionError"),
        (f"ConnectionError: {SENTINEL}", "unmapped"),
        # A provider's own text that ends like a self-authored sentence.
        (f"{SENTINEL}: {TURN_INTERRUPTED}", "unmapped"),
        ("RuntimeError: boom", "unmapped"),
        ("", "unmapped"),
        (None, "unmapped"),
    ],
)
def test_remote_error_label_names_the_branch_never_the_pods_text(error, expected):
    label = te.remote_error_label(error=error)

    assert label == expected
    assert SENTINEL not in label
