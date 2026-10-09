"""An in-process turn whose artifact index fails names its conversation,
project and slugs on the log line. A database refusal reduces the message to
its call site, so the ids ride the record."""
from __future__ import annotations

import logging

import cowork.services.task_objects as task_objects
from cowork.db.units import DatabaseBusy
from cowork.harnesses.anton_harness import harness as harness_module

from test_harness_scratchpad_teardown import _completes, _drain, turns  # noqa: F401  (turns is a fixture)


def _warnings(*, caplog) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == harness_module.logger.name and record.levelno == logging.WARNING
    ]


async def test_a_refused_index_unit_names_the_conversation_project_and_slugs(
    turns, monkeypatch, caplog, owned_logger
):
    monkeypatch.setattr(
        task_objects, "turn_artifact_changes",
        lambda **_k: task_objects.ArtifactChanges(created=["sales-report"], touched={"sales-report"}),
    )

    async def no_cards(*_a, **_k):
        return []

    monkeypatch.setattr(task_objects, "publish_and_card_turn_artifacts", no_cards)
    real_run_db = harness_module.run_db

    async def refuse_the_index_unit(fn, *, scope):
        if getattr(fn, "func", None) is task_objects.record_new_artifacts:
            raise DatabaseBusy("no connection freed within POOL_TIMEOUT")
        return await real_run_db(fn, scope=scope)

    monkeypatch.setattr(harness_module, "run_db", refuse_the_index_unit)
    logged = owned_logger(harness_module.logger.name, level=logging.WARNING)

    await _drain(turns.run(_completes))

    [record] = _warnings(caplog=caplog)
    assert record.getMessage() == (
        f"Database operation failed: error_type=DatabaseBusy sqlstate=unknown "
        f"site=harness.record_turn_cleanup:{record.lineno}"
    )
    assert (record.conversation_id, record.project_id, record.artifact_slugs) == (
        "conv-1", "proj-1", ("sales-report",),
    )
    assert "[Project:proj-1][Conversation:conv-1][Artifacts:'sales-report']: Database operation failed" in (
        logged.output()
    )


async def test_a_failure_before_any_slug_is_known_renders_no_empty_slug_field(
    turns, monkeypatch, caplog, owned_logger
):
    def _indexing_fails(**_k):
        raise OSError("artifacts dir unreadable")

    monkeypatch.setattr(task_objects, "turn_artifact_changes", _indexing_fails)
    logged = owned_logger(harness_module.logger.name, level=logging.WARNING)

    await _drain(turns.run(_completes))

    [record] = _warnings(caplog=caplog)
    assert record.getMessage() == "Could not index artifacts created this turn"
    assert not hasattr(record, "artifact_slugs")
    assert "[Project:proj-1][Conversation:conv-1]: Could not index artifacts created this turn" in logged.output()
