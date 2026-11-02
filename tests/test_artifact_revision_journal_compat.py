"""A journal an earlier build wrote reads, answers and writes back unchanged.

The fixture is an artifact journal cowork-server wrote at staging dc0fcd42,
with what that build answered on it. A later build must give the same answers,
keep the records it doesn't change byte for byte, and write back each record
it changes with the same keys, so a journal moves between builds in either
direction.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from cowork.services.artifact_draft_review import draft_review_grant
from cowork.services.artifact_revisions import (
    active_agent_repair,
    agent_repair_detail,
    finalize_agent_repair,
    list_revisions,
    revision_with_content,
    save_source,
)

FIXTURE = yaml.safe_load(
    (Path(__file__).parent / "fixtures" / "artifact_revisions" / "journal_at_dc0fcd42.yaml").read_text(
        encoding="utf-8"
    )
)
EXPECTED = FIXTURE["expected"]
READY = FIXTURE["ids"]["ready_repair"]
CANCELLED = FIXTURE["ids"]["cancelled_repair"]


@pytest.fixture
def folder(tmp_path) -> Path:
    folder = tmp_path / "alice" / ".anton" / "artifacts" / "report"
    for name, text in {**FIXTURE["artifact"], **{f".revisions/{name}": text for name, text in FIXTURE["journal"].items()}}.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return folder


def _metadata(folder: Path) -> dict:
    return json.loads((folder / "metadata.json").read_text(encoding="utf-8"))


def _journal(folder: Path, name: str) -> bytes:
    return (folder / ".revisions" / name).read_bytes()


def _keys_and_values(record: dict) -> list:
    """A record's keys in their order, each with its value."""
    return list(record.items())


def test_the_journal_answers_as_the_build_that_wrote_it(folder):
    revisions = list_revisions(folder)

    assert revisions == EXPECTED["revisions"]
    assert [_keys_and_values(revision) for revision in revisions] == [
        _keys_and_values(revision) for revision in EXPECTED["revisions"]
    ]
    assert {
        revision["id"]: revision_with_content(folder, revision["id"])["content"] for revision in revisions
    } == EXPECTED["contents"]
    assert active_agent_repair(folder, "brief.md") == EXPECTED["active_repair"]
    assert agent_repair_detail(folder, READY) == EXPECTED["repair_detail"]
    assert draft_review_grant(folder) == EXPECTED["grant"]


def test_a_restore_and_a_decision_keep_every_record_they_do_not_change(folder):
    metadata = _metadata(folder)
    head = list_revisions(folder)[0]
    untouched = {name: _journal(folder, name) for name in ("comments.json", "draft-review.json", f"repairs/{CANCELLED}.json")}
    ready = json.loads(_journal(folder, f"repairs/{READY}.json"))
    first = EXPECTED["revisions"][-1]

    restored = save_source(
        folder,
        metadata,
        metadata["id"],
        content=revision_with_content(folder, first["id"])["content"],
        expected_revision_id=head["id"],
        actor_id="alice",
        summary=f"Restored revision {first['number']}",
    )["revision"]
    finalize_agent_repair(
        folder, metadata, metadata["id"], READY, "accepted", expected_head_revision_id=restored["id"]
    )

    manifest = json.loads(_journal(folder, "manifest.json"))
    assert [_keys_and_values(entry) for entry in manifest["revisions"][:-1]] == [
        _keys_and_values(entry) for entry in reversed(EXPECTED["revisions"])
    ]
    assert manifest["revisions"][-1] == restored
    assert (folder / "brief.md").read_text(encoding="utf-8") == EXPECTED["contents"][first["id"]]
    assert {name: _journal(folder, name) for name in untouched} == untouched
    accepted = json.loads(_journal(folder, f"repairs/{READY}.json"))
    assert list(accepted) == list(ready)
    assert {**accepted, "updatedAt": ready["updatedAt"]} == {**ready, "status": "accepted"}
    assert draft_review_grant(folder) == EXPECTED["grant"]
