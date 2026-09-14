"""The artifact card carries the conversation that created the artifact, so
"Address with agent" from the artifacts list can resume that chat instead of
opening a new one. The value comes from metadata `provenance`, the same source
`task_objects` uses to decide artifact ownership — these tests pin the two to
one derivation."""

import json
from pathlib import Path
from uuid import uuid4

from cowork.services.artifacts import card_for_folder, origin_conversation_id
from cowork.services.task_objects import _artifact_owner

CONVERSATION = "3f6a1c8e-6b1d-4a2f-9a1e-2c7d5b0e4a11"


def _artifact(tmp_path: Path, meta: dict, name: str = "report.md") -> Path:
    root = tmp_path / f"art-{uuid4().hex[:8]}"
    root.mkdir()
    (root / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (root / name).write_text("# hi\n", encoding="utf-8")
    return root


def _base_meta(**extra) -> dict:
    return {"id": uuid4().hex, "type": "document", "primary": "report.md", **extra}


def test_card_reports_the_creating_conversation(tmp_path: Path):
    root = _artifact(tmp_path, _base_meta(provenance=[{"conversation": CONVERSATION}]))

    assert card_for_folder(root)["originConversationId"] == CONVERSATION


def test_later_provenance_entries_do_not_move_the_origin(tmp_path: Path):
    # Every turn that touches the artifact appends an entry; only the first one
    # names the chat that created it, and that is the chat we resume.
    root = _artifact(tmp_path, _base_meta(provenance=[
        {"conversation": CONVERSATION},
        {"conversation": str(uuid4())},
    ]))

    assert card_for_folder(root)["originConversationId"] == CONVERSATION


def test_card_and_task_objects_agree_on_the_owner(tmp_path: Path):
    root = _artifact(tmp_path, _base_meta(provenance=[{"conversation": CONVERSATION}]))

    assert card_for_folder(root)["originConversationId"] == _artifact_owner(root)


def test_artifact_without_provenance_reports_no_origin(tmp_path: Path):
    # Artifacts written before provenance existed: the client falls back to a
    # new conversation, so the key must be present and empty, never missing.
    root = _artifact(tmp_path, _base_meta())

    card = card_for_folder(root)
    assert card["originConversationId"] == ""
    assert _artifact_owner(root) is None


def test_malformed_provenance_reports_no_origin(tmp_path: Path):
    # Hand-edited or truncated metadata must degrade to "unknown origin", not
    # raise out of the artifacts list.
    for provenance in ({"conversation": CONVERSATION}, ["not-a-dict"], [], [{}]):
        root = _artifact(tmp_path, _base_meta(provenance=provenance))
        assert card_for_folder(root)["originConversationId"] == ""


def test_origin_conversation_id_tolerates_absent_metadata():
    assert origin_conversation_id(None) == ""
    assert origin_conversation_id({}) == ""
