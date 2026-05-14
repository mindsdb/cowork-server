from __future__ import annotations

from pathlib import Path

from cowork.runtime.artifacts import scan_artifact_root, validate_artifact_folder
from cowork.runtime.conversations import CoworkConversationStore
from cowork.runtime.events import cowork_event_to_legacy_sse, iter_sse_payloads, normalize_legacy_payload
from cowork.runtime.schemas import CoworkMessage, ResolvedInferenceProfile


def test_event_roundtrip() -> None:
    event = normalize_legacy_payload({"type": "response.output_text.delta", "delta": "hi"}, "turn_1")
    assert event.type == "response.delta"
    payloads = iter_sse_payloads(cowork_event_to_legacy_sse(event))
    assert payloads[0][0] == "response.output_text.delta"
    assert payloads[0][1]["delta"] == "hi"


def test_conversation_store_persists_turns(tmp_path: Path) -> None:
    store = CoworkConversationStore(tmp_path)
    profile = ResolvedInferenceProfile(
        provider_type="minds-cloud",
        provider_label="MindsHub",
        planning_model="_reason_",
        coding_model="_code_",
    )
    conv = store.create(project_id="general", harness="anton", inference=profile, title="Hello")
    conv = store.append_message(conv, CoworkMessage(role="user", content="Hello"))
    conv, turn, _assistant = store.start_turn(conv, conv.messages[-1].id)
    event = normalize_legacy_payload({"type": "response.output_text.delta", "delta": "Hi"}, turn.id)
    store.append_event(conv, turn.id, event)
    store.finish_turn(conv, turn.id, "completed")

    reloaded = store.get(conv.id)
    assert reloaded is not None
    assert reloaded.messages[-1].content == "Hi"
    assert reloaded.turns[-1].status == "completed"


def test_artifact_validation_ignores_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    folder = root / "deck"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text('{"name":"Deck","primary":"deck.md"}', encoding="utf-8")
    (folder / "deck.md").write_text("# Deck", encoding="utf-8")

    assert validate_artifact_folder(folder, root)["title"] == "Deck"
    assert scan_artifact_root(root)[0]["primary"] == "deck.md"
    assert validate_artifact_folder(tmp_path / "outside", root) is None

