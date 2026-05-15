from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from cowork.harnesses.legacy_events import normalize_legacy_payload
from cowork.runtime.access import build_access_policy, classify_resource
from cowork.runtime.artifact_events import TurnArtifactCollector
from cowork.runtime.artifacts import snapshot_artifacts
from cowork.runtime.conversations import CoworkConversationStore
from cowork.runtime.events import cowork_event_to_legacy_sse, iter_sse_payloads
from cowork.runtime.inference import build_inference_profile, validate_inference_profile
from cowork.runtime.schemas import CoworkEvent, CoworkMessage, CoworkResourceRef, ProjectContext


class RuntimePrimitiveTests(unittest.TestCase):
    def test_canonical_events_round_trip_to_responses_sse(self) -> None:
        event = normalize_legacy_payload(
            {"type": "response.output_text.delta", "delta": "hello"},
            "turn_1",
        )

        self.assertEqual(event.type, "message.delta")
        payload = iter_sse_payloads(cowork_event_to_legacy_sse(event))[0][1]
        self.assertEqual(payload["type"], "response.output_text.delta")
        self.assertEqual(payload["at_ms"], event.at_ms)
        self.assertEqual(payload["cowork_event_type"], "message.delta")

    def test_event_schema_rejects_unknown_types(self) -> None:
        with self.assertRaises(ValidationError):
            CoworkEvent(type="legacy.random", turn_id="turn_1")

    def test_access_policy_classifies_project_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            policy = build_access_policy(
                project_context=ProjectContext(id="general", name="general", path=str(root)),
                artifact_root=str(artifacts),
                approvals_mode="require",
            )
            decision = classify_resource(policy, CoworkResourceRef(
                resource_type="file",
                operation="write",
                path=str(root / "notes.md"),
                scope=str(root / "notes.md"),
            ))

        self.assertEqual(decision.status, "approval_required")

    def test_artifact_collector_validates_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifacts"
            root.mkdir()
            before = snapshot_artifacts(root)
            collector = TurnArtifactCollector(root)
            self.assertEqual(collector.before, before)
            folder = root / "deck"
            folder.mkdir()
            (folder / "metadata.json").write_text(json.dumps({
                "name": "Deck",
                "type": "document",
                "primary": "deck.md",
            }), encoding="utf-8")
            (folder / "README.md").write_text("# Deck", encoding="utf-8")
            (folder / "deck.md").write_text("Hello", encoding="utf-8")

            events = collector.collect("turn_1")

        self.assertEqual([event.type for event in events], ["artifact.created"])

    def test_conversation_store_restores_canonical_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = CoworkConversationStore(tmp)
            profile = build_inference_profile(
                provider_type="minds-cloud",
                provider_label="MindsHub",
                planning_model="_reason_",
                coding_model="_code_",
            )
            conv = store.create(project_id="general", harness="anton", inference=profile, title="Hello")
            conv = store.append_message(conv, CoworkMessage(role="user", content="Hello"))
            conv, turn, _assistant = store.start_turn(conv, conv.messages[-1].id)
            event = normalize_legacy_payload({"type": "response.output_text.delta", "delta": "Hi"}, turn.id)
            conv = store.append_event(conv, turn.id, event)
            store.finish_turn(conv, turn.id, "completed")

            reloaded = store.get(conv.id)

        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        self.assertEqual(reloaded.messages[-1].content, "Hi")
        self.assertEqual(reloaded.turns[-1].events[-1].type, "message.delta")

    def test_inference_profile_builder_and_validation(self) -> None:
        profile = build_inference_profile(
            provider_type="minds-cloud",
            provider_label="MindsHub",
            base_url="https://mdb.ai/api/v1",
            planning_model="_reason_",
            coding_model="_code_",
        )

        self.assertEqual(profile.planning_model, "_reason_")
        self.assertEqual(validate_inference_profile(profile), (True, ""))


if __name__ == "__main__":
    unittest.main()
