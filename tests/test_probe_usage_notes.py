"""The credential prober sees the connector's usage notes, so API traps (e.g. a
redirect that strips Authorization) can live there instead of in the
user-facing method description."""
import asyncio

from cowork.handlers import probe as probe_handler
from cowork.services.connectors.probe import CredentialProbe, ProbeOutcome
from cowork.services.connectors.submissions import SubmissionStore


def _prompt(**kw):
    probe = CredentialProbe(
        engine="roam_research", credentials={}, llm_client=None, workspace=None, **kw
    )
    return probe._build_prompt("/tmp/x.env", ["DS_TOKEN"])


def test_notes_are_rendered_as_a_known_quirks_section():
    prompt = _prompt(usage_notes="Re-attach Authorization after the 308.")

    assert "——— KNOWN API QUIRKS ———" in prompt
    assert "Re-attach Authorization after the 308." in prompt
    assert prompt.index("——— KNOWN API QUIRKS ———") > prompt.index("——— CURRENT FORM ROSTER ———")
    assert prompt.index("——— KNOWN API QUIRKS ———") < prompt.index("——— STEPS (follow in order) ———")


def test_no_notes_no_section():
    assert "KNOWN API QUIRKS" not in _prompt()
    assert "KNOWN API QUIRKS" not in _prompt(usage_notes="   ")


def test_handler_hands_the_resolved_notes_to_the_probe(monkeypatch):
    created: list[dict] = []

    class RecordingProbe:
        def __init__(self, **kwargs):
            created.append(kwargs)

        async def run(self):
            yield "verdict", ProbeOutcome(status="failure", error="stop here")

    monkeypatch.setattr(probe_handler, "CredentialProbe", RecordingProbe)
    monkeypatch.setattr(probe_handler.ProbeHandler, "_build_llm_client", staticmethod(lambda: object()))
    monkeypatch.setattr(
        probe_handler.registry, "usage_notes_for",
        lambda engines: {e: f"NOTE-{e}" for e in engines},
    )
    store = SubmissionStore()
    monkeypatch.setattr(probe_handler, "store", store)
    submission_id = store.stage(
        form_id="langfuse-connector", connector_id="langfuse", conversation_id=None,
        values={"public_key": "pk", "secret_key": "sk"},
    )

    async def drain():
        return [
            chunk async for chunk in probe_handler.ProbeHandler(session=None).run(
                submission_id, "langfuse", None, "", None
            )
        ]

    asyncio.run(drain())

    assert [kw["usage_notes"] for kw in created] == ["NOTE-langfuse"]
    assert created[0]["engine"] == "langfuse"
