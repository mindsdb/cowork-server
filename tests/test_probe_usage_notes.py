"""The credential prober sees the connector's usage notes, so API traps (e.g. a
redirect that strips Authorization) can live there instead of in the
user-facing method description."""
from cowork.handlers import probe as probe_handler
from cowork.services.connectors.probe import CredentialProbe


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


def test_handler_resolves_notes_from_the_registry(monkeypatch):
    monkeypatch.setattr(
        probe_handler.registry, "usage_notes_for",
        lambda engines: {e: f"NOTE-{e}" for e in engines},
    )
    assert probe_handler._probe_usage_notes("langfuse") == "NOTE-langfuse"

    monkeypatch.setattr(probe_handler.registry, "usage_notes_for", lambda engines: {})
    assert probe_handler._probe_usage_notes("langfuse") is None
