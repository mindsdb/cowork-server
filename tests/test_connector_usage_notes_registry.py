"""The registry hands out agent-facing usage notes by engine."""
import json

from cowork.services.connectors.specs._registry import ConnectorSpecRegistry


def _spec(tmp_path, cid, **extra):
    (tmp_path / f"{cid}.json").write_text(json.dumps({
        "id": cid, "label": cid, "description": "d", "category": "other",
        "form": {"form_id": f"{cid}-connector", "title": cid, "methods": []},
        **extra,
    }))


def test_returns_notes_only_for_requested_engines_that_declare_them(tmp_path):
    _spec(tmp_path, "a", usage_notes="A-NOTE")
    _spec(tmp_path, "b", usage_notes="B-NOTE")
    _spec(tmp_path, "c")
    _spec(tmp_path, "blank", usage_notes="  ")
    _spec(tmp_path, "wrong_type", usage_notes=["x"])
    reg = ConnectorSpecRegistry(tmp_path)

    assert reg.usage_notes_for(["a", "c", "blank", "wrong_type", "missing"]) == {"a": "A-NOTE"}


def test_accepts_any_iterable(tmp_path):
    _spec(tmp_path, "a", usage_notes="A-NOTE")
    reg = ConnectorSpecRegistry(tmp_path)

    assert reg.usage_notes_for(e for e in ["a"]) == {"a": "A-NOTE"}


def test_notes_are_not_copied_into_the_api_response(tmp_path):
    _spec(tmp_path, "a", usage_notes="A-NOTE")
    reg = ConnectorSpecRegistry(tmp_path)

    assert reg.get_connector("a").usage_notes is None
