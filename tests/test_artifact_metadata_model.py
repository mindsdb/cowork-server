"""ArtifactMetadata reads metadata.json leniently and writes back the document it read."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cowork.services.artifact_metadata import ArtifactMetadata, MetadataNotAnObject

SURROGATE = "bad" + chr(0xD800)
TEXT_FIELDS = ["slug", "name", "title", "description", "type", "primary"]


def _meta(document: dict) -> ArtifactMetadata:
    return ArtifactMetadata.model_validate(document)


@pytest.mark.parametrize("key", TEXT_FIELDS)
@pytest.mark.parametrize("value", [5, ["html"], {"a": 1}, True, None, SURROGATE])
def test_a_text_field_of_the_wrong_type_reads_as_missing(key, value):
    assert getattr(_meta({key: value}), key) is None


@pytest.mark.parametrize("key", TEXT_FIELDS)
@pytest.mark.parametrize("value", ["Dash", ""])
def test_a_text_field_keeps_its_text(key, value):
    assert getattr(_meta({key: value}), key) == value


@pytest.mark.parametrize("value", ["8080", " 8080 ", 8080.0, True, 0, 65536, -1, None, [8080]])
def test_a_port_that_is_not_an_int_in_range_reads_as_missing(value):
    assert _meta({"port": value}).port is None


@pytest.mark.parametrize("value", [1, 8080, 65535])
def test_a_port_in_range_reads(value):
    assert _meta({"port": value}).port == value


def test_identity_fields_pass_through_unchanged():
    deep = json.loads("[" * 300 + "]" * 300)
    meta = _meta({"id": 5, "stableId": ["x"], "createdAt": deep})

    assert (meta.id, meta.stableId, meta.createdAt) == (5, ["x"], deep)


def test_missing_fields_read_as_none_or_empty():
    meta = _meta({})

    assert (meta.id, meta.slug, meta.port, meta.datasources, meta.provenance) == (None, None, None, [], [])


def test_datasources_keep_only_objects():
    meta = _meta({"datasources": [
        {"engine": "postgres", "name": "prod"}, "postgres-prod", {"engine": 5, "name": "x"},
    ]})

    assert [(d.engine, d.name) for d in meta.datasources] == [("postgres", "prod"), (None, "x")]


@pytest.mark.parametrize("value", ["x", 5, {"engine": "pg"}, None])
def test_datasources_that_are_not_a_list_read_as_empty(value):
    assert _meta({"datasources": value}).datasources == []


def test_provenance_keeps_the_position_of_each_entry():
    meta = _meta({"provenance": ["junk", {"conversation": "c-2"}, {"conversation": 7}]})

    assert meta.provenance[0] is None
    assert meta.provenance[1].conversation == "c-2"
    assert meta.provenance[2].conversation is None


@pytest.mark.parametrize("value", ["c-1", 5, {"conversation": "c-1"}, None])
def test_provenance_that_is_not_a_list_reads_as_empty(value):
    assert _meta({"provenance": value}).provenance == []


@pytest.mark.parametrize("text", ["[]", "5", '"x"', "null", "{"])
def test_parse_refuses_a_document_that_is_not_a_json_object(text):
    with pytest.raises(ValueError):
        ArtifactMetadata.parse(text)


def test_parse_refuses_a_document_nested_too_deeply():
    with pytest.raises(ValueError):
        ArtifactMetadata.parse('{"a": ' + "[" * 100_000 + "]" * 100_000 + "}")


def test_parse_reads_a_document_with_a_lone_surrogate():
    meta = ArtifactMetadata.parse(json.dumps({"name": "x", "files": [{"path": SURROGATE}]}))

    assert meta.name == "x"


def test_the_document_is_kept_as_validated():
    document = {"name": 5, "custom": {"a": 1}, "id": "x"}

    assert _meta(document).to_document() == document


def test_with_value_changes_one_key_and_keeps_order_and_wrong_types():
    meta = ArtifactMetadata.parse(json.dumps({"id": "a", "name": 5, "primary": "x.html", "custom": {"a": 1}}))

    updated = meta.with_value("primary", "y.html")

    assert list(updated.to_document().items()) == [
        ("id", "a"), ("name", 5), ("primary", "y.html"), ("custom", {"a": 1}),
    ]
    assert updated.primary == "y.html"
    assert meta.primary == "x.html"


def test_with_value_appends_a_new_key():
    assert list(_meta({"id": "a"}).with_value("port", 8080).to_document()) == ["id", "port"]


def test_with_id_sets_the_id_in_place_and_drops_stable_id():
    meta = _meta({"id": "a", "stableId": "b", "name": "n"})

    assert list(meta.with_id("c").to_document().items()) == [("id", "c"), ("name", "n")]


def test_a_document_that_is_not_an_object_raises_its_own_error():
    with pytest.raises(MetadataNotAnObject):
        ArtifactMetadata.parse("[]")


def test_the_model_is_not_hashable():
    with pytest.raises(TypeError):
        hash(_meta({}))


def test_to_document_is_a_copy():
    meta = _meta({"id": "a"})
    document = meta.to_document()
    document["x"] = 1

    assert meta.to_document() == {"id": "a"}


def test_empty_is_an_empty_document():
    assert ArtifactMetadata.empty().to_document() == {}


def test_encoded_matches_the_write_format():
    document = {"name": "Dash", "type": "html-app", "note": "ü"}

    assert _meta(document).encoded() == (
        json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def test_encoded_refuses_a_document_that_cannot_be_written_back():
    with pytest.raises(ValueError):
        _meta({"files": [{"path": SURROGATE}]}).encoded()


def test_a_field_cannot_be_assigned():
    meta = _meta({"name": "a"})

    with pytest.raises(ValidationError):
        meta.name = "b"


def test_model_copy_with_an_update_is_refused():
    meta = _meta({"name": "a"})

    with pytest.raises(TypeError):
        meta.model_copy(update={"name": "b"})
    assert meta.model_copy().to_document() == {"name": "a"}
