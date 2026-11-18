from __future__ import annotations

import pytest
from pydantic import ValidationError

from cowork.api.v1.endpoints.coding_personal_skills import PersonalSkillImport


@pytest.mark.parametrize(("name", "newline"), [
    pytest.param("123", "\n", id="integer-lf"),
    pytest.param("1.5", "\n", id="float-lf"),
    pytest.param("[example]", "\n", id="sequence-lf"),
    pytest.param("{example: value}", "\n", id="mapping-lf"),
    pytest.param("true", "\n", id="boolean-lf"),
    pytest.param("null", "\n", id="null-lf"),
    pytest.param("0", "\n", id="zero-lf"),
    pytest.param("[]", "\n", id="empty-sequence-lf"),
    pytest.param("{}", "\n", id="empty-mapping-lf"),
    pytest.param("2026-09-15", "\n", id="date-lf"),
    pytest.param("123", "\r\n", id="integer-crlf"),
    pytest.param("123", "\r", id="integer-cr"),
])
def test_import_model_rejects_non_text_yaml_name(name: str, newline: str) -> None:
    """Reject raw YAML name types before the canonical parser normalizes them."""
    source = f"---\nname: {name}\ndescription: Example\n---\nInstructions"
    with pytest.raises(ValidationError) as error:
        PersonalSkillImport(content=source.replace("\n", newline))
    details = error.value.errors()
    assert len(details) == 1
    assert details[0]["loc"] == ("content",)
    assert details[0]["type"] == "value_error"
    assert "name" in details[0]["msg"].lower()
    assert "text" in details[0]["msg"].lower()


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
def test_import_model_preserves_quoted_numeric_name(newline: str) -> None:
    """Accept a quoted numeric name without rewriting the supplied content."""
    source = '---\nname: "123"\ndescription: Example\n---\nInstructions'.replace("\n", newline)
    assert PersonalSkillImport(content=source).content == source
