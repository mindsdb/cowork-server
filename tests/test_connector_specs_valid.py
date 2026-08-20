"""Every connector spec on disk must load, validate, and use known keys.

`ConnectorSpecRegistry._load_all` wraps its `json.loads` in
`except Exception: continue`, so a malformed spec is not an error — the
connector simply ceases to exist, with nothing logged and nothing failing.
That is invisible in review and invisible in CI, which is what these tests
close.

The models do NOT set `extra="forbid"` (pydantic defaults to `extra="ignore"`),
so validation alone cannot catch a misspelled optional key: `hepl_url` silently
drops the help link, `keywrods` silently kills discovery, and the spec still
validates clean. Hence the explicit unknown-key check below.
"""

import json
from pathlib import Path

import pytest

from cowork.schemas.connectors import (
    ConnectorField,
    ConnectorForm,
    ConnectorMethod,
    ConnectorSpecResponse,
)
from cowork.services.connectors.specs._registry import ConnectorSpecRegistry

SPECS_DIR = Path(__file__).parent.parent / "cowork" / "services" / "connectors" / "specs"
SPEC_FILES = sorted(SPECS_DIR.glob("*.json"))

# Keys present in shipped specs that no model declares, so validation drops
# them. Allowed here so this suite goes green on the existing corpus rather
# than blocking on a cleanup — but each one is dead weight, not a feature:
#
#   form.logo_url   (168 specs) — `ConnectorForm` has no `logo_url`, so it is
#                   stripped on validation. `lookup_connector` backfills `logo`
#                   and `logo_color` onto the form blob but not this, so the
#                   credential form falls back to the glyph instead of the
#                   brand mark. The top-level `logo_url` is the one that works.
#   form.engine     (3 specs)
#   method.name_from (2 specs)
#
# Do not add to this list to make a new spec pass. It exists to bound
# pre-existing debt, and a new entry means a key that will be silently ignored.
LEGACY_EXTRA_KEYS = {
    "form": {"logo_url", "engine"},
    "method": {"name_from"},
    "field": set(),
    "spec": set(),
}

KNOWN_KEYS = {
    "spec": set(ConnectorSpecResponse.model_fields) | LEGACY_EXTRA_KEYS["spec"],
    "form": set(ConnectorForm.model_fields) | LEGACY_EXTRA_KEYS["form"],
    "method": set(ConnectorMethod.model_fields) | LEGACY_EXTRA_KEYS["method"],
    "field": set(ConnectorField.model_fields) | LEGACY_EXTRA_KEYS["field"],
}


def test_specs_directory_is_not_empty():
    """Guard the guard: a bad glob would make every test below vacuously pass."""
    assert len(SPEC_FILES) > 100, f"expected the full catalog, found {len(SPEC_FILES)}"


@pytest.mark.parametrize("path", SPEC_FILES, ids=lambda p: p.stem)
def test_spec_is_valid_json_and_matches_the_schema(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path.name}: top level must be an object"
    # The registry fills `id` from the filename when absent; mirror that so a
    # spec relying on it is not failed here for a difference the loader erases.
    data.setdefault("id", path.stem)
    ConnectorSpecResponse(**data)


@pytest.mark.parametrize("path", SPEC_FILES, ids=lambda p: p.stem)
def test_spec_has_no_unknown_keys(path: Path):
    """A typo'd optional key validates clean and is silently dropped."""
    data = json.loads(path.read_text(encoding="utf-8"))
    unknown: list[str] = []

    def check(scope: str, obj: dict, where: str):
        for key in set(obj) - KNOWN_KEYS[scope]:
            unknown.append(f"{where}.{key}")

    check("spec", data, path.stem)
    form = data.get("form") or {}
    check("form", form, f"{path.stem}.form")
    for i, method in enumerate(form.get("methods") or []):
        check("method", method, f"{path.stem}.form.methods[{i}]")
        for j, field in enumerate(method.get("fields") or []):
            check("field", field, f"{path.stem}.form.methods[{i}].fields[{j}]")
    for j, field in enumerate(form.get("fields") or []):
        check("field", field, f"{path.stem}.form.fields[{j}]")

    assert not unknown, (
        f"unknown keys will be silently ignored by the schema: {unknown}. "
        "Fix the spelling, or add the field to the model in cowork/schemas/connectors.py."
    )


@pytest.mark.parametrize("path", SPEC_FILES, ids=lambda p: p.stem)
def test_spec_id_matches_its_filename(path: Path):
    """`id` and the filename must agree, and it is load-bearing in two places.

    The registry keys connectors by the `id` field while every consumer reaches
    for them by slug, and `FormLogo` derives its brand mark from
    `logos/{connector_id}.svg` (DataVaultForm.jsx, ENG-1534) — so a spec whose
    id drifts from its filename would silently lose its logo.

    Asserted separately from the load test below so a divergence reports itself
    as a mismatch rather than as a phantom "the registry skipped this file".
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("id", path.stem) == path.stem, (
        f"{path.name} declares id={data.get('id')!r}; the registry keys on the id "
        "while consumers use the filename slug."
    )


def test_registry_loads_every_spec_file():
    """The direct guard on the silent skip.

    `_load_all` swallows a parse failure, so a broken spec shows up only as a
    connector that is not there. Comparing counts is what turns that into a
    red test instead of a missing feature.

    Keyed on the spec's own `id` (which the registry uses), not the filename —
    `test_spec_id_matches_its_filename` above is what ties the two together, so
    a failure here means a genuine load failure rather than a naming drift.
    """
    loaded = ConnectorSpecRegistry(SPECS_DIR).get_connectors()
    declared = {
        json.loads(p.read_text(encoding="utf-8")).get("id", p.stem) for p in SPEC_FILES
    }
    missing = declared - set(loaded)
    assert not missing, f"specs on disk that the registry silently skipped: {sorted(missing)}"
    assert len(loaded) == len(SPEC_FILES)


class TestLangfuseSpec:
    """The Langfuse spec's credential shape, which a later edit could quietly break."""

    @pytest.fixture
    def spec(self):
        s = ConnectorSpecRegistry(SPECS_DIR).get_connector("langfuse")
        assert s is not None, "langfuse spec did not load"
        return s

    def test_public_key_is_not_stored_as_a_secret(self, spec):
        """It is the Basic-auth username and Langfuse ships it in browser SDKs.

        Marking it secret would make it unreadable for support without
        protecting anything.
        """
        field = self._field(spec, "public_key")
        assert field.secret is False
        assert field.type == "text"

    def test_secret_key_is_stored_as_a_secret(self, spec):
        field = self._field(spec, "secret_key")
        assert field.secret is True
        assert field.type == "password"

    def test_host_is_free_text_with_a_cloud_default(self, spec):
        """Self-hosting is a normal Langfuse deployment, so this cannot be a
        fixed region `select`."""
        field = self._field(spec, "base_url")
        assert field.type == "url"
        assert field.options is None
        assert field.default == "https://cloud.langfuse.com"
        assert field.required is False

    @pytest.mark.parametrize(
        "query", ["langfuse", "langfuse.com", "langfuse cloud", "llm observability"]
    )
    def test_is_discoverable_by_name_and_alias(self, query):
        result = ConnectorSpecRegistry(SPECS_DIR).match_connector(query)
        assert result.candidates, f"no candidate for {query!r}"
        assert result.candidates[0].id == "langfuse"

    @staticmethod
    def _field(spec, name):
        method = next(m for m in spec.form.methods if m.id == "api-key")
        return next(f for f in method.fields if f.name == name)
