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
    CloudMethod,
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
#                   stripped on validation and nothing reads it. Harmless
#                   rather than broken: `FormLogo` derives the brand mark from
#                   `logos/{connector_id}.svg` when the form blob carries no
#                   url (DataVaultForm.jsx, ENG-1534), and the picker and
#                   connection panel read the TOP-LEVEL `logo_url`, which
#                   survives. Tidying it would be a no-op, not a fix.
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
    "cloud": set(),
}

KNOWN_KEYS = {
    "spec": set(ConnectorSpecResponse.model_fields) | LEGACY_EXTRA_KEYS["spec"],
    "form": set(ConnectorForm.model_fields) | LEGACY_EXTRA_KEYS["form"],
    "method": set(ConnectorMethod.model_fields) | LEGACY_EXTRA_KEYS["method"],
    "field": set(ConnectorField.model_fields) | LEGACY_EXTRA_KEYS["field"],
    "cloud": set(CloudMethod.model_fields) | LEGACY_EXTRA_KEYS["cloud"],
}


# LEGACY_EXTRA_KEYS bounds key *names*, not the number of specs using them, so
# on its own it is a request for restraint rather than a gate — a new spec can
# adopt a dead key and stay green. This baseline makes the two genuinely dead
# ones a tripwire.
#
# `form.logo_url` is deliberately NOT here. It is inert but it is also what 169
# of 213 specs do, so it is the convention; failing a new connector for
# following it would be backwards. The other two have no known consumer, so a
# fourth `form.engine` is almost certainly a mistake worth catching.
DEAD_KEY_BASELINE = {"form.engine": 3, "method.name_from": 2}


def _count_dead_keys() -> dict[str, int]:
    counts = {k: 0 for k in DEAD_KEY_BASELINE}
    for path in SPEC_FILES:
        data = json.loads(path.read_text(encoding="utf-8"))
        form = data.get("form") or {}
        if "engine" in form:
            counts["form.engine"] += 1
        for method in form.get("methods") or []:
            if "name_from" in method:
                counts["method.name_from"] += 1
    return counts


def test_specs_directory_is_not_empty():
    """Guard the guard: a bad glob would make every test below vacuously pass."""
    assert len(SPEC_FILES) > 100, f"expected the full catalog, found {len(SPEC_FILES)}"


def test_dead_keys_are_not_spreading():
    """A key nothing reads should not gain new users.

    Equality, not `<=`: a cleanup that removes one should lower the baseline
    deliberately rather than drift past an inequality unnoticed.
    """
    assert _count_dead_keys() == DEAD_KEY_BASELINE, (
        "the count of specs carrying a key no model declares has changed. "
        "If you added one: the schema will silently ignore it — either drop it "
        "or declare the field in cowork/schemas/connectors.py. If you removed "
        "one: lower the baseline in DEAD_KEY_BASELINE."
    )


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
        cloud = method.get("cloud") or {}
        check("cloud", cloud, f"{path.stem}.form.methods[{i}].cloud")
        for j, field in enumerate(cloud.get("fields") or []):
            check("field", field, f"{path.stem}.form.methods[{i}].cloud.fields[{j}]")
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
        "query", ["langfuse", "langfuse.com", "langfuse cloud"]
    )
    def test_is_discoverable_by_name_and_alias(self, query):
        result = ConnectorSpecRegistry(SPECS_DIR).match_connector(query)
        assert result.candidates, f"no candidate for {query!r}"
        assert result.candidates[0].id == "langfuse"

    @staticmethod
    def _field(spec, name):
        method = next(m for m in spec.form.methods if m.id == "api-key")
        return next(f for f in method.fields if f.name == name)


# The two connectors ENG-2806 enables for cloud execution. Every other method
# in the corpus carries no `cloud` block and is therefore desktop-only.
CLOUD_DATABASE_METHODS = {"postgres": "host-port", "mysql": "host-password"}

# The field names the desktop forms render today. Pinned because the cloud
# block is a second, independent list: if a cloud edit ever reaches the desktop
# list, this is the test that says so.
DESKTOP_FIELDS = {
    "postgres": {
        "connection-string": ["connection_uri"],
        "host-port": ["host", "port", "database", "username", "password", "ssl_enabled"],
    },
    "mysql": {
        "host-password": [
            "host",
            "port",
            "database",
            "username",
            "password",
            "use_ssl",
            "ssl_ca_cert",
        ],
        "connection-string": ["connection_string", "ssl_ca_cert"],
    },
}


class TestCloudDatabaseSpecs:
    """The cloud blocks on postgres and mysql.

    They carry TLS constraints the hosted path enforces and the desktop path
    does not, so an edit that collapses the two forms back together is the
    failure this class exists to catch.
    """

    @pytest.fixture(
        params=sorted(CLOUD_DATABASE_METHODS), ids=sorted(CLOUD_DATABASE_METHODS)
    )
    def connector_id(self, request):
        return request.param

    @pytest.fixture
    def spec(self, connector_id):
        s = ConnectorSpecRegistry(SPECS_DIR).get_connector(connector_id)
        assert s is not None, f"{connector_id} spec did not load"
        return s

    def test_only_the_password_method_declares_cloud_support(self, spec, connector_id):
        declared = [m.id for m in spec.form.methods if m.cloud is not None]
        assert declared == [CLOUD_DATABASE_METHODS[connector_id]]

    def test_the_connection_string_method_stays_desktop_only(self, spec):
        """A raw DSN is never persisted on the hosted path, so the method that
        collects one has no cloud form to submit."""
        method = next(m for m in spec.form.methods if m.id == "connection-string")
        assert method.cloud is None

    def test_cloud_is_not_advertised_before_the_adapters_are_proven(
        self, spec, connector_id
    ):
        assert self._cloud(spec, connector_id).available is False

    def test_certificate_trust_offers_no_downgrade(self, spec, connector_id):
        field = self._cloud_field(spec, connector_id, "tls_mode")
        assert field.type == "select"
        assert field.required is True
        assert field.default == "system"
        assert [o["value"] for o in field.options] == ["system", "custom_ca"]

    def test_the_ca_is_pasted_content_and_bounded(self, spec, connector_id):
        """A path or a URL would let the form choose what the gateway trusts."""
        field = self._cloud_field(spec, connector_id, "ca_pem")
        assert field.type == "textarea"
        assert field.required is False
        assert field.secret is False
        assert "64 KiB" in field.description
        assert "path or URL is not accepted" in field.description

    def test_no_cloud_field_toggles_tls(self, spec, connector_id):
        """`ssl_enabled` and `use_ssl` are desktop fields. A boolean here would
        be a cloud form that can ask for an unverified connection."""
        names = {f.name for f in self._cloud(spec, connector_id).fields}
        assert names.isdisjoint({"ssl_enabled", "use_ssl", "ssl", "tls", "ssl_ca_cert"})

    @pytest.mark.parametrize("phrase", ["sslmode=disable", "leave SSL off"])
    def test_cloud_copy_never_inherits_the_desktop_ssl_off_guidance(
        self, spec, connector_id, phrase
    ):
        cloud = self._cloud(spec, connector_id)
        copy = " ".join(
            [cloud.description or "", cloud.how_to or ""]
            + [f.description or "" for f in cloud.fields]
        )
        assert phrase.lower() not in copy.lower()

    def test_desktop_fields_are_untouched(self, spec, connector_id):
        for method in spec.form.methods:
            expected = DESKTOP_FIELDS[connector_id][method.id]
            assert [f.name for f in method.fields] == expected

    @staticmethod
    def _cloud(spec, connector_id):
        wanted = CLOUD_DATABASE_METHODS[connector_id]
        method = next(m for m in spec.form.methods if m.id == wanted)
        assert method.cloud is not None
        return method.cloud

    @classmethod
    def _cloud_field(cls, spec, connector_id, name):
        return next(f for f in cls._cloud(spec, connector_id).fields if f.name == name)


class TestMySQLCloudProducts:
    """MySQL compatibility is claimed by product, not by the connector's name."""

    @pytest.fixture
    def method(self):
        spec = ConnectorSpecRegistry(SPECS_DIR).get_connector("mysql")
        assert spec is not None, "mysql spec did not load"
        return next(m for m in spec.form.methods if m.id == "host-password")

    def test_cloud_copy_names_the_products_it_accepts(self, method):
        """The spec's own aliases and description cover MariaDB and Percona for
        desktop discovery, so the cloud block has to say they are refused."""
        copy = f"{method.cloud.description} {method.cloud.how_to}"
        assert "Oracle MySQL 8.0 and 8.4" in copy
        assert "MariaDB" in copy
        assert "Percona" in copy
        assert "refused" in copy

    def test_desktop_copy_makes_no_claim_about_the_hosted_path(self, method):
        """It previously promised the password is never transmitted to Anton's
        servers, which the cloud relay makes false."""
        assert "never transmits it to Anton's servers" not in (method.how_to or "")
