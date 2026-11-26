"""Which connections' credentials a full-stack publish may send."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cowork.db.personal_resources import ALL_PERSONAL_RESOURCES, PersonalResource
from cowork.db.scoped import LOCAL_SCOPE, TenantScope
from cowork.db.units import unit_session
from cowork.models.artifact_datasource_approval import ArtifactDatasourceApproval
from cowork.services.connectors.layered_vault import LABEL_FIELDS
from cowork.services import datasource_approval as da
from cowork.services.datasource_approval import (
    AllowedDataVault,
    DatasourceKey,
    datasource_labels,
    declared_datasources,
    record_approval,
    stored_approvals,
)
from tests._artifact_metadata import write_artifact_metadata

ORG_SCOPE = TenantScope(org_mode=True, org_id="org-1", user_id="user-1")
PERSON_A = TenantScope(
    org_id="partition-a", user_id="person-a", personal=True, personal_resources=ALL_PERSONAL_RESOURCES,
)
PERSON_B = TenantScope(
    org_id="partition-b", user_id="person-b", personal=True, personal_resources=ALL_PERSONAL_RESOURCES,
)
SHARED_CONNECTIONS = TenantScope(
    org_id="partition-a", user_id="person-a", personal=True,
    personal_resources=ALL_PERSONAL_RESOURCES - {PersonalResource.CONNECTIONS},
)
ALL_SCOPES = (LOCAL_SCOPE, ORG_SCOPE, PERSON_A, PERSON_B)


def _clear(scope: TenantScope) -> None:
    with unit_session(scope=scope) as session:
        for row in session.exec(session.select(ArtifactDatasourceApproval)):
            session.delete(row)


@pytest.fixture(autouse=True)
def _no_approvals():
    for scope in ALL_SCOPES:
        _clear(scope)
    yield
    for scope in ALL_SCOPES:
        _clear(scope)


def _folder(tmp_path: Path, datasources: list[dict], name: str = "app") -> Path:
    return write_artifact_metadata(
        tmp_path / name, slug=name, artifact_type="fullstack-stateless-app", datasources=datasources
    )


class FakeVault:
    def __init__(self, records: dict[tuple[str, str], dict]):
        self.records = records

    def read_record(self, engine, name):
        fields = self.records.get((engine, name))
        return None if fields is None else {"engine": engine, "name": name, "fields": fields}

    def env_for(self, engine, name, *, flat=False):
        fields = self.records.get((engine, name))
        return None if fields is None else {f"DS_{engine}_{name}__PASSWORD".upper(): fields.get("password", "")}

    def load(self, engine, name):
        return self.records.get((engine, name))


def test_datasource_key_keeps_the_exact_spelling():
    """anton's vault keys a record by its exact engine and name."""
    assert DatasourceKey("Postgres", "SCM") == ("Postgres", "SCM")
    assert DatasourceKey("postgres", "Prod") != DatasourceKey("postgres", "prod")


def test_declared_datasources_keep_the_names_as_written(tmp_path):
    folder = _folder(tmp_path, [{"engine": "Postgres", "name": "SCM"}])
    assert declared_datasources(folder) == [DatasourceKey("Postgres", "SCM")]


def test_two_spellings_are_two_declared_connections(tmp_path):
    folder = _folder(tmp_path, [{"engine": "postgres", "name": "prod"}, {"engine": "postgres", "name": "Prod"}])
    assert declared_datasources(folder) == [("postgres", "prod"), ("postgres", "Prod")]


def test_the_helper_writes_metadata_anton_accepts(tmp_path):
    from anton.publisher import _load_artifact_metadata

    folder = _folder(tmp_path, [{"engine": "pg", "name": "a"}])
    assert _load_artifact_metadata(folder) is not None


def test_declared_datasources_are_empty_without_valid_metadata(tmp_path):
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "metadata.json").write_text("{not json")
    assert declared_datasources(folder) == []
    assert declared_datasources(tmp_path / "missing") == []


def test_metadata_anton_rejects_declares_nothing(tmp_path):
    """anton publishes such a folder as HTML with no credentials, so there is
    nothing to approve. The parse must stay as strict as anton's."""
    folder = tmp_path / "partial"
    folder.mkdir()
    (folder / "metadata.json").write_text(json.dumps({
        "slug": "partial", "name": "partial", "type": "fullstack-stateless-app",
        "datasources": [{"engine": "pg", "name": "a"}],
    }))
    assert declared_datasources(folder) == []


def test_nothing_is_approved_without_a_row(tmp_path):
    folder = _folder(tmp_path, [])
    assert stored_approvals(artifact_key="artifact/1", artifact_dir=folder, scope=None) == frozenset()


def test_an_approval_adds_to_what_was_approved(tmp_path):
    folder = _folder(tmp_path, [])
    record_approval(artifact_key="artifact/1", artifact_dir=folder, datasources={DatasourceKey("postgres", "a")}, scope=None)
    record_approval(artifact_key="artifact/1", artifact_dir=folder, datasources={DatasourceKey("MySQL", "B")}, scope=None)
    assert stored_approvals(artifact_key="artifact/1", artifact_dir=folder, scope=None) == {
        ("postgres", "a"), ("MySQL", "B"),
    }


def test_an_approval_holds_only_for_its_folder(tmp_path):
    first, second = _folder(tmp_path, [], "first"), _folder(tmp_path, [], "second")
    record_approval(artifact_key="artifact/1", artifact_dir=first, datasources={DatasourceKey("pg", "a")}, scope=None)
    assert stored_approvals(artifact_key="artifact/1", artifact_dir=second, scope=None) == frozenset()


def test_an_org_approval_is_its_orgs_and_names_who_approved(tmp_path):
    folder = _folder(tmp_path, [])
    record_approval(artifact_key="artifact/1", artifact_dir=folder, datasources={DatasourceKey("pg", "a")}, scope=ORG_SCOPE)
    assert stored_approvals(artifact_key="artifact/1", artifact_dir=folder, scope=None) == frozenset()
    assert stored_approvals(artifact_key="artifact/1", artifact_dir=folder, scope=ORG_SCOPE) == {("pg", "a")}
    with unit_session(scope=ORG_SCOPE) as session:
        row = session.exec(session.select(ArtifactDatasourceApproval)).one()
        assert (row.org_id, row.approved_by) == ("org-1", "user-1")


def test_a_personal_approval_is_partitioned_when_connections_are(tmp_path):
    folder = _folder(tmp_path, [])
    record_approval(artifact_key="artifact/1", artifact_dir=folder, datasources={DatasourceKey("pg", "a")}, scope=PERSON_A)
    assert stored_approvals(artifact_key="artifact/1", artifact_dir=folder, scope=PERSON_A) == {("pg", "a")}
    assert stored_approvals(artifact_key="artifact/1", artifact_dir=folder, scope=PERSON_B) == frozenset()
    assert stored_approvals(artifact_key="artifact/1", artifact_dir=folder, scope=None) == frozenset()
    with unit_session(scope=PERSON_A) as session:
        assert session.exec(session.select(ArtifactDatasourceApproval)).one().org_id == "partition-a"


def test_a_personal_approval_is_deployment_wide_when_connections_are_shared(tmp_path):
    folder = _folder(tmp_path, [])
    record_approval(artifact_key="artifact/1", artifact_dir=folder, datasources={DatasourceKey("pg", "org")}, scope=ORG_SCOPE)
    assert stored_approvals(artifact_key="artifact/1", artifact_dir=folder, scope=SHARED_CONNECTIONS) == frozenset()

    record_approval(artifact_key="artifact/1", artifact_dir=folder, datasources={DatasourceKey("pg", "a")}, scope=SHARED_CONNECTIONS)
    assert stored_approvals(artifact_key="artifact/1", artifact_dir=folder, scope=SHARED_CONNECTIONS) == {("pg", "a")}
    assert stored_approvals(artifact_key="artifact/1", artifact_dir=folder, scope=None) == {("pg", "a")}
    assert stored_approvals(artifact_key="artifact/1", artifact_dir=folder, scope=ORG_SCOPE) == {("pg", "org")}
    with unit_session(scope=LOCAL_SCOPE) as session:
        rows = session.exec(session.select(ArtifactDatasourceApproval).where(ArtifactDatasourceApproval.org_id.is_(None))).all()
        assert len(rows) == 1


def test_the_label_order_names_every_label_field():
    assert set(da._LABEL_ORDER) == LABEL_FIELDS


def test_labels_show_the_label_or_name_and_the_engine():
    vault = FakeVault({("postgres", "scm"): {"_user_label": "Demo data", "password": "secret"}, ("mysql", "raw"): {}})
    assert datasource_labels(vault, [("postgres", "scm"), ("mysql", "raw")]) == [
        {"engine": "postgres", "name": "scm", "label": "Demo data (postgres)"},
        {"engine": "mysql", "name": "raw", "label": "raw (mysql)"},
    ]


def test_an_agent_set_label_cannot_rewrite_the_card():
    label = 'Demo data". Nothing sensitive is uploaded;\nthis only confirms the link. "' + "x" * 80
    vault = FakeVault({("postgres", "prod"): {"_user_label": label}, ("mysql", "raw"): {"_user_label": '"\n"'}})
    labels = datasource_labels(vault, [("postgres", "prod"), ("mysql", "raw")])
    shown = labels[0]["label"]
    assert shown.endswith(" (postgres)")
    assert '"' not in shown and "\n" not in shown and ";" not in shown
    assert len(shown) <= 60 + len(" (postgres)")
    # A label with nothing left shows the name instead.
    assert labels[1]["label"] == "raw (mysql)"


def test_a_vault_that_fails_still_gets_name_and_engine():
    class Broken:
        def read_record(self, engine, name):
            raise OSError("unreadable")

    assert datasource_labels(Broken(), [("pg", "a")]) == [{"engine": "pg", "name": "a", "label": "a (pg)"}]


def test_the_allowed_vault_reads_only_approved_connections():
    source = FakeVault({("Postgres", "SCM"): {"password": "a"}, ("mysql", "late"): {"password": "b"}})
    vault = AllowedDataVault(source=source, allowed={DatasourceKey("Postgres", "SCM")})
    assert vault.env_for("Postgres", "SCM") == {"DS_POSTGRES_SCM__PASSWORD": "a"}
    assert vault.env_for("mysql", "late") is None
    assert vault.read_record("mysql", "late") is None
    assert vault.load("mysql", "late") is None
    assert vault.refused == [DatasourceKey("mysql", "late")]


def test_approving_one_spelling_lets_no_other_through_the_allowed_vault():
    source = FakeVault({("postgres", "prod"): {"password": "a"}, ("postgres", "Prod"): {"password": "b"}})
    vault = AllowedDataVault(source=source, allowed={DatasourceKey("postgres", "prod")})
    assert vault.env_for("postgres", "prod") == {"DS_POSTGRES_PROD__PASSWORD": "a"}
    assert vault.env_for("postgres", "Prod") is None
    assert vault.read_record("postgres", "Prod") is None
    assert vault.load("postgres", "Prod") is None
    assert vault.refused == [DatasourceKey("postgres", "Prod")]


def test_a_connection_that_is_not_saved_is_not_reported_as_refused():
    vault = AllowedDataVault(source=FakeVault({}), allowed=set())
    assert vault.env_for("pg", "ghost") is None
    assert vault.refused == []
