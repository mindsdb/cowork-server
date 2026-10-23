"""The operator sees ownership and stale storage, with no credential output or writes."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from anton.core.datasources.data_vault import LocalDataVault

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE
from cowork.ops import Person
from cowork.ops import connections as connections_ops
from cowork.ops.connection_inventory import MAX_RECORD_BYTES
from cowork.services.connectors.persist import own_vault_for_scope, vault_for_scope
from tests.test_ops_commands import ALICE, BOB, SUB_A, SUB_B, _arguments, _run, _tree, personal_scope, stack  # noqa: F401

SECRET = "inventory-secret-must-never-appear"


def _save(person, name="warehouse"):
    return vault_for_scope(personal_scope(person)).save("postgres", name, {"PASSWORD": SECRET, "HOST": "private-host"})


def _list(capsys, person=None):
    args = [] if person is None else _arguments(person)
    status, out, err = _run(connections_ops.main, ["list", *args], capsys)
    assert status == 0, err
    assert SECRET not in out + err and "private-host" not in out + err
    assert "fields" not in out and "PASSWORD" not in out
    return json.loads(out)


def test_inventory_lists_private_shared_managed_legacy_and_missing_entries_without_writes(stack, tmp_path, capsys):
    _save(ALICE)
    _save(BOB)
    managed = LocalDataVault(stack.vault_dir)
    managed.save("postgres", "team", {"PASSWORD": SECRET})
    legacy = LocalDataVault(tmp_path / "home" / "data-vault")
    legacy.save("postgres", "old-team", {"PASSWORD": SECRET})
    registry = stack.shared_connections_file
    registry.parent.mkdir(exist_ok=True)
    registry.write_text(json.dumps([
        {"user_id": SUB_A, "partition_id": ALICE.partition_id, "engine": "postgres", "name": "warehouse"},
        {"user_id": SUB_A, "partition_id": ALICE.partition_id, "engine": "postgres", "name": "gone"},
    ]))
    before = _tree(tmp_path)
    result = _list(capsys)
    assert result["personal_connections"] is True
    rows = {(r["origin"], r["partition_id"], r["name"]): r for r in result["connections"]}
    assert rows["personal", ALICE.partition_id, "warehouse"]["visibility"] == "shared"
    assert rows["personal", BOB.partition_id, "warehouse"]["visibility"] == "private"
    assert rows["personal", BOB.partition_id, "warehouse"]["user_id"] == SUB_B
    assert rows["managed", None, "team"]["visibility"] == "shared"
    assert rows["base", None, "old-team"]["visibility"] == "inactive"
    assert rows["registry", ALICE.partition_id, "gone"]["visibility"] == "stale"
    assert _tree(tmp_path) == before
    selected = _list(capsys, BOB)["connections"]
    assert {r["partition_id"] for r in selected} == {None, BOB.partition_id}


def test_inventory_retains_unstamped_orphan_spaces_and_marks_inactive_personal_stores(stack, tmp_path, monkeypatch, capsys):
    own_vault_for_scope(personal_scope(ALICE)).save("postgres", "old", {"PASSWORD": SECRET})
    assert _list(capsys)["connections"][0]["user_id"] is None
    monkeypatch.setenv("COWORK_PERSONAL_RESOURCES", "conversations,projects,files,pins,schedules")
    get_app_settings.cache_clear()
    LocalDataVault(tmp_path / "home" / "data-vault").save("postgres", "team", {"PASSWORD": SECRET})
    result = _list(capsys)
    assert result["personal_connections"] is False
    assert [(r["origin"], r["visibility"]) for r in result["connections"]] == [("base", "shared"), ("personal", "inactive")]


@pytest.mark.parametrize(("enforce", "visibility"), [("enforce", "inactive"), ("audit", "operator")])
def test_base_records_stay_the_operator_views_while_audit_serves_it(stack, tmp_path, monkeypatch, capsys, enforce, visibility):
    # A request with no identity pair is the operator view on LOCAL_SCOPE, which
    # reads the base vault; only audit lets such a request through.
    monkeypatch.setenv("COWORK_IDENTITY_ENFORCE", enforce)
    get_app_settings.cache_clear()
    LocalDataVault(tmp_path / "home" / "data-vault").save("postgres", "old-team", {"PASSWORD": SECRET})
    assert [(row["engine"], row["name"]) for row in vault_for_scope(LOCAL_SCOPE).list_connections()] == [("postgres", "old-team")]
    [row] = _list(capsys)["connections"]
    assert (row["origin"], row["visibility"]) == ("base", visibility)
    # A managed connection of the same DS_ prefix hides it from the operator view.
    LocalDataVault(stack.vault_dir).save("postgres", "old_team", {"PASSWORD": SECRET})
    [row] = [row for row in _list(capsys)["connections"] if row["origin"] == "base"]
    assert row["notes"] == (["hidden-by-managed-prefix"] if visibility == "operator" else [])


def test_inventory_marks_managed_prefix_override_and_same_vault_collision(stack, capsys):
    _save(ALICE, "team-db")
    _save(ALICE, "team_db")
    LocalDataVault(stack.vault_dir).save("postgres", "TEAM_DB", {"PASSWORD": SECRET})
    rows = [r for r in _list(capsys)["connections"] if r["origin"] == "personal"]
    assert len(rows) == 2
    assert all(r["notes"] == ["hidden-by-managed-prefix", "same-vault-prefix-collision"] for r in rows)


def test_inventory_reads_pinned_vault_after_directory_replacement(stack, tmp_path, monkeypatch, capsys):
    from cowork.ops import connection_inventory as inventory

    _save(ALICE)
    vault = tmp_path / "shared" / ALICE.partition_id / "data-vault"
    replacement = tmp_path / "replacement"
    LocalDataVault(replacement).save("postgres", "foreign", {"password": SECRET})
    real_open = inventory.dir_open
    replaced = False

    def replace_before_open(d, name, flags):
        nonlocal replaced
        if d.path == vault and not replaced:
            vault.rename(tmp_path / "original-vault")
            vault.symlink_to(replacement)
            replaced = True
        return real_open(d, name, flags)

    monkeypatch.setattr(inventory, "dir_open", replace_before_open)
    rows = _list(capsys)["connections"]
    assert replaced
    assert [(row["name"], row["user_id"]) for row in rows] == [("warehouse", SUB_A)]


def test_duplicate_shares_report_the_registry_precedence(stack, capsys):
    _save(ALICE)
    _save(BOB)
    stack.shared_dir.mkdir(exist_ok=True)
    stack.shared_connections_file.write_text(json.dumps([
        {"user_id": person.user_id, "partition_id": person.partition_id, "engine": "postgres", "name": "warehouse"}
        for person in (ALICE, BOB)
    ]))
    rows = {row["user_id"]: row for row in _list(capsys)["connections"]}
    assert rows[SUB_A]["notes"] == ["recipient-own-prefix-takes-priority"]
    assert rows[SUB_B]["notes"] == ["recipient-own-prefix-takes-priority", "earlier-share-takes-priority-for-other-recipients"]


def test_inventory_matches_a_live_share_by_vault_filename_after_an_alias_update(stack, tmp_path, capsys):
    _save(ALICE, "sales.db")
    status, _, err = _run(connections_ops.main, [
        "share", *_arguments(ALICE), "--name", "sales.db", "--managed-dir", str(stack.root),
    ], capsys)
    assert status == 0, err
    _save(ALICE, "sales_db")

    recipient = vault_for_scope(personal_scope(BOB))
    assert [(row["engine"], row["name"]) for row in recipient.list_connections()] == [("postgres", "sales.db")]
    assert recipient.load("postgres", "sales.db")["PASSWORD"] == SECRET
    before = _tree(tmp_path)
    rows = _list(capsys)["connections"]
    assert len(rows) == 1
    assert (rows[0]["name"], rows[0]["visibility"]) == ("sales_db", "shared")
    assert rows[0]["shared_as"] == [{"engine": "postgres", "name": "sales.db"}]
    assert "registry-name-alias" in rows[0]["notes"]
    assert _tree(tmp_path) == before

    # A genuinely missing canonical file still leaves a stale registry entry.
    own_vault_for_scope(personal_scope(ALICE)).delete("postgres", "sales_db")
    assert recipient.read_record("postgres", "sales.db") is None
    assert [(row["name"], row["visibility"]) for row in _list(capsys)["connections"]] == [("sales.db", "stale")]


def test_alias_shares_keep_registry_names_for_precedence_and_prefix_collisions(stack, capsys):
    _save(ALICE, "sales_db")
    _save(BOB, "sales_db")
    # Both aliases reach Alice's one file. Bob's duplicate registry name stays
    # behind Alice's, while two different aliases share one DS_ prefix.
    stack.shared_dir.mkdir(exist_ok=True)
    stack.shared_connections_file.write_text(json.dumps([
        {"user_id": person.user_id, "partition_id": person.partition_id, "engine": "postgres", "name": name}
        for person, name in ((ALICE, "sales.db"), (ALICE, "sales/db"), (BOB, "sales.db"))
    ]))
    rows = {row["user_id"]: row for row in _list(capsys)["connections"]}
    assert len(rows) == 2 and all(row["visibility"] == "shared" for row in rows.values())
    assert rows[SUB_A]["shared_as"] == [
        {"engine": "postgres", "name": "sales.db"}, {"engine": "postgres", "name": "sales/db"},
    ]
    assert all("shared-prefix-collision-for-other-recipients" in row["notes"] for row in rows.values())
    assert "earlier-share-takes-priority-for-other-recipients" not in rows[SUB_A]["notes"]
    assert "earlier-share-takes-priority-for-other-recipients" in rows[SUB_B]["notes"]


def test_inventory_reports_a_managed_override_of_the_shared_alias_prefix(stack, capsys):
    _save(ALICE, "sales")
    LocalDataVault(stack.vault_dir).save("postgres", "sales_", {"PASSWORD": SECRET})
    stack.shared_dir.mkdir(exist_ok=True)
    stack.shared_connections_file.write_text(json.dumps([
        {"user_id": SUB_A, "partition_id": ALICE.partition_id, "engine": "postgres", "name": "sales."},
    ]))
    assert [row["name"] for row in vault_for_scope(personal_scope(BOB)).list_connections()] == ["sales_"]
    [row] = [row for row in _list(capsys)["connections"] if row["origin"] == "personal"]
    assert row["visibility"] == "shared"
    assert "hidden-by-managed-prefix" not in row["notes"]
    assert "shared-alias-hidden-by-managed-prefix" in row["notes"]


def test_distinct_shared_names_report_prefix_collisions_across_partitions(stack, capsys):
    entries = [(ALICE, "team-db"), (BOB, "team_db")]
    for person, name in entries:
        _save(person, name)
    stack.shared_dir.mkdir(exist_ok=True)
    stack.shared_connections_file.write_text(json.dumps([
        {"user_id": person.user_id, "partition_id": person.partition_id, "engine": "postgres", "name": name}
        for person, name in entries
    ]))
    rows = _list(capsys)["connections"]
    assert len(rows) == 2
    assert all("shared-prefix-collision-for-other-recipients" in row["notes"] for row in rows)


@pytest.mark.parametrize("kind", ["invalid-json", "hardlink", "linked-partition"])
def test_one_persons_listing_opens_no_other_persons_records(stack, tmp_path, monkeypatch, capsys, kind):
    from cowork.ops import connection_inventory as inventory

    _save(ALICE)
    path = Path(_save(BOB))
    if kind == "invalid-json":
        path.write_text(SECRET)
    elif kind == "hardlink":
        os.link(path, tmp_path / "second-name")
    else:
        partition = tmp_path / "shared" / BOB.partition_id
        partition.rename(tmp_path / "elsewhere")
        partition.symlink_to(tmp_path / "elsewhere")
    opened = []
    real_read = inventory._read

    def recording_read(d, name, *, limit):
        opened.append(d.path)
        return real_read(d, name, limit=limit)

    monkeypatch.setattr(inventory, "_read", recording_read)
    rows = _list(capsys, ALICE)["connections"]
    assert [(row["partition_id"], row["name"]) for row in rows] == [(ALICE.partition_id, "warehouse")]
    assert opened and not any(BOB.partition_id in str(folder) for folder in opened)
    # The full inventory still refuses what it can't read safely.
    status, out, err = _run(connections_ops.main, ["list"], capsys)
    assert status == 1 and out == "" and SECRET not in err


@pytest.mark.parametrize("kind", ["hardlink", "symlink"])
def test_a_planted_link_leaves_that_partitions_shares_out_of_another_persons_notes(stack, tmp_path, capsys, kind):
    # Runtime drops every share of a vault holding a link (refuse_planted_links),
    # so a third person reads Bob's warehouse, not Alice's earlier one.
    _save(ALICE)
    vault_for_scope(personal_scope(BOB)).save("postgres", "warehouse", {"PASSWORD": "bobs-password"})
    stack.shared_dir.mkdir(exist_ok=True)
    stack.shared_connections_file.write_text(json.dumps([
        {"user_id": person.user_id, "partition_id": person.partition_id, "engine": "postgres", "name": "warehouse"}
        for person in (ALICE, BOB)
    ]))
    alices_vault = tmp_path / "shared" / ALICE.partition_id / "data-vault"
    if kind == "hardlink":
        os.link(alices_vault / "postgres-warehouse", tmp_path / "second-name")
    else:
        (alices_vault / "postgres-elsewhere").symlink_to(tmp_path / "elsewhere")
    third = Person(organization_id=ALICE.organization_id, user_id="33333333-3333-4333-8333-333333333333")
    assert vault_for_scope(personal_scope(third)).load("postgres", "warehouse")["PASSWORD"] == "bobs-password"
    [bobs] = [row for row in _list(capsys, BOB)["connections"] if row["origin"] == "personal"]
    assert "earlier-share-takes-priority-for-other-recipients" not in bobs["notes"]


def test_one_persons_listing_keeps_the_full_inventorys_rows_and_notes(stack, capsys):
    shares = [(ALICE, "warehouse"), (BOB, "warehouse"), (ALICE, "team-db"), (BOB, "team_db")]
    for person, name in shares:
        _save(person, name)
    stack.shared_dir.mkdir(exist_ok=True)
    stack.shared_connections_file.write_text(json.dumps([
        {"user_id": SUB_A, "partition_id": ALICE.partition_id, "engine": "postgres", "name": "gone"},
        *({"user_id": person.user_id, "partition_id": person.partition_id, "engine": "postgres", "name": name}
          for person, name in shares),
    ]))
    full = _list(capsys)["connections"]
    # Bob's notes depend on Alice's live shares, which his listing never opens.
    [bobs] = [row for row in full if row["user_id"] == SUB_B and row["name"] == "warehouse"]
    assert "earlier-share-takes-priority-for-other-recipients" in bobs["notes"]
    for person in (ALICE, BOB):
        assert _list(capsys, person)["connections"] == [row for row in full if row["partition_id"] in {None, person.partition_id}]


@pytest.mark.parametrize("kind", ["invalid-json", "invalid-creator", "invalid-fields", "wrong-name", "symlink", "hardlink", "fifo", "directory", "oversized"])
def test_inventory_refuses_bad_records_without_printing_their_values(stack, tmp_path, capsys, kind):
    path = Path(_save(ALICE))
    if kind == "invalid-json":
        path.write_text(SECRET)
    elif kind == "invalid-creator":
        row = json.loads(path.read_text())
        row["fields"]["_created_by"] = SECRET
        path.write_text(json.dumps(row))
    elif kind == "invalid-fields":
        row = json.loads(path.read_text())
        row["fields"] = SECRET
        path.write_text(json.dumps(row))
    elif kind == "wrong-name":
        row = json.loads(path.read_text())
        row["name"] = SECRET
        path.write_text(json.dumps(row))
    elif kind == "oversized":
        path.write_text(SECRET + "x" * MAX_RECORD_BYTES)
    else:
        target = tmp_path / "outside"
        path.rename(target)
        if kind == "symlink":
            path.symlink_to(target)
        elif kind == "hardlink":
            os.link(target, path)
        elif kind == "fifo":
            os.mkfifo(path)
        else:
            path.mkdir()
    status, out, err = _run(connections_ops.main, ["list"], capsys)
    assert status == 1 and out == ""
    assert "Connection inventory" in err
    assert SECRET not in err and "private-host" not in err


@pytest.mark.parametrize("component", ["partition", "vault", "registry-folder", "registry-file"])
def test_inventory_refuses_links_in_directory_chain(stack, tmp_path, capsys, component):
    _save(ALICE)
    partition = tmp_path / "shared" / ALICE.partition_id
    paths = {"partition": partition, "vault": partition / "data-vault", "registry-folder": stack.shared_dir}
    if component == "registry-file":
        path = stack.shared_connections_file
        path.parent.mkdir(exist_ok=True)
        path.write_text("[]")
    else:
        path = paths[component]
        path.mkdir(exist_ok=True)
    moved = tmp_path / "elsewhere"
    path.rename(moved)
    path.symlink_to(moved)
    status, out, err = _run(connections_ops.main, ["list"], capsys)
    assert status == 1 and out == ""
    assert SECRET not in err


def test_inventory_registry_errors_do_not_echo_validation_input(stack, capsys):
    registry = stack.shared_connections_file
    registry.parent.mkdir(exist_ok=True)
    registry.write_text(json.dumps([{"user_id": SECRET}]))
    status, out, err = _run(connections_ops.main, ["list"], capsys)
    assert status == 1 and out == ""
    assert "malformed sharing registry" in err and SECRET not in err


def test_empty_inventory_creates_no_stores(stack, tmp_path, capsys):
    before = _tree(tmp_path)
    assert _list(capsys)["connections"] == []
    assert _tree(tmp_path) == before


def test_missing_managed_mount_does_not_report_an_empty_inventory(stack, tmp_path, capsys):
    stack.vault_dir.rename(tmp_path / "unmounted-vault")
    status, out, err = _run(connections_ops.main, ["list"], capsys)
    assert status == 1 and out == ""
    assert "cannot find the configured managed vault" in err
    assert not stack.vault_dir.exists()


def test_inventory_requires_both_identity_arguments(stack, capsys):
    with pytest.raises(SystemExit) as error:
        connections_ops.main(["list", "--user-id", SUB_A])
    assert error.value.code == 2
    assert "both --organization-id and --user-id" in capsys.readouterr().err


def test_inventory_refuses_untrusted_desktop(monkeypatch, capsys):
    monkeypatch.setenv("COWORK_TRUST_IDENTITY_HEADERS", "false")
    get_app_settings.cache_clear()
    status, out, err = _run(connections_ops.main, ["list"], capsys)
    assert status == 1 and out == ""
    assert "COWORK_TRUST_IDENTITY_HEADERS is off" in err
    get_app_settings.cache_clear()
