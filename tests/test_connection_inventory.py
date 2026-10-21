"""The operator sees ownership and stale storage, with no credential output or writes."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from anton.core.datasources.data_vault import LocalDataVault

from cowork.common.settings.app_settings import get_app_settings
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
