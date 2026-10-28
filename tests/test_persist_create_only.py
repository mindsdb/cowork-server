"""A create-only save takes its exact name or nothing.

Without it a save whose name is held by another account lands under a ``-N``
suffix; a caller that promised one name must refuse instead.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from anton.core.datasources.data_vault import LocalDataVault

from cowork.services.connectors.persist import ConnectionExistsError, persist_connection

FIRST = {"host": "db.example.com", "user": "first", "password": "first-pw-3c1d"}
SECOND = {"host": "db.example.com", "user": "second", "password": "second-pw-8a4e"}


@pytest.fixture
def vault(tmp_path):
    return LocalDataVault(Path(tmp_path) / "vault")


def test_a_free_name_is_saved_under_that_name(vault):
    slug = persist_connection("postgres", None, "corp", FIRST, create_only=True, vault=vault)

    assert slug == "corp"
    assert vault.read_record("postgres", "corp")["fields"]["user"] == "first"


@pytest.mark.parametrize("taken_by", [FIRST, SECOND], ids=["same-account", "other-account"])
def test_a_taken_name_raises_and_leaves_every_record_as_it_was(vault, taken_by):
    persist_connection("postgres", None, "corp", taken_by, vault=vault)
    before = vault.read_record("postgres", "corp")

    with pytest.raises(ConnectionExistsError):
        persist_connection("postgres", None, "corp", SECOND, create_only=True, vault=vault)

    assert vault.read_record("postgres", "corp") == before
    assert [c["name"] for c in vault.list_connections()] == ["corp"]


def test_without_create_only_another_account_still_gets_a_suffix(vault):
    persist_connection("postgres", None, "corp", FIRST, vault=vault)

    slug = persist_connection("postgres", None, "corp", SECOND, vault=vault)

    assert slug == "corp-2"
