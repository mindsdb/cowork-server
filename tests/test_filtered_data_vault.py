"""A turn's vault with some connections turned off (FilteredDataVault).

On the desktop (trust off, no managed folder, LOCAL_SCOPE), a connection the
conversation turned off is hidden from the agent's environment, its listing and
anton's paste match, while a connection the agent saves in the turn writes
straight through and outlives the turn. The connecting view still lists a hidden
connection's label, so a pasted connection never repeats it, but loads it as its
label alone, so the paste is saved as a new connection rather than updating the
hidden one.
"""
from __future__ import annotations

from pathlib import Path

from anton.core.datasources.data_vault import LocalDataVault

from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import LOCAL_SCOPE
from cowork.services.connectors.layered_vault import FilteredDataVault
from cowork.services.connectors.persist import own_vault_for_scope
from tests._turns import built_turn, connect_in_chat

HIDDEN = frozenset({("postgres", "old")})


def _source(tmp_path: Path) -> LocalDataVault:
    vault = LocalDataVault(tmp_path / "vault")
    vault.save("postgres", "old", {"host": "old.example", "database": "sales", "_user_label": "postgres"})
    return vault


def test_a_hidden_connection_is_invisible_to_the_turn(tmp_path: Path):
    filtered = FilteredDataVault(source=_source(tmp_path), hidden=HIDDEN)

    assert filtered.list_connections() == []
    assert filtered.load("postgres", "old") is None
    assert filtered.read_record("postgres", "old") is None
    assert filtered.env_for("postgres", "old") is None
    assert filtered.inject_env("postgres", "old") is None


def test_an_agent_save_in_a_filtered_turn_outlives_it(tmp_path: Path):
    source = _source(tmp_path)
    filtered = FilteredDataVault(source=source, hidden=HIDDEN)

    filtered.save("mysql", "new", {"host": "new.example", "database": "crm"})

    # The save wrote straight through to the real vault.
    assert source.load("mysql", "new")["host"] == "new.example"
    # The hidden connection is untouched.
    assert source.load("postgres", "old")["host"] == "old.example"


def test_a_delete_in_a_filtered_turn_reaches_the_source(tmp_path: Path):
    source = _source(tmp_path)
    filtered = FilteredDataVault(source=source, hidden=HIDDEN)

    assert filtered.delete("postgres", "old") is True
    assert source.load("postgres", "old") is None


def test_the_connecting_view_exposes_a_hidden_label_but_not_its_identity(tmp_path: Path):
    filtered = FilteredDataVault(source=_source(tmp_path), hidden=HIDDEN)

    connecting = filtered.for_connecting()

    # The hidden connection is listed, so a pasted connection's label is
    # de-duplicated against it and never repeats "postgres".
    assert [(c["engine"], c["name"]) for c in connecting.list_connections()] == [("postgres", "old")]
    # But it loads as its label alone, so the paste matches no identity field
    # and is saved as a new connection rather than updating the hidden one.
    assert connecting.load("postgres", "old") == {"_user_label": "postgres"}
    assert connecting.read_record("postgres", "old")["fields"] == {"_user_label": "postgres"}


async def test_a_desktop_turn_with_a_connection_off_saves_a_paste_as_a_new_one(tmp_path: Path, monkeypatch):
    """A desktop turn whose conversation turned a connection off doesn't list
    it, and a connection pasted in chat lands in the desktop vault as a new
    connection, labelled apart from the hidden one, which stays unchanged."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "local")
    monkeypatch.delenv("COWORK_TRUST_IDENTITY_HEADERS", raising=False)
    monkeypatch.delenv("COWORK_MANAGED_DIR", raising=False)
    monkeypatch.setenv("COWORK_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "vault"))
    get_app_settings.cache_clear()
    try:
        vault = own_vault_for_scope(None)
        vault.save("postgres", "old", {"host": "old.example", "database": "sales", "_user_label": "postgres"})
        old = vault.read_record("postgres", "old")

        async with built_turn(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            scope=LOCAL_SCOPE,
            disabled_connections=[{"engine": "postgres", "name": "old"}],
        ) as turn:
            listed = [(c["engine"], c["name"]) for c in turn.config.data_vault.list_connections()]
            assert ("postgres", "old") not in listed
            connected = await connect_in_chat(
                turn,
                monkeypatch,
                known_variables={
                    "host": "new.example", "database": "sales", "user": "me", "password": "newSecret123",
                },
            )

        assert connected.answer.startswith("Saved connection"), connected.answer
        vault = own_vault_for_scope(None)
        [new] = [c["name"] for c in vault.list_connections() if c["name"] != "old"]
        stored = vault.load("postgres", new)
        assert (stored["host"], stored["_user_label"]) == ("new.example", "postgres 2")
        assert vault.read_record("postgres", "old") == old
    finally:
        get_app_settings.cache_clear()
