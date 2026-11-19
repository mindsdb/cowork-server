"""Connections that normalize to one DS_ env prefix, across a reader's layers.

anton builds the scratchpad DS_* environment by iterating list_connections and
calling env_for per connection, and _slug_env_prefix maps '-', '_' and case onto
one name, so a later connection's env overwrites an earlier one's under that
prefix. LayeredDataVault.list_connections leaves out an entry whose prefix an
earlier layer holds (managed, then the reader's own, then everyone else's in a
deployment-wide vault, then the shared ones), so nobody else's credentials stand
in for the reader's. Two entries of one layer both stay listed, as on the
desktop.
"""
from __future__ import annotations

from pathlib import Path

from anton.core.datasources.data_vault import LocalDataVault

from cowork.creators import Changer
from cowork.managed import SharedConnection
from cowork.services.connectors.layered_vault import CREATED_BY_FIELD, LayeredDataVault, SharedFrom

A_PARTITION = "44e92829-e701-599b-88f9-69f5bf1165eb"
A_SUB = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
B_SUB = "1b2c3d4e-5f60-4718-8a9b-0c1d2e3f4a5b"


def _merged_env(vault: LayeredDataVault) -> dict[str, str]:
    """The DS_* environment anton builds from the listing: each listed
    connection's env in order, a later one overwriting an earlier one."""
    env: dict[str, str] = {}
    for connection in vault.list_connections():
        env.update(vault.env_for(connection["engine"], connection["name"]) or {})
    return env


def _listed(vault: LayeredDataVault) -> list[tuple[str, str]]:
    return [(c["engine"], c["name"]) for c in vault.list_connections()]


def test_a_shared_connection_sharing_an_env_prefix_is_hidden(tmp_path: Path):
    own = LocalDataVault(tmp_path / "own")
    own.save("postgres", "sales-db", {"host": "a.example"})
    other = LocalDataVault(tmp_path / "other")
    other.save("postgres", "sales_db", {"host": "shared.example"})
    entry = SharedConnection(user_id=A_SUB, partition_id=A_PARTITION, engine="postgres", name="sales_db")
    vault = LayeredDataVault(own=own, shared=(SharedFrom(entry=entry, vault=other),))

    # The reader's own entry stays; the shared one sharing its DS_ prefix drops.
    assert _listed(vault) == [("postgres", "sales-db")]
    assert _merged_env(vault)["DS_POSTGRES_SALES_DB__HOST"] == "a.example"


def test_a_managed_connection_keeps_its_prefix_over_an_own_one(tmp_path: Path):
    managed = LocalDataVault(tmp_path / "managed")
    managed.save("postgres", "warehouse_1", {"host": "managed.example"})
    own = LocalDataVault(tmp_path / "own")
    own.save("postgres", "warehouse-1", {"host": "own.example"})
    vault = LayeredDataVault(own=own, managed=managed)

    assert _listed(vault) == [("postgres", "warehouse_1")]
    assert _merged_env(vault)["DS_POSTGRES_WAREHOUSE_1__HOST"] == "managed.example"


def test_two_own_connections_sharing_a_prefix_both_stay_listed(tmp_path: Path):
    """Both are the reader's own: hiding one would leave a connection they can
    neither see nor delete. The desktop lists both too."""
    own = LocalDataVault(tmp_path / "own")
    own.save("postgres", "sales-db", {"host": "one.example"})
    own.save("postgres", "sales_db", {"host": "two.example"})
    vault = LayeredDataVault(own=own)

    assert sorted(_listed(vault)) == [("postgres", "sales-db"), ("postgres", "sales_db")]


def test_in_a_deployment_wide_vault_the_readers_own_entry_keeps_the_prefix(tmp_path: Path):
    """Everyone's connections sit in one vault. Alice's entry sorts first, but
    Bob's listing keeps his own, so his agent never gets her credentials under
    the prefix."""
    shared_vault = LocalDataVault(tmp_path / "base")
    shared_vault.save("postgres", "sales-db", {"host": "alice.example", CREATED_BY_FIELD: A_SUB})
    shared_vault.save("postgres", "sales_db", {"host": "bob.example", CREATED_BY_FIELD: B_SUB})
    bob = Changer(user_id=B_SUB, operator_view=False, deployment_wide=True)
    alice = Changer(user_id=A_SUB, operator_view=False, deployment_wide=True)

    bobs = LayeredDataVault(own=shared_vault, changer=bob)
    alices = LayeredDataVault(own=shared_vault, changer=alice)

    assert _listed(bobs) == [("postgres", "sales_db")]
    assert _merged_env(bobs)["DS_POSTGRES_SALES_DB__HOST"] == "bob.example"
    assert _listed(alices) == [("postgres", "sales-db")]
    assert _merged_env(alices)["DS_POSTGRES_SALES_DB__HOST"] == "alice.example"
