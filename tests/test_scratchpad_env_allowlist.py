"""Regression: the scratchpad env allowlist must never collapse to None.

The harness restricts the scratchpad subprocess env to the DS_* vars of
connections enabled in the current conversation. In anton, ``None`` means
"copy the full os.environ" (legacy CLI behaviour) — and since inject_env()
side-effects persist in the server process, a conversation that yields zero
keys (empty vault, or all connections disabled) would inherit DS_* creds
injected by *other* conversations if the empty set were coerced to None.
The allowlist is therefore always an explicit set, even when empty.
"""
from __future__ import annotations

from cowork.harnesses.anton_harness.harness import _scratchpad_env_allowlist


class _FakeVault:
    """Duck-typed stand-in for anton's LocalDataVault."""

    def __init__(self, connections: dict[tuple[str, str], list[str] | None]) -> None:
        self._connections = connections

    def list_connections(self) -> list[dict[str, str]]:
        return [{"engine": e, "name": n} for e, n in self._connections]

    def inject_env(self, engine: str, name: str) -> list[str] | None:
        return self._connections[(engine, name)]


def test_empty_vault_yields_empty_set_not_none():
    allowlist = _scratchpad_env_allowlist(_FakeVault({}))
    assert allowlist == set()
    assert allowlist is not None


def test_missing_vault_yields_empty_set():
    assert _scratchpad_env_allowlist(None) == set()


def test_collects_injected_keys_across_connections():
    vault = _FakeVault({
        ("postgres", "prod"): ["DS_POSTGRES_PROD__HOST", "DS_POSTGRES_PROD__PASSWORD"],
        ("mysql", "analytics"): ["DS_MYSQL_ANALYTICS__HOST"],
    })
    assert _scratchpad_env_allowlist(vault) == {
        "DS_POSTGRES_PROD__HOST",
        "DS_POSTGRES_PROD__PASSWORD",
        "DS_MYSQL_ANALYTICS__HOST",
    }


def test_connection_that_fails_to_inject_is_skipped():
    vault = _FakeVault({
        ("postgres", "gone"): None,
        ("postgres", "prod"): ["DS_POSTGRES_PROD__HOST"],
    })
    assert _scratchpad_env_allowlist(vault) == {"DS_POSTGRES_PROD__HOST"}
