from __future__ import annotations

import logging

from anton.utils.datasources import scrub_credentials

import cowork.services.connectors.vault_secrets as vault_secrets_mod
from cowork.db.scoped import MissingTenantScopeError
from cowork.services.connectors.vault_secrets import register_vault_secrets


async def test_an_unreadable_vault_falls_back_to_shape_scrubbing(monkeypatch, caplog):
    """Registration runs on every turn, so a vault it cannot read must not
    fail the turn; the history is still scrubbed by API-key shape."""
    def missing_scope(scope):
        raise MissingTenantScopeError("no org in scope")

    monkeypatch.setattr(vault_secrets_mod, "vault_for_scope", missing_scope)
    # The migration tests' fileConfig disables every logger created before it.
    monkeypatch.setattr(vault_secrets_mod.logger, "disabled", False)

    with caplog.at_level(logging.WARNING, logger=vault_secrets_mod.__name__):
        await register_vault_secrets(None)

    assert "Could not register vault secrets" in caplog.text
    leaked_key = "sk-" + "a" * 30
    assert scrub_credentials(f"my key is {leaked_key}") == "my key is [REDACTED_API_KEY]"
