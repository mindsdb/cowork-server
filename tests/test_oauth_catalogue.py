from pathlib import Path

from anton.core.datasources.data_vault import LocalDataVault

from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.services.connectors.oauth import google as oauth_google


class TestCatalogueReturnsRealUserLabel:
    def test_catalogue_connection_has_user_label(self, tmp_path, monkeypatch):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        vault.save("gmail", "abc123", {"email": "a@b.com", "_user_label": "Support"})
        # `ConnectorSettings(vault_dir=...)` does NOT work — `vault_dir` declares a
        # `validation_alias` (COWORK_VAULT_DIR/CONNECTOR_VAULT_DIR) and this model
        # has no `populate_by_name=True`, so pydantic only accepts the alias keys;
        # the plain `vault_dir` kwarg is silently dropped (extra="ignore") and the
        # field falls back to its real on-disk default. Override via the env var
        # instead, matching TestOAuthCallbackDedup's working pattern.
        monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "vault"))
        connector_settings = ConnectorSettings()
        oauth_settings = OAuthSettings(state_path=str(tmp_path / "oauth_state.json"))

        svc = oauth_google.OAuthService()
        items = svc.get_catalogue(connector_settings, oauth_settings)
        gmail_item = next(i for i in items if i["engine"] == "gmail")
        assert gmail_item["connections"][0]["user_label"] == "Support"
        assert gmail_item["connections"][0].get("label") != "abc123"
