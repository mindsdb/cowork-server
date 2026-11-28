import errno
from pathlib import Path

from anton.core.datasources.data_vault import LocalDataVault

from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.services.connectors.oauth import google as oauth_google
from cowork.services.connectors.oauth.config import OAUTH_SERVICES


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


class TestCatalogueSurvivesAnUnreadableStateFile:
    def test_catalogue_lists_every_service_when_the_state_read_fails(self, tmp_path, monkeypatch):
        # The state store raises on a failed read so no change saves over it.
        # The catalogue only shows that state, so it keeps the page.
        monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "vault"))
        state_path = tmp_path / "oauth_state.json"
        state_path.write_text("{}", encoding="utf-8")
        real_read_text = Path.read_text

        def read_text(self, *args, **kwargs):
            if self == state_path:
                raise OSError(errno.EIO, "Input/output error")
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text)

        items = oauth_google.OAuthService().get_catalogue(
            ConnectorSettings(), OAuthSettings(state_path=str(state_path))
        )

        assert [item["engine"] for item in items] == [cfg.engine for cfg in OAUTH_SERVICES.values()]
        assert not any(item["oauth"]["pending"] for item in items)
