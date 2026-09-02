"""save_connection_direct() — the /connectors/connections/save endpoint.

Calls the FastAPI route function directly rather than going through a
TestClient/app fixture — none exists for this router yet in this repo, and
route handlers are plain Python functions, so a direct call exercises the
same logic without needing to stand up the full app.
"""
from pathlib import Path

from cowork.api.v1.endpoints.connectors.connections import save_connection_direct
from cowork.db.scoped import LOCAL_SCOPE
from cowork.schemas.connectors import DirectSaveRequest


class TestSaveConnectionDirectReturnsUserLabel:
    def test_response_includes_user_label(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "cowork.api.v1.endpoints.connectors.connections.ConnectorSettings",
            lambda: type("S", (), {"vault_dir": str(tmp_path / "vault")})(),
        )
        body = DirectSaveRequest(
            connector_id="gmail",
            method="app-password",
            name="",
            values={"email": "a@b.com", "app_password": "x", "user_label": "Support"},
        )
        # LOCAL_SCOPE, not None: the endpoint now takes the request's scope so
        # the vault is org-keyed, and an unscoped call fails closed on an org
        # deployment. A desktop scope keeps the vault path unchanged.
        result = save_connection_direct(body, LOCAL_SCOPE)
        assert result["user_label"] == "Support"

    def test_account_name_titles_a_brand_new_connections_tile(self, tmp_path, monkeypatch):
        # ENG-2188: Electron's OAuth PKCE flow (index.ts) calls this endpoint
        # with account_name set to the provider's fetched account/org/
        # workspace name (e.g. Linear's workspace, Supabase's organization)
        # but no explicit user_label — without default_label, that fell back
        # to the generic engine id ("linear") instead of the workspace name.
        monkeypatch.setattr(
            "cowork.api.v1.endpoints.connectors.connections.ConnectorSettings",
            lambda: type("S", (), {"vault_dir": str(tmp_path / "vault")})(),
        )
        body = DirectSaveRequest(
            connector_id="linear",
            method="browser_oauth_builtin",
            name="",
            values={
                "access_token": "tok",
                "account_email": "user@example.com:org-1",
                "account_name": "Acme Workspace",
            },
        )
        result = save_connection_direct(body, LOCAL_SCOPE)
        assert result["user_label"] == "Acme Workspace"

    def test_second_workspace_with_distinct_account_name_gets_its_own_tile(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "cowork.api.v1.endpoints.connectors.connections.ConnectorSettings",
            lambda: type("S", (), {"vault_dir": str(tmp_path / "vault")})(),
        )
        first = DirectSaveRequest(
            connector_id="linear",
            method="browser_oauth_builtin",
            name="",
            values={
                "access_token": "tok-1",
                "account_email": "user@example.com:org-1",
                "account_name": "Acme Workspace",
            },
        )
        second = DirectSaveRequest(
            connector_id="linear",
            method="browser_oauth_builtin",
            name="",
            values={
                "access_token": "tok-2",
                "account_email": "user@example.com:org-2",
                "account_name": "Other Workspace",
            },
        )
        first_result = save_connection_direct(first, LOCAL_SCOPE)
        second_result = save_connection_direct(second, LOCAL_SCOPE)
        assert first_result["name"] != second_result["name"]
        assert second_result["user_label"] == "Other Workspace"
