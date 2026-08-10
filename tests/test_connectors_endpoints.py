"""save_connection_direct() — the /connectors/connections/save endpoint.

Calls the FastAPI route function directly rather than going through a
TestClient/app fixture — none exists for this router yet in this repo, and
route handlers are plain Python functions, so a direct call exercises the
same logic without needing to stand up the full app.
"""
from pathlib import Path

from cowork.api.v1.endpoints.connectors.connections import save_connection_direct
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
        result = save_connection_direct(body)
        assert result["user_label"] == "Support"
