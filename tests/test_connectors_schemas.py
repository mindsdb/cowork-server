import asyncio

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from cowork.schemas.connectors import (
    ConnectorSpecResponse,
    ConnectionDetailResponse,
    OAuthConfig,
    ConnectionSummaryResponse,
    SaveConnectionResponse,
)


def test_oauth_redirect_host_is_loopback_only():
    assert OAuthConfig(auth_url="https://a", token_url="https://t").redirect_host == "127.0.0.1"
    assert OAuthConfig(auth_url="https://a", token_url="https://t", redirect_host="localhost").redirect_host == "localhost"
    with pytest.raises(ValidationError):
        OAuthConfig(auth_url="https://a", token_url="https://t", redirect_host="attacker.example")


def test_supabase_uses_dedicated_localhost_redirect():
    config = OAuthConfig(
        auth_url="https://a",
        token_url="https://t",
        redirect_port=47292,
        redirect_host="localhost",
    )
    assert config.redirect_port == 47292
    assert config.redirect_host == "localhost"


class TestSchemasHaveUserLabel:
    def test_summary_defaults_to_none(self):
        r = ConnectionSummaryResponse(engine="postgres", name="a1b2c3")
        assert r.user_label is None

    def test_detail_defaults_to_none(self):
        r = ConnectionDetailResponse(engine="postgres", name="a1b2c3")
        assert r.user_label is None

    def test_save_response_defaults_to_none(self):
        r = SaveConnectionResponse(status="ok", submission_id="s1", engine="postgres", name="a1b2c3", method=None)
        assert r.user_label is None


class TestSpecUsageNotesStayOutOfResponses:
    @staticmethod
    def _spec() -> ConnectorSpecResponse:
        return ConnectorSpecResponse(
            id="demo",
            label="Demo",
            description="Demo connector.",
            category="other",
            form={"form_id": "demo-connector", "title": "Demo", "methods": []},
            usage_notes="Pass the token as a header.",
        )

    def test_notes_validate_but_are_not_serialized(self):
        spec = self._spec()

        assert spec.usage_notes == "Pass the token as a header."
        assert "usage_notes" in ConnectorSpecResponse.model_fields
        assert "usage_notes" not in spec.model_dump()
        assert "usage_notes" not in spec.model_dump_json()

    def test_route_response_omits_notes(self):
        app = FastAPI()

        @app.get("/x", response_model=ConnectorSpecResponse)
        def read_spec():
            return self._spec()

        async def fetch() -> httpx.Response:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                return await client.get("/x")

        response = asyncio.run(fetch())

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == "demo"
        assert "usage_notes" not in body
        properties = app.openapi()["components"]["schemas"]["ConnectorSpecResponse"]["properties"]
        assert "id" in properties
        assert "usage_notes" not in properties
