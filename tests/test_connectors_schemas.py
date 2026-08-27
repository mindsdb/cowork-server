import pytest
from pydantic import ValidationError

from cowork.schemas.connectors import (
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
