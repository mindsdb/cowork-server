import pytest
import httpx
from cowork.turnqueue.auth_keys import mint_turn_key, revoke_turn_key
from cowork.common.settings.app_settings import TurnQueueSettings


class _Settings(TurnQueueSettings):
    auth_internal_base_url: str = "http://auth.internal"
    auth_internal_secret: str = "shh"
    turn_key_ttl_seconds: int = 1200


@pytest.mark.asyncio
async def test_mint_turn_key_posts_and_returns_plaintext(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 201
        def json(self): return {"key": "mdb_turnkey123"}
        def raise_for_status(self): pass

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json, headers):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    key = await mint_turn_key(
        user_id="u1", org_id="o1", correlation_id="corr-1",
        ttl_seconds=1200, settings=_Settings(),
    )
    assert key == "mdb_turnkey123"
    assert captured["url"].endswith("/internal/turn-keys/")
    assert "/v1/internal/turn-keys/" not in captured["url"]
    assert captured["headers"]["X-Internal-Auth"] == "shh"
    assert "Authorization" not in captured["headers"]
    assert captured["json"]["instance_id"] == "corr-1"
    assert captured["json"]["expiry_date"]  # present


@pytest.mark.asyncio
async def test_revoke_turn_key_uses_cluster_only_route(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 204

        def raise_for_status(self):
            pass

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def delete(self, url, headers):
            captured["url"] = url
            captured["headers"] = headers
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    await revoke_turn_key(instance_id="corr-1", settings=_Settings())

    assert captured["url"] == "http://auth.internal/internal/turn-keys/corr-1/"
    assert "/v1/internal/turn-keys/" not in captured["url"]
    assert captured["headers"] == {"X-Internal-Auth": "shh"}
