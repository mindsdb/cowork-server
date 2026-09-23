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
    assert "workspace_id" not in captured["json"]


@pytest.mark.asyncio
async def test_mint_turn_key_sends_workspace_id_when_given(monkeypatch):
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
            captured["json"] = json
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    await mint_turn_key(
        user_id="u1", org_id="o1", correlation_id="corr-1",
        ttl_seconds=1200, settings=_Settings(), workspace_id="ws-1",
    )
    assert captured["json"]["workspace_id"] == "ws-1"


class _SeqResp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


def _sequenced_client(monkeypatch, responses):
    """Fake AsyncClient answering each POST with the next response; returns the bodies sent."""
    sent = []
    queue = list(responses)

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json, headers):
            sent.append(dict(json))
            return queue.pop(0)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return sent


@pytest.fixture
def forgotten(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "cowork.turnqueue.auth_keys.forget_stale_hub_workspace",
        lambda **kw: calls.append(kw),
    )
    return calls


@pytest.mark.asyncio
async def test_refused_workspace_retries_on_the_default_and_forgets_the_pick(monkeypatch, forgotten):
    """A grant removed after the pick: auth refuses the stale id, and the turn
    recovers on the default instead of failing every turn from then on."""
    sent = _sequenced_client(monkeypatch, [
        _SeqResp(404, {"code": "workspace_not_found", "detail": "not found"}),
        _SeqResp(201, {"key": "mdb_default"}),
    ])

    key = await mint_turn_key(
        user_id="u1", org_id="o1", correlation_id="corr-1",
        ttl_seconds=1200, settings=_Settings(), workspace_id="ws-stale",
    )

    assert key == "mdb_default"
    assert sent[0]["workspace_id"] == "ws-stale"
    assert "workspace_id" not in sent[1]
    assert forgotten == [{"org_id": "o1", "user_id": "u1", "workspace_id": "ws-stale"}]


@pytest.mark.asyncio
async def test_any_other_404_still_fails_without_retrying(monkeypatch, forgotten):
    """Only the workspace refusal is recoverable; an unknown user/org is not."""
    from cowork.services.product_permissions import ProductPermissionUnavailable

    sent = _sequenced_client(monkeypatch, [_SeqResp(404, {"detail": "user_id u1 not found"})])

    with pytest.raises(ProductPermissionUnavailable):
        await mint_turn_key(
            user_id="u1", org_id="o1", correlation_id="corr-1",
            ttl_seconds=1200, settings=_Settings(), workspace_id="ws-1",
        )

    assert len(sent) == 1
    assert forgotten == []


@pytest.mark.asyncio
async def test_workspace_refusal_without_a_workspace_sent_is_not_retried(monkeypatch, forgotten):
    from cowork.services.product_permissions import ProductPermissionUnavailable

    sent = _sequenced_client(monkeypatch, [_SeqResp(404, {"code": "workspace_not_found"})])

    with pytest.raises(ProductPermissionUnavailable):
        await mint_turn_key(
            user_id="u1", org_id="o1", correlation_id="corr-1",
            ttl_seconds=1200, settings=_Settings(),
        )

    assert len(sent) == 1
    assert forgotten == []


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
