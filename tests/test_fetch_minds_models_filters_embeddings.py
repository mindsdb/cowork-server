"""fetch_minds_models must drop embedding models from MindsHub's /v1/models.

MindsHub's model listing includes embedding models alongside chat/completion
models, flagged `"embedding": true`. Picking one for a planning/coding role
would error every turn, so fetch_minds_models filters them out at the one
place every row is parsed — the picker and default-resolution never see them.
"""
import asyncio

import cowork.services.providers as providers
from cowork.services.providers import fetch_minds_models


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _FakeClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None):
        return _Resp(200, {
            "data": [
                {"id": "mindshub_air", "enabled": True},
                {"id": "text-embed-3", "enabled": True, "embedding": True},
                {"id": "sonnet", "enabled": False},
            ],
        })


def test_fetch_minds_models_drops_embedding_rows(monkeypatch):
    monkeypatch.setattr(providers.httpx, "AsyncClient", _FakeClient)
    providers._minds_models_cache.clear()

    ids, efforts, enabled = asyncio.run(
        fetch_minds_models("https://api.mindshub.ai", "mdb_test")
    )

    assert ids == ["mindshub_air", "sonnet"]
    assert "text-embed-3" not in ids
    assert "text-embed-3" not in enabled
