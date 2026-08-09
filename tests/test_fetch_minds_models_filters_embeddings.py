"""fetch_minds_models must drop embedding models from MindsHub's /v1/models,
and pass through each row's display `label` separately from its id.

MindsHub's model listing includes embedding models alongside chat/completion
models, flagged `"embedding": true`. Picking one for a planning/coding role
would error every turn, so fetch_minds_models filters them out at the one
place every row is parsed — the picker and default-resolution never see them.

Each row may also carry a human-readable `"label"` — display-only, surfaced
separately from `ids` so the id remains the value used for
selection/storage/resolution everywhere else.
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
                {"id": "text-embed-3", "enabled": True, "embedding": True, "label": "Text Embed 3"},
                {"id": "sonnet", "enabled": False, "label": "Claude Sonnet 4.6"},
            ],
        })


def _client_returning(rows):
    class _Client(_FakeClient):
        async def get(self, url, headers=None):
            return _Resp(200, {"data": rows})

    return _Client


def test_fetch_minds_models_drops_embedding_rows(monkeypatch):
    monkeypatch.setattr(providers.httpx, "AsyncClient", _FakeClient)
    providers._minds_models_cache.clear()

    listing = asyncio.run(fetch_minds_models("https://api.mindshub.ai", "mdb_test"))
    ids, efforts, enabled, labels = listing.ids, listing.efforts, listing.enabled, listing.labels

    assert ids == ["mindshub_air", "sonnet"]
    assert "text-embed-3" not in ids
    assert "text-embed-3" not in enabled
    # Embedding row's label must not leak through even though it has one.
    assert "text-embed-3" not in labels
    # A chat model's label passes through; one with no label is just absent
    # (client falls back to its own id-derived label).
    assert labels == {"sonnet": "Claude Sonnet 4.6"}


def test_embedding_ids_are_dropped_when_the_endpoint_has_no_flag(monkeypatch):
    # A BYO openai-compatible endpoint (or OpenAI itself) has no field for this
    # — /v1/models lists text-embedding-3-small exactly like a chat model — so
    # the id is the only signal available.
    rows = [
        {"id": "gpt-4o"},
        {"id": "text-embedding-3-small"},
        {"id": "text-embedding-ada-002"},
        {"id": "nomic-embed-text"},
        {"id": "my-embedding-v2"},
    ]
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_returning(rows))
    providers._minds_models_cache.clear()

    ids = asyncio.run(fetch_minds_models("https://byo.example", "sk-test")).ids

    assert ids == ["gpt-4o"]


def test_explicit_embedding_false_beats_the_id_hint(monkeypatch):
    # An endpoint that publishes the flag is believed in both directions: a
    # chat model that happens to be named like an embeddings one stays listed,
    # so the id heuristic can't override a source that actually knows.
    rows = [
        {"id": "embed-chat-preview", "embedding": False},
        {"id": "text-embedding-3-small", "embedding": True},
    ]
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_returning(rows))
    providers._minds_models_cache.clear()

    ids = asyncio.run(fetch_minds_models("https://byo.example", "sk-test")).ids

    assert ids == ["embed-chat-preview"]


def test_chat_models_are_not_dropped_by_a_coincidental_name(monkeypatch):
    # The hints are deliberately narrow. Embedding *family* names (bge, gte, e5)
    # aren't matched, because a substring test on those would be far more likely
    # to drop a chat model than to catch an embeddings one.
    rows = [{"id": "bge-large"}, {"id": "gte-base"}, {"id": "e5-mistral"}, {"id": "embedded-agent-v1"}]
    monkeypatch.setattr(providers.httpx, "AsyncClient", _client_returning(rows))
    providers._minds_models_cache.clear()

    ids = asyncio.run(fetch_minds_models("https://byo.example", "sk-test")).ids

    assert ids == ["bge-large", "gte-base", "e5-mistral", "embedded-agent-v1"]
