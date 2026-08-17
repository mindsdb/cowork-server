import httpx
import pytest

from cowork.services.connectors.posthog import PostHogDiscoveryError, discover_projects


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Client:
    response = None
    calls = []

    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


@pytest.mark.asyncio
async def test_discovers_posthog_project_choices(monkeypatch):
    _Client.calls = []
    _Client.response = _Response(200, {"results": [{"id": 12, "name": "Production"}]})
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    projects = await discover_projects(
        personal_api_key="secret-key", host="https://us.posthog.com"
    )

    assert projects[0].id == "12"
    assert projects[0].name == "Production"
    assert _Client.calls == [(
        "https://us.posthog.com/api/projects/",
        {"headers": {"Authorization": "Bearer secret-key"}},
    )]


@pytest.mark.asyncio
async def test_hides_posthog_auth_failure(monkeypatch):
    _Client.response = _Response(401, {})
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    with pytest.raises(PostHogDiscoveryError, match="rejected"):
        await discover_projects(personal_api_key="secret-key", host="https://eu.posthog.com")


@pytest.mark.asyncio
async def test_rejects_invalid_posthog_host():
    with pytest.raises(PostHogDiscoveryError, match="valid HTTPS PostHog host"):
        await discover_projects(personal_api_key="secret-key", host="not a URL")
