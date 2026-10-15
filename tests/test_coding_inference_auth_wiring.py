"""Exercise the actual app middleware AND Code Mode's inference guard."""
import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from cowork.auth_middleware import _read_token
from cowork.common.settings.app_settings import get_app_settings
from cowork.coding.engines.codex_config import LOCAL_PROXY_TOKEN


@pytest.fixture
def app(monkeypatch, tmp_path):
    from cowork.api.v1.endpoints import coding
    monkeypatch.setenv('COWORK_HOME', str(tmp_path))
    monkeypatch.setenv('COWORK_TENANCY_MODE', 'local')
    monkeypatch.setenv('COWORK_REQUIRE_AUTH', 'true')
    get_app_settings.cache_clear()
    calls = []
    async def upstream(request, path, credentials):
        calls.append(path)
        return JSONResponse({'ok': True})
    monkeypatch.setattr(coding, 'proxy_inference', upstream)
    from cowork.server import create_app
    application = create_app()
    yield application, calls, _read_token(tmp_path / '.env')
    get_app_settings.cache_clear()


@pytest.mark.parametrize('path', ['responses', 'responses/compact', 'models'])
def test_only_private_inference_credential_reaches_upstream(app, path):
    application, calls, desktop_token = app
    client = TestClient(application, client=('127.0.0.1', 50123))
    url = f'/api/v1/coding/inference/{path}'
    for token in (None, 'wrong', desktop_token):
        headers = {'Authorization': f'Bearer {token}'} if token else {}
        assert client.post(url, headers=headers, json={}).status_code == 401
    assert calls == []
    assert client.post(url, headers={'Authorization': f'Bearer {LOCAL_PROXY_TOKEN}'}, json={}).status_code == 200
    assert calls == [path]


def test_private_inference_credential_cannot_access_task_or_other_api_routes(app):
    application, calls, _ = app
    client = TestClient(application, client=('127.0.0.1', 50123))
    for path in ('/api/v1/coding/sessions', '/api/v1/conversations/', '/api/v1/coding/inference/anything-else'):
        assert client.get(path, headers={'Authorization': f'Bearer {LOCAL_PROXY_TOKEN}'}).status_code == 401
    assert calls == []


def test_valid_inference_credential_still_cannot_be_used_by_a_remote_peer(app):
    application, calls, _ = app
    client = TestClient(application, client=('198.51.100.1', 50123))
    result = client.post('/api/v1/coding/inference/responses', headers={'Authorization': f'Bearer {LOCAL_PROXY_TOKEN}'}, json={})
    assert result.status_code == 403
    assert calls == []
