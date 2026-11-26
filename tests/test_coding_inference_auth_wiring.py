"""Exercise the actual app middleware AND Code Mode's inference guard.

The inference routes take an engine session's own ``inference`` capability
token (cowork.capability_tokens), never the window's bearer token.
"""
import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from cowork import capability_tokens
from cowork.auth_middleware import _read_token
from cowork.common.settings.app_settings import get_app_settings


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


@pytest.fixture
def session_token():
    """A live engine session's inference token, as RuntimeManager mints it."""
    token = capability_tokens.mint(purpose='inference', session_id='coding-session-1')
    yield token
    capability_tokens.revoke_session(session_id='coding-session-1')


@pytest.mark.parametrize('path', ['responses', 'responses/compact', 'models'])
def test_only_a_live_session_inference_token_reaches_upstream(app, session_token, path):
    application, calls, desktop_token = app
    client = TestClient(application, client=('127.0.0.1', 50123))
    url = f'/api/v1/coding/inference/{path}'
    for token in (None, 'wrong', desktop_token):
        headers = {'Authorization': f'Bearer {token}'} if token else {}
        assert client.post(url, headers=headers, json={}).status_code == 401
    assert calls == []
    assert client.post(url, headers={'Authorization': f'Bearer {session_token}'}, json={}).status_code == 200
    assert calls == [path]


def test_a_revoked_session_token_is_refused(app, session_token):
    application, calls, _ = app
    client = TestClient(application, client=('127.0.0.1', 50123))
    url = '/api/v1/coding/inference/responses'
    assert client.post(url, headers={'Authorization': f'Bearer {session_token}'}, json={}).status_code == 200
    capability_tokens.revoke(token=session_token)
    assert client.post(url, headers={'Authorization': f'Bearer {session_token}'}, json={}).status_code == 401
    assert calls == ['responses']


def test_a_token_of_another_purpose_is_refused(app):
    application, calls, _ = app
    client = TestClient(application, client=('127.0.0.1', 50123))
    # A later purpose (hooks, mcp, telemetry) the Literal does not name yet.
    other = capability_tokens.mint(purpose='hooks', session_id='coding-session-2')  # type: ignore[arg-type]
    try:
        result = client.post(
            '/api/v1/coding/inference/responses', headers={'Authorization': f'Bearer {other}'}, json={},
        )
    finally:
        capability_tokens.revoke(token=other)
    assert result.status_code == 401
    assert calls == []


def test_private_inference_credential_cannot_access_task_or_other_api_routes(app, session_token):
    application, calls, _ = app
    client = TestClient(application, client=('127.0.0.1', 50123))
    for path in ('/api/v1/coding/sessions', '/api/v1/conversations/', '/api/v1/coding/inference/anything-else'):
        assert client.get(path, headers={'Authorization': f'Bearer {session_token}'}).status_code == 401
    assert calls == []


def test_valid_inference_credential_still_cannot_be_used_by_a_remote_peer(app, session_token):
    application, calls, _ = app
    client = TestClient(application, client=('198.51.100.1', 50123))
    result = client.post('/api/v1/coding/inference/responses', headers={'Authorization': f'Bearer {session_token}'}, json={})
    assert result.status_code == 403
    assert calls == []
