from dataclasses import dataclass
from unittest.mock import AsyncMock
import httpx
import pytest
from fastapi import HTTPException
from cowork.services import content_safety as safety
from cowork.schemas.responses import ResponsesRequest

def config(**kwargs):
    return safety.ContentSafetySettings(required=True, endpoint='https://safety.example/check', **kwargs)

def transport(body, status=200):
    return httpx.MockTransport(lambda request: httpx.Response(status, json=body))

def test_moderation_settings_read_only_their_own_environment(monkeypatch):
    prefix = "COWORK_CONTENT_SAFETY_"
    for name, value in {
        "REQUIRED": "true", "ENDPOINT": "https://other.example", "MODEL": "other", "API_KEY": "unrelated-key",
    }.items():
        monkeypatch.delenv(prefix + name, raising=False)
        monkeypatch.setenv(name, value)

    defaults = safety.ContentSafetySettings(_env_file=None)
    assert defaults.required is False
    assert defaults.endpoint == ""
    assert defaults.model == safety.MINDS_FREE_MODEL
    assert defaults.api_key is None

    monkeypatch.setenv(prefix + "REQUIRED", "true")
    monkeypatch.setenv(prefix + "ENDPOINT", "https://moderation.example")
    monkeypatch.setenv(prefix + "MODEL", "classifier")
    monkeypatch.setenv(prefix + "API_KEY", "moderation-key")
    settings = safety.ContentSafetySettings(_env_file=None)
    assert settings.required is True
    assert settings.endpoint == "https://moderation.example"
    assert settings.model == "classifier"
    assert settings.api_key.get_secret_value() == "moderation-key"


@pytest.mark.parametrize('body,status,expected', [
    ({'decision':'block'},200,422), ({'decision':'unknown'},200,503),
    ({},200,503), ([],200,503), ({'decision':'allow'},500,503),
    ({'decision':'allow'},302,503),
])
async def test_denials_and_service_failures_never_allow(body,status,expected):
    with pytest.raises(HTTPException) as exc:
        await safety.check_content('test',purpose='input',settings=config(),transport=transport(body,status))
    assert exc.value.status_code == expected

async def test_allow_requires_explicit_service_verdict():
    await safety.check_content('test',purpose='input',settings=config(),transport=transport({'decision':'allow'}))

async def test_missing_service_and_oversized_content_fail_closed():
    with pytest.raises(HTTPException) as exc:
        await safety.check_content('test',purpose='input',settings=safety.ContentSafetySettings(required=True))
    assert exc.value.status_code == 503
    with pytest.raises(HTTPException) as exc:
        await safety.check_content('x'*(safety.MAX_CONTENT_CHARS+1),purpose='output',settings=config())
    assert exc.value.status_code == 413

async def test_text_split_across_deltas_checked_before_exposure(monkeypatch):
    from anton.core.llm.provider import StreamTextDelta
    monkeypatch.setenv('COWORK_CONTENT_SAFETY_REQUIRED','true')
    check = AsyncMock(side_effect=HTTPException(422,'blocked'))
    monkeypatch.setattr(safety,'check_content',check)
    async def source():
        yield StreamTextDelta(text='split ')
        yield StreamTextDelta(text='unsafe content')
    visible=[]
    with pytest.raises(HTTPException):
        async for event in safety.guarded_events(source()): visible.append(event)
    assert visible == []
    assert check.await_args.args == ('split unsafe content',)

async def test_attachment_ids_cannot_stand_in_for_scanned_contents(monkeypatch):
    monkeypatch.setenv('COWORK_CONTENT_SAFETY_REQUIRED','true')
    with pytest.raises(HTTPException) as exc:
        await safety.check_request(ResponsesRequest(input='Read this',attachment_ids=['00000000-0000-0000-0000-000000000001']))
    assert exc.value.status_code == 503

async def test_structured_tool_messages_are_checked(monkeypatch):
    monkeypatch.setenv('COWORK_CONTENT_SAFETY_REQUIRED','true')
    check = AsyncMock()
    monkeypatch.setattr(safety,'check_content',check)
    @dataclass
    class Event:
        message: str
    async def source(): yield Event(message='tool message')
    result=[event async for event in safety.guarded_events(source())]
    assert len(result)==1
    assert 'tool message' in check.await_args.args[0]

async def test_non_text_event_cannot_reset_moderation_context(monkeypatch):
    from anton.core.llm.provider import StreamTextDelta
    monkeypatch.setenv('COWORK_CONTENT_SAFETY_REQUIRED','true')
    check=AsyncMock()
    monkeypatch.setattr(safety,'check_content',check)
    async def source():
        yield StreamTextDelta(text='first half')
        yield {'progress':'waiting'}
        yield StreamTextDelta(text='second half')
    result=[event async for event in safety.guarded_events(source())]
    assert len(result)==3
    assert 'first half' in check.await_args.args[0]
    assert 'second half' in check.await_args.args[0]

async def test_inline_file_is_not_treated_as_text(monkeypatch):
    monkeypatch.setenv('COWORK_CONTENT_SAFETY_REQUIRED','true')
    with pytest.raises(HTTPException) as exc:
        await safety.check_request(ResponsesRequest(input=[{'role':'user','content':[{'type':'input_file','file_id':'file'}]}]))
    assert exc.value.status_code == 503

async def test_moderator_never_receives_key_shapes_or_provider_secrets(monkeypatch):
    import json
    key = 'sk-' + 'a' * 48
    secret = 'synthetic-vault-password-1234'
    monkeypatch.setenv('OPENAI_API_KEY', secret)
    seen = []
    def respond(request):
        seen.append(json.loads(request.content)['content'])
        return httpx.Response(200, json={'decision': 'allow'})
    await safety.check_content(f'Please use {key} and {secret}', purpose='input',
                               settings=config(), transport=httpx.MockTransport(respond))
    assert len(seen) == 1
    assert key not in seen[0]
    assert secret not in seen[0]
    assert 'Please use' in seen[0]

async def test_tool_fragments_are_checked_once_and_never_leak_before_denial(monkeypatch):
    from anton.core.llm.provider import StreamToolUseDelta, StreamToolUseEnd
    monkeypatch.setenv('COWORK_CONTENT_SAFETY_REQUIRED','true')
    check = AsyncMock(side_effect=HTTPException(422, 'blocked'))
    monkeypatch.setattr(safety, 'check_content', check)
    async def source():
        for _ in range(250):
            yield StreamToolUseDelta(id='tool', json_delta='x' * 20)
        yield StreamToolUseEnd(id='tool')
    visible = []
    with pytest.raises(HTTPException):
        async for event in safety.guarded_events(source()):
            visible.append(event)
    assert visible == []
    assert check.await_count == 1
    assert check.await_args.args == ('x' * 5000,)

async def test_internal_history_does_not_consume_new_answer_budget(monkeypatch):
    from anton.core.llm.provider import StreamTextDelta
    from cowork.harnesses.anton_harness.stream_formatter import TurnContext, TurnHistory
    monkeypatch.setenv('COWORK_CONTENT_SAFETY_REQUIRED','true')
    check = AsyncMock()
    monkeypatch.setattr(safety, 'check_content', check)
    context = TurnContext(question='current', history=[{'text': 'x' * 200000}])
    history = TurnHistory(rows=[{'text': 'y' * 200000}])
    async def source():
        yield context
        yield StreamTextDelta(text='A short safe answer.')
        yield history
    result = [event async for event in safety.guarded_events(source())]
    assert len(result) == 3
    assert check.await_count == 1
    assert check.await_args.args == ('A short safe answer.',)

async def test_reasoning_fragments_coalesce_and_tool_payload_is_preserved(monkeypatch):
    from anton.core.llm.provider import StreamReasoningDelta, StreamToolUseDelta, StreamToolUseEnd
    monkeypatch.setenv('COWORK_CONTENT_SAFETY_REQUIRED','true')
    check = AsyncMock()
    monkeypatch.setattr(safety, 'check_content', check)
    async def source():
        yield StreamReasoningDelta(text='Thinking ')
        yield StreamReasoningDelta(text='safely')
        yield StreamToolUseDelta(id='a', json_delta='{"a":')
        yield StreamToolUseDelta(id='a', json_delta='1}')
        yield StreamToolUseEnd(id='a')
    result = [event async for event in safety.guarded_events(source())]
    assert result == [StreamReasoningDelta(text='Thinking safely'),
                      StreamToolUseDelta(id='a', json_delta='{"a":1}'), StreamToolUseEnd(id='a')]
    assert check.await_count == 2

@pytest.mark.parametrize('data,mime,status', [
    (b'<script>media</script>', 'text/html', 503),
    (b'\x89PNG', 'image/png', 503),
    (b'%PDF', 'application/pdf', 503),
    (b'\xff', 'text/plain', 503),
    (b'abc\x00def', 'text/plain', 503),
    (b'x' * 100001, 'text/plain', 413),
])
async def test_unsupported_file_contents_fail_closed(monkeypatch, data, mime, status):
    from cowork.db.scoped import LOCAL_SCOPE
    monkeypatch.setenv('COWORK_CONTENT_SAFETY_REQUIRED', 'true')
    check = AsyncMock()
    monkeypatch.setattr(safety, 'check_content', check)
    with pytest.raises(HTTPException) as exc:
        await safety.check_file_bytes(data, content_type=mime, purpose='input', scope=LOCAL_SCOPE)
    assert exc.value.status_code == status
    check.assert_not_called()

@pytest.mark.parametrize('kind', ['efs', 'pinned'])
async def test_file_response_serves_only_the_approved_snapshot(monkeypatch, tmp_path, kind):
    from contextlib import ExitStack
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.api.v1.efs_file_response import EfsFileResponse
    from cowork.api.v1.pinned_response import PinnedStreamingResponse
    from tests.support.pinned_responses import served_body
    monkeypatch.setenv('COWORK_CONTENT_SAFETY_REQUIRED', 'true')
    target = tmp_path / 'test.txt'
    target.write_bytes(b'approved text')
    checked = []
    async def approve(content, **kwargs):
        checked.append(content)
        target.write_bytes(b'unchecked changed text')
    monkeypatch.setattr(safety, 'check_content', approve)
    resources = ExitStack()
    closed = []
    resources.callback(lambda: closed.append(True))
    if kind == 'efs':
        response = EfsFileResponse(target, scope=LOCAL_SCOPE, media_type='text/plain')
    else:
        handle = resources.enter_context(target.open('rb'))
        response = PinnedStreamingResponse(iter(lambda: handle.read(65536), b''), resources=resources,
                                           scope=LOCAL_SCOPE, media_type='text/plain')
    assert await served_body(response) == b'approved text'
    assert checked == ['approved text']
    assert response.headers['cache-control'] == 'no-store'
    assert response.headers['content-length'] == '13'
    if kind == 'pinned':
        assert closed == [True]

@pytest.mark.parametrize('kind', ['efs', 'pinned'])
async def test_file_denial_sends_no_headers_or_body_and_releases_resources(monkeypatch, tmp_path, kind):
    from contextlib import ExitStack
    from cowork.db.scoped import LOCAL_SCOPE
    from cowork.api.v1.efs_file_response import EfsFileResponse
    from cowork.api.v1.pinned_response import PinnedStreamingResponse
    monkeypatch.setenv('COWORK_CONTENT_SAFETY_REQUIRED', 'true')
    monkeypatch.setattr(safety, 'check_content', AsyncMock(side_effect=HTTPException(422, 'blocked')))
    target = tmp_path / 'test.txt'
    target.write_bytes(b'blocked content')
    resources = ExitStack()
    closed = []
    resources.callback(lambda: closed.append(True))
    if kind == 'efs':
        response = EfsFileResponse(target, scope=LOCAL_SCOPE, media_type='text/plain')
    else:
        handle = resources.enter_context(target.open('rb'))
        response = PinnedStreamingResponse(iter(lambda: handle.read(65536), b''), resources=resources,
                                           scope=LOCAL_SCOPE, media_type='text/plain')
    send = AsyncMock()
    with pytest.raises(HTTPException):
        await response({'type': 'http', 'method': 'GET', 'headers': []}, AsyncMock(), send)
    send.assert_not_called()
    if kind == 'pinned':
        assert closed == [True]

async def test_project_upload_writes_only_approved_bytes(monkeypatch):
    from io import BytesIO
    from fastapi import UploadFile
    from starlette.datastructures import Headers
    from cowork.api.v1.endpoints import project_files
    from cowork.db.scoped import LOCAL_SCOPE
    monkeypatch.setenv('COWORK_CONTENT_SAFETY_REQUIRED', 'true')
    source = BytesIO(b'original')
    upload = UploadFile(source, filename='note.txt', headers=Headers({'content-type': 'text/plain'}))
    async def approve(content, **kwargs):
        source.seek(0)
        source.write(b'changed!')
    monkeypatch.setattr(safety, 'check_content', approve)
    written = []
    async def write(call, **kwargs):
        written.extend(item.file.read() for item in call.keywords['uploads'])
        return 'written'
    monkeypatch.setattr(project_files, 'run_efs_unit', write)
    assert await project_files.upload_project_files('project', LOCAL_SCOPE, [upload]) == 'written'
    assert written == [b'original']

async def test_denied_upload_never_reaches_storage(monkeypatch):
    from cowork.services import files
    from cowork.db.scoped import LOCAL_SCOPE
    monkeypatch.setenv('COWORK_CONTENT_SAFETY_REQUIRED', 'true')
    store = AsyncMock()
    monkeypatch.setattr(files, 'run_efs', store)
    with pytest.raises(HTTPException):
        await files.store_file(scope=LOCAL_SCOPE, filename='image.png', content_type='image/png', data=b'PNG', purpose='test')
    store.assert_not_called()

async def test_project_text_read_is_checked_before_return(monkeypatch):
    from cowork.api.v1.endpoints import project_files
    from cowork.db.scoped import LOCAL_SCOPE
    monkeypatch.setenv('COWORK_CONTENT_SAFETY_REQUIRED', 'true')
    monkeypatch.setattr(project_files, 'run_efs_unit', AsyncMock(return_value={'content': 'denied text'}))
    check = AsyncMock(side_effect=HTTPException(422, 'blocked'))
    monkeypatch.setattr(safety, 'check_content', check)
    with pytest.raises(HTTPException):
        await project_files.read_project_file('project', 'note.txt', LOCAL_SCOPE, None)
    assert check.await_args.args == ('denied text',)

async def test_synthesized_historical_sources_are_checked_before_emission(monkeypatch):
    from types import SimpleNamespace
    from anton.core.llm.provider import StreamTextDelta
    from cowork.harnesses.anton_harness import stream_formatter as formatter
    monkeypatch.setenv('COWORK_CONTENT_SAFETY_REQUIRED', 'true')
    async def check(content, **kwargs):
        if 'denied historical label' in content:
            raise HTTPException(422, 'blocked')
    monkeypatch.setattr(safety, 'check_content', check)
    sources = SimpleNamespace(figures=[1], model_dump=lambda **kwargs: {'figures': ['denied historical label']})
    monkeypatch.setattr(formatter, 'find_answer_sources', AsyncMock(return_value=sources))
    async def events():
        yield StreamTextDelta(text='Safe answer')
    visible = []
    with pytest.raises(HTTPException):
        async for event in formatter.format_responses_stream(events(), model='test'):
            visible.append(event)
    assert not any('response.figure_sources' in item for item in visible)
    assert not any('denied historical label' in item for item in visible)
