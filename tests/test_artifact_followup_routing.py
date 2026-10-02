import pytest
from cowork.handlers import response_routing as routing
ROUTE=True

def tool_turn():
    return [{'role':'user','content':'Build'},
        {'role':'assistant','content':[{'type':'text','text':'I will read'},
            {'type':'tool_use','id':'x','name':'read_file','input':{}}]},
        {'role':'user','content':[{'type':'tool_result','tool_use_id':'x','content':'result'}]},
        {'role':'assistant','content':'Complete'}]


@pytest.mark.asyncio
@pytest.mark.parametrize('prompt', ['Update it', 'Actualízalo', '再確認してください'])
async def test_tool_followup_delegates_without_provider_call(monkeypatch, prompt):
    def forbidden(*args, **kw):
        raise AssertionError('Unnecessary routing model call')
    monkeypatch.setattr(routing, '_settings_binding', forbidden)
    monkeypatch.setattr(routing, '_gate', forbidden)
    decision = await routing.decide_route(history=tool_turn() + [{'role':'user','content':prompt}],
        has_non_text_input=False, has_attachments=False, has_disabled_connections=False)
    assert decision.route == routing.DELEGATED_AGENTIC
    assert decision.reason == 'prior_turn_ran_tools'
    assert decision.text == ''


def test_history_boundary_returns_to_normal_conversation():
    prompt = {'role':'user','content':'Hello'}
    assert routing.prior_turn_ran_tools(tool_turn() + [prompt])
    assert routing.prior_turn_ran_tools(tool_turn()[2:] + [prompt])
    assert not routing.prior_turn_ran_tools(tool_turn() + [prompt,
        {'role':'assistant','content':'Hello'}, prompt])
    assert not routing.prior_turn_ran_tools([prompt])
    assert not routing.prior_turn_ran_tools([{'role':'system','content':[{'type':'tool_use'}]},
        {'role':'user','content':'Hi'}, {'role':'assistant','content':'Hi'}, prompt])
    assert not routing.prior_turn_ran_tools([{'role':'user','content':'Hi'},
        {'role':'assistant','content':[None, {'type':'unknown','text':'tool_use'}]}, prompt])


@pytest.mark.asyncio
async def test_attachment_delegation_remains_unchanged(monkeypatch):
    monkeypatch.setattr(routing, '_settings_binding', lambda: (_ for _ in ()).throw(AssertionError()))
    decision = await routing.decide_route(history=tool_turn() + [{'role':'user','content':'Inspect'}],
        has_non_text_input=False, has_attachments=True, has_disabled_connections=False)
    assert decision.reason == 'attachments_present'
