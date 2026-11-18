"""Doc tools a remote turn's pod calls, run here: right after the write the
doc's owner, the turn's provenance and the folder for the turn end are
recorded, and only the doc's owner's agent may edit it."""
from __future__ import annotations

import asyncio
import json
import threading
from functools import partial
from uuid import UUID, uuid4

import fakeredis.aioredis
import pytest

from cowork.api.v1.endpoints import docs as docs_api
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import TenantScope
from cowork.db.units import DatabaseBusy, on_efs_thread, unit_session
from cowork.docs import remote as remote_mod
from cowork.docs import tools as doc_tools
from cowork.docs.hub import doc_topic, hub
from cowork.docs.model import DocError
from cowork.docs.remote import owner_check, remote_doc_context, start_remote_doc_tools
from cowork.docs.service import create_doc
from cowork.docs.store import DOC_FILENAME
from cowork.docs.tools import run_doc_tool
from cowork.services import artifact_ownership as ownership
from cowork.turnqueue.redis_client import answer_queue_key
from test_artifact_ownership import make_conversation, make_project, project_root

pytestmark = pytest.mark.usefixtures("cleanup_tmp_projects")

CALL = "htool:" + "cd" * 16


async def _emit(_text: str) -> None:
    return None


def _member(project, user: str) -> TenantScope:
    return TenantScope(org_mode=True, org_id=project.org_id, user_id=user)


def _context(project, scope):
    source = project_root(project)
    return remote_doc_context(scope=scope, conversation_id=str(uuid4()), project_id=source.project_id,
                              project_name=project.name, artifacts_base=source.base, files=None)


def _tools(project, scope, conversation_id, creator, redis, correlation_id="corr-remote"):
    source = project_root(project)
    return start_remote_doc_tools(
        scope=scope, correlation_id=correlation_id, conversation_id=str(conversation_id), turn_index=1,
        artifacts_base=source.base, project_id=source.project_id, project_name=project.name,
        creator=creator, files=None, r=redis,
    )


async def test_a_members_agent_cannot_edit_another_members_doc_but_reads_and_lists_it(tmp_path):
    alice, bob = str(uuid4()), str(uuid4())
    project = make_project(tmp_path, str(uuid4()))
    source = project_root(project)
    made = create_doc(source.base, title="Alice plan", markdown="Lead.", by="user")
    with unit_session(scope=_member(project, alice)) as session:
        ownership.record_artifact_owner(session, source.project_id, made.slug, alice, action="create")
    before = (made.folder / DOC_FILENAME).read_bytes()
    edit = {"doc": "Alice plan", "ops": [{"op": "edit_text", "find": "Lead", "replace": "Led"}]}

    bobs = _context(project, _member(project, bob))
    refused = await run_doc_tool(bobs, "edit_doc", edit, _emit)
    assert not refused.ok and refused.text.startswith("Refused (not_owner)")
    assert (made.folder / DOC_FILENAME).read_bytes() == before
    read = await run_doc_tool(bobs, "read_doc", {"doc": "Alice plan"}, _emit)
    listed = await run_doc_tool(bobs, "list_docs", {}, _emit)
    assert read.ok and "Lead." in read.text
    assert listed.ok and "“Alice plan”" in listed.text

    alices = _context(project, _member(project, alice))
    assert (await run_doc_tool(alices, "edit_doc", edit, _emit)).ok


async def test_a_remote_call_does_its_file_work_on_the_file_stores_threads(tmp_path):
    alice = str(uuid4())
    project = make_project(tmp_path, str(uuid4()))
    context = _context(project, _member(project, alice))
    checked: list[bool] = []
    written: list[bool] = []
    real_check = context.owner_check

    def check(folder):
        checked.append(on_efs_thread())
        real_check(folder)

    context.owner_check = check
    made = await run_doc_tool(context, "create_doc", {"title": "Plan", "markdown": "Lead."}, _emit,
                              on_written=lambda slug, tool: written.append(on_efs_thread()))
    assert made.ok and written == [True]
    with unit_session(scope=_member(project, alice)) as session:
        ownership.record_artifact_owner(session, project_root(project).project_id, made.touched_slug, alice,
                                        action="create")
    edited = await run_doc_tool(context, "edit_doc", {
        "doc": "Plan", "ops": [{"op": "edit_text", "find": "Lead", "replace": "Led"}]}, _emit)
    assert edited.ok and checked == [True]


def test_a_busy_database_refuses_the_edit_as_busy_not_as_maybe_applied(tmp_path, monkeypatch):
    def busy(*, scope):
        raise DatabaseBusy("no database connection freed within POOL_TIMEOUT")

    monkeypatch.setattr(remote_mod, "unit_session", busy)
    check = owner_check(scope=TenantScope(org_mode=True, org_id=str(uuid4()), user_id=str(uuid4())),
                        project_id=str(uuid4()), artifacts_base=tmp_path)
    with pytest.raises(DocError) as err:
        check(tmp_path / "plan")
    assert err.value.code == "busy" and "nothing was changed" in err.value.message


async def test_a_call_cancelled_after_its_write_still_leaves_owner_provenance_and_slug(tmp_path, monkeypatch):
    alice = str(uuid4())
    project = make_project(tmp_path, str(uuid4()))
    conversation_id = make_conversation(project, alice)
    source = project_root(project)
    scope = _member(project, alice)
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    tools = _tools(project, scope, conversation_id, alice, redis, correlation_id="corr-k2")
    entered, release = threading.Event(), threading.Event()
    real_create = doc_tools.create_doc

    def slow_create(*args, **kwargs):
        entered.set()
        release.wait(5)
        return real_create(*args, **kwargs)

    monkeypatch.setattr(doc_tools, "create_doc", slow_create)
    tools.runner.start({"step": "host_tool_call", "call_id": CALL, "tool": "create_doc",
                        "input": {"title": "Plan", "markdown": "Lead."}})
    assert await asyncio.to_thread(entered.wait, 5)
    # The call runs on run_efs: the cancel is delivered now, and the task ends
    # only after the thread returns, which is what keeps the write's follow-ups.
    for task in tools.runner.pending_tasks():
        task.cancel()
    tools.runner.close()
    release.set()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    while not tools.writes.slugs():
        assert loop.time() < deadline, "the worker thread never recorded the doc"
        await asyncio.sleep(0.01)
    (slug,) = tools.writes.slugs()
    assert tools.writes.slug_for(CALL) == slug
    with unit_session(scope=scope) as session:
        owner = ownership.resolve_artifact_owner(session, source, slug)
    assert owner == ownership.OwnerResolution(alice, "recorded")
    assert ownership.provenance_origin(source.base / slug) == conversation_id
    kinds = [json.loads(e)["kind"] for e in await redis.lrange(answer_queue_key("corr-k2"), 0, -1)]
    assert "host_tool_result" not in kinds


class _Open:
    async def is_disconnected(self):
        return False


async def test_the_person_edits_a_doc_the_agent_just_made_before_the_turn_end_indexes_it(
    tmp_path, monkeypatch, granted_product_permissions,
):
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    try:
        alice = str(uuid4())
        project = make_project(tmp_path, str(uuid4()))
        conversation_id = make_conversation(project, alice)
        source = project_root(project)
        scope = _member(project, alice)
        tools = _tools(project, scope, conversation_id, alice, fakeredis.aioredis.FakeRedis(decode_responses=True))
        made = await run_doc_tool(tools.context, "create_doc", {"title": "Plan", "markdown": "Lead."}, _emit,
                                  on_written=partial(tools.writes.written, CALL))
        assert made.ok
        folder = source.base / made.touched_slug
        artifact_id = UUID(str(json.loads((folder / "metadata.json").read_text())["id"])).hex

        body = docs_api._OpsBody(ops=[{"op": "edit_text", "find": "Lead", "replace": "Led"}])
        edited = await docs_api.edit_doc(str(project.id), artifact_id, body, scope)
        assert edited["doc"]["rev"] == 2
        got = await docs_api.get_doc(str(project.id), artifact_id, scope)
        assert got["capabilities"]["canEdit"] is True
        stream = (await docs_api.doc_events(str(project.id), artifact_id, _Open(), scope)).body_iterator
        try:
            first = await asyncio.wait_for(anext(stream), 5)
        finally:
            await stream.aclose()
        assert first.startswith("event: doc.ready")
    finally:
        get_app_settings.cache_clear()


async def test_an_agent_edit_reaches_the_editor_on_the_folder_the_docs_api_resolves(
    tmp_path, monkeypatch, granted_product_permissions,
):
    """Hub topics are folder paths (doc_topic): the remote adapter announces on
    the folder spelled exactly as the docs API's _resolve spells it, or an open
    editor, even on this replica, never hears the agent's edit."""
    monkeypatch.setenv("COWORK_TENANCY_MODE", "org")
    get_app_settings.cache_clear()
    try:
        alice = str(uuid4())
        project = make_project(tmp_path, str(uuid4()))
        conversation_id = make_conversation(project, alice)
        source = project_root(project)
        scope = _member(project, alice)
        tools = _tools(project, scope, conversation_id, alice, fakeredis.aioredis.FakeRedis(decode_responses=True))
        made = await run_doc_tool(tools.context, "create_doc", {"title": "Plan", "markdown": "Lead."}, _emit,
                                  on_written=partial(tools.writes.written, CALL))
        folder = source.base / made.touched_slug
        artifact_id = UUID(str(json.loads((folder / "metadata.json").read_text())["id"])).hex
        _source, resolved, _metadata, _is_own = await docs_api._resolve(
            scope, str(project.id), artifact_id, write=False)
        queue = hub.subscribe(doc_topic(resolved))
        try:
            edited = await run_doc_tool(tools.context, "edit_doc", {
                "doc": "Plan", "ops": [{"op": "edit_text", "find": "Lead", "replace": "Led"}]}, _emit,
                on_written=partial(tools.writes.written, CALL))
            assert edited.ok
            event = await asyncio.wait_for(queue.get(), 2)
        finally:
            hub.unsubscribe(doc_topic(resolved), queue)
        assert event["type"] == "doc.updated" and event["by"] == "agent" and event["rev"] == 2
    finally:
        get_app_settings.cache_clear()
