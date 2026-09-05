"""The job payload must carry org-relative paths and no file blobs.

cowork-server sees the shared tree at <root>/<org_id>; the worker pod mounts
its own org's access point AT <root>. The two sit at different depths, so an
absolute path built on one side names nothing on the other.
"""
import json

import pytest

from cowork.turnqueue import producer as prod


class _StopEnqueue(Exception):
    pass


class _FakeRedis:
    def __init__(self, captured):
        self._captured = captured

    async def delete(self, *_a):
        return None

    async def sadd(self, *_a):
        return None

    async def xadd(self, _stream, fields):
        self._captured["payload"] = json.loads(fields["payload"])
        raise _StopEnqueue


@pytest.fixture
def captured(monkeypatch):
    box: dict = {}
    monkeypatch.setattr(prod, "get_redis", lambda: _FakeRedis(box))

    async def _fake_llm(**_kw):
        return {"provider": "minds-cloud", "api_key": "k", "base_url": "u"}

    monkeypatch.setattr(prod, "_mint_llm_block", _fake_llm)
    return box


async def _enqueue(**kwargs):
    base = dict(
        conversation_id="conv-1", org_id="org-1", user_id="user-1",
        input_text="hi", model="m",
    )
    base.update(kwargs)
    gen = prod.stream_remote_replies(**base)
    with pytest.raises(_StopEnqueue):
        async for _ in gen:
            pass


@pytest.mark.asyncio
async def test_workspace_path_is_org_relative(captured):
    await _enqueue(project_id="proj-1", workspace_rel_path="projects/general")
    params = captured["payload"]["params"]
    assert params["workspace_path"] == "projects/general"
    assert not params["workspace_path"].startswith("/")


@pytest.mark.asyncio
async def test_project_id_travels_on_the_job(captured):
    await _enqueue(project_id="proj-1", workspace_rel_path="projects/general")
    assert captured["payload"]["project_id"] == "proj-1"


@pytest.mark.asyncio
async def test_only_project_memory_is_shipped_in_anton_job_params(captured):
    """Global memory is already mounted per user; the conversation workspace
    cannot reach its project-level sibling, so Anton receives exactly that tier."""
    project_memory = {
        "rules": "Always cite the source.",
        "lessons": "Retries need idempotency.",
    }
    await _enqueue(
        project_id="proj-1",
        workspace_rel_path="projects/general",
        memory={
            "global": {"rules": "private preference"},
            "project": project_memory,
        },
    )
    params = captured["payload"]["params"]
    assert "skills" not in params
    assert params["memory"] == {"project": project_memory}


@pytest.mark.asyncio
async def test_empty_or_global_only_memory_is_omitted(captured):
    await _enqueue(memory={"global": {"rules": "private preference"}})
    assert "memory" not in captured["payload"]["params"]


@pytest.mark.asyncio
async def test_a_leading_slash_is_stripped_rather_than_shipped(captured):
    """Defence in depth: a caller passing an absolute-looking path must not put
    one on the wire, because the pod would join it under its own root."""
    await _enqueue(workspace_rel_path="/projects/general")
    assert captured["payload"]["params"]["workspace_path"] == "projects/general"
