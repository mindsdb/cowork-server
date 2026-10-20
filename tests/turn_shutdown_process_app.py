"""Isolated app for the real Uvicorn shutdown/restart regression.

Only the external harness and unrelated startup/teardown services are replaced.
The app lifespan, detached producer, stream file and SQLite persistence are real.
For a remote turn, so are the relay, the Redis job, reply and turn streams, and
the save; the controller and the auth service are the stand-ins below.
Loaded in a child process; never imported by the pytest process.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import pkgutil
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from fastapi.responses import StreamingResponse
from fastapi import Request
from sqlmodel import SQLModel

import cowork.handlers.responses as responses
import cowork.models as models
import cowork.server as server
from cowork.common.settings.app_settings import get_app_settings
from cowork.db.scoped import SYSTEM_SCOPE, ScopedSession
from cowork.db.session import get_engine, get_open_session
from cowork.models.project import Project
from cowork.services.conversations import ConversationService
from cowork.services.projects import GENERAL_PROJECT, GENERAL_PROJECT_ID
from cowork.common.settings.app_settings import TurnQueueSettings
from cowork.streaming import TurnLifecycle, new_buffer, registry, sse_frame
from cowork.streaming.cancellation import cancel_disconnected_run
from cowork.streaming.disconnect import DisconnectStreamingResponse, DisconnectWatch
from cowork.streaming.registry import RunIdentity
from cowork.turnqueue import producer
from cowork.turnqueue.models import TurnJob, TurnReply
from cowork.turnqueue.redis_client import get_redis

PARTIAL = "persist this partial answer"


def _seed_schema() -> None:
    # The same schema setup as conftest.py, in this process's private database.
    for _, name, _ in pkgutil.iter_modules(models.__path__):
        importlib.import_module(f"cowork.models.{name}")
    SQLModel.metadata.create_all(get_engine(get_app_settings().database.uri))
    with get_open_session() as session:
        if session.get(Project, GENERAL_PROJECT_ID) is None:
            project_dir = Path(os.environ["COWORK_PROJECTS_DIR"]) / GENERAL_PROJECT
            project_dir.mkdir(parents=True, exist_ok=True)
            session.add(Project(id=GENERAL_PROJECT_ID, name=GENERAL_PROJECT, path=str(project_dir)))
            session.commit()


async def _noop():
    return None


async def _start_channels(app):
    app.state.channel_ingress = SimpleNamespace(stop_all=_noop)
    app.state.channel_adapters = SimpleNamespace(shutdown=_noop)


async def _formatter(stream, model, event_sink):
    yield sse_frame("response.created", {"type": "response.created"})
    payload = {"type": "response.output_text.delta", "delta": PARTIAL}
    event_sink(payload["type"], payload)
    yield sse_frame(payload["type"], payload)
    # Keeps both producer and response live when the parent sends its signal.
    await asyncio.Event().wait()


server.run_dev_setup = _seed_schema
server._warm_model_map_on_boot = _noop
server.start_scheduler = lambda: None
server._start_channels = _start_channels
responses.get_harness = lambda name: SimpleNamespace(
    stream_response=lambda **kwargs: None, formatter=_formatter,
)

# Nothing in the scenario starts coding engines, channels or scratchpads.
# Avoid instantiating those unrelated services merely to tear them down.
import cowork.coding.service as coding_service  # noqa: E402

coding_service.get_coding_service = lambda: SimpleNamespace(close_all=lambda: None)


# The auth service's part of a remote turn: the turn key and the connections
# the turn may use. Neither is reachable here, and neither is under test.
async def _turn_key(**kwargs):
    return {"provider": "minds-cloud", "api_key": "turn-key", "base_url": "http://127.0.0.1:9/v1"}


async def _no_connections(**kwargs):
    return None


producer._mint_llm_block = _turn_key
producer._mint_oauth_block = _no_connections

# Held because the event loop keeps only a weak reference to a task.
_controllers: set[asyncio.Task] = set()


async def _controller_that_never_finishes(conversation_id: str) -> None:
    """Stands in for scratchpad-controller: takes the turn's job from its
    Redis stream, replies with the partial answer, and never sends a terminal."""
    r = get_redis()
    jobs = f"{TurnQueueSettings().jobs_stream}:{conversation_id}"
    while not (queued := await r.xread({jobs: "0-0"}, count=1, block=1000)):
        pass
    job = TurnJob.model_validate_json(queued[0][1][0][1]["payload"])
    for kind, data in (
        ("progress", {"phase": "workspace_authorized", "workspace_mode": job.params["workspace_mode"]}),
        ("turn_delta", {"text": PARTIAL}),
    ):
        reply = TurnReply(correlation_id=job.correlation_id, kind=kind, data=data)
        await r.xadd(job.reply_stream, {"payload": reply.model_dump_json()})


app = server.app


def _new_conversation():
    session = ScopedSession(get_open_session(), SYSTEM_SCOPE)
    try:
        return ConversationService(session).create_conversation(
            "process shutdown regression", project_id=GENERAL_PROJECT_ID,
        ).id
    finally:
        session.close()


@app.get("/__shutdown_test/ready")
async def ready():
    return {"ready": True}


@app.get("/__shutdown_test/start")
async def start(request: Request, cancel_on_disconnect: bool = False):
    conversation_id = _new_conversation()
    lifecycle = TurnLifecycle()
    handler = object.__new__(responses.ResponsesHandler)
    handler.principal = None  # local, single-process deployment
    handle = await registry.start(
        conversation_id=str(conversation_id), turn_id=0,
        open_buffer=partial(new_buffer, str(conversation_id), 0),
        lifecycle=lifecycle,
        produce=lambda buffer: handler._run_turn(
            conv_id=conversation_id, harness_input=[], original_content="hello",
            model="anton", disabled=None, harness_name="anton", harness_id="anton",
            buffer=buffer, lifecycle=lifecycle,
        ),
    )

    async def tail():
        async for record in handle.buffer.tail():
            if record.type == "sse":
                yield record.data["sse"]

    response_type = DisconnectStreamingResponse if cancel_on_disconnect else StreamingResponse
    kwargs = {
        "watch": DisconnectWatch(request.receive),
        "on_disconnect": lambda: cancel_disconnected_run(handle),
    } if cancel_on_disconnect else {}
    return response_type(
        tail(), media_type="text/event-stream",
        **kwargs,
        headers={"X-Test-Conversation-Id": str(conversation_id)},
    )


@app.get("/__shutdown_test/start_remote")
async def start_remote():
    """A remote turn as handle() starts one, streamed as POST /responses
    streams it, with the controller above on the other end."""
    conversation_id = _new_conversation()
    lifecycle = TurnLifecycle(identity=RunIdentity(remote=True))
    handler = responses.ResponsesHandler(principal=None)
    handle = await registry.start(
        conversation_id=str(conversation_id), turn_id=0,
        open_buffer=partial(new_buffer, str(conversation_id), 0),
        lifecycle=lifecycle,
        produce=lambda buffer: handler._produce_remote(
            conv_id=conversation_id, input_text="hello", original_content="hello",
            model="test-model", harness_id="anton", buffer=buffer, lifecycle=lifecycle,
        ),
    )
    controller = asyncio.create_task(_controller_that_never_finishes(str(conversation_id)))
    _controllers.add(controller)
    controller.add_done_callback(_controllers.discard)
    return StreamingResponse(
        responses.sse_from_buffer(handle.buffer, 0), media_type="text/event-stream",
        headers={
            "X-Test-Conversation-Id": str(conversation_id),
            "X-Test-Correlation-Id": lifecycle.identity.correlation_id,
        },
    )
