"""Isolated app for the real Uvicorn shutdown/restart regression.

Only the external harness and unrelated startup/teardown services are replaced.
The app lifespan, detached producer, stream file and SQLite persistence are real.
Loaded in a child process; never imported by the pytest process.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import pkgutil
from pathlib import Path
from types import SimpleNamespace

from fastapi.responses import StreamingResponse
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
from cowork.streaming import TurnLifecycle, new_buffer, registry, sse_frame


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
    payload = {"type": "response.output_text.delta", "delta": "persist this partial answer"}
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

app = server.app


@app.get("/__shutdown_test/ready")
async def ready():
    return {"ready": True}


@app.get("/__shutdown_test/start")
async def start():
    session = ScopedSession(get_open_session(), SYSTEM_SCOPE)
    try:
        conversation = ConversationService(session).create_conversation(
            "process shutdown regression", project_id=GENERAL_PROJECT_ID,
        )
        conversation_id = conversation.id
    finally:
        session.close()
    buffer = new_buffer(str(conversation_id), 0)
    lifecycle = TurnLifecycle()
    handler = object.__new__(responses.ResponsesHandler)
    handler.principal = None  # local, single-process deployment
    await registry.start(
        conversation_id=str(conversation_id), turn_id=0, buffer=buffer,
        lifecycle=lifecycle,
        producer_coro=handler._run_turn(
            conv_id=conversation_id, harness_input=[], original_content="hello",
            model="anton", disabled=None, harness_name="anton", harness_id="anton",
            buffer=buffer, lifecycle=lifecycle,
        ),
    )

    async def tail():
        async for record in buffer.tail():
            if record.type == "sse":
                yield record.data["sse"]

    return StreamingResponse(
        tail(), media_type="text/event-stream",
        headers={"X-Test-Conversation-Id": str(conversation_id)},
    )
