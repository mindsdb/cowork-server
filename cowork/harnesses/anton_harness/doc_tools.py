"""The agent's doc tools for a turn this server runs: anton ToolDefs around
cowork.docs.tools.run_doc_tool, the same executor a remote turn's host tools use.

Each step the executor emits streams as a ToolProgress, so the chat shows the
call as a labelled step, then the result as the ToolOutcome. A doc the call
writes is marked as this turn's (anton's _track_artifact) in the worker
thread, right after the write, so the card built after it sees the turn's
provenance.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from anton.core.artifacts import ArtifactStore
from anton.core.tools.progress import ToolProgress
from anton.core.tools.registry import ToolOutcome
from anton.core.tools.tool_defs import ToolDef
from anton.core.tools.tool_handlers import _track_artifact

from cowork.docs.prompt import DOC_PROMPT
from cowork.docs.tools import (
    DOC_TOOL_NAMES,
    DOC_TOOL_SPECS,
    DocToolContext,
    DocToolResult,
    OnWritten,
    edit_label,
    run_doc_tool,
)

logger = logging.getLogger(__name__)

__all__ = ["DOC_TOOL_NAMES", "DocToolContext", "build_doc_tools", "edit_label"]

_TRACK_SUMMARY = {"create_doc": "Created doc", "edit_doc": "Edited doc"}


def _track(session: Any, artifacts_dir: Path, slug: str, summary: str) -> None:
    """Mark the doc as touched this turn, so its card lands on this turn's answer.

    The same call anton's own artifact tools make; anton has no public one.
    """
    try:
        _track_artifact(session, ArtifactStore(artifacts_dir), slug, summary=summary)
    except Exception:
        logger.debug("could not record doc %s as touched", slug, exc_info=True)


async def _stream(
    ctx: DocToolContext, name: str, tc_input: dict, artifacts_dir: Path, on_written: OnWritten,
) -> AsyncIterator[str | DocToolResult]:
    """run_doc_tool's steps as they come, then its result."""
    steps: asyncio.Queue[str] = asyncio.Queue()

    async def emit(text: str) -> None:
        steps.put_nowait(text)

    task = asyncio.create_task(
        run_doc_tool(ctx, name, tc_input, emit, artifacts_dir=artifacts_dir, on_written=on_written)
    )
    step: asyncio.Future | None = None
    try:
        while True:
            step = asyncio.ensure_future(steps.get())
            done, _pending = await asyncio.wait({task, step}, return_when=asyncio.FIRST_COMPLETED)
            if step in done:
                yield step.result()
                continue
            step.cancel()
            while not steps.empty():
                yield steps.get_nowait()
            yield task.result()
            return
    finally:
        if step is not None and not step.done():
            step.cancel()
        if not task.done():
            task.cancel()


def _handler(ctx: DocToolContext, name: str):
    async def handle(session: Any, tc_input: dict) -> Any:
        artifacts_dir = ctx.artifacts_dir(session)
        summary = _TRACK_SUMMARY.get(name)

        def written(slug: str, _tool: str) -> None:
            if summary is not None:
                _track(session, artifacts_dir, slug, summary)

        async for item in _stream(ctx, name, tc_input, artifacts_dir, written):
            if isinstance(item, DocToolResult):
                yield ToolOutcome(content=item.text, ok=item.ok)
            else:
                yield ToolProgress(text=item)

    return handle


def build_doc_tools(ctx: DocToolContext) -> list[ToolDef]:
    return [
        ToolDef(name=name, description=description, input_schema=schema, handler=_handler(ctx, name),
                **({"prompt": DOC_PROMPT} if name == "create_doc" else {}))
        for name, description, schema in DOC_TOOL_SPECS
    ]
