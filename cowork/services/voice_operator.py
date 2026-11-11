"""The voice call's operator (ENG-4138, prototype): requests in, app actions out.

GPT-Live hands every request back to the web app with no task text and calls no
tools of ours (client delegation). The web app sends the request here, with a
snapshot of what is on screen (projects, recent tasks, the open task); this asks
a fast MindsHub model, as the caller, which of the app's actions to take. The
web app runs the actions it is told to, sends their results back, and calls
again until the model answers in words, which GPT-Live then says.

The actions are fixed here (``TOOLS``) and run in the web app, as the person,
through the same code their clicks use: nothing here changes a project or a
task. Work itself (data, research, writing, the browser) goes to Anton through
``ask_anton``, in a task the person can watch.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

import httpx

from cowork.common.settings.app_settings import default_turn_minds_api_host, get_app_settings
from cowork.services.providers import minds_chat_base_url
from cowork.services.voice import VOICE_STYLES, local_now

logger = logging.getLogger(__name__)

_TIMEOUT_S = 30.0
MAX_MESSAGES = 60
MAX_SNAPSHOT_CHARS = 12_000

OPERATOR_PROMPT = (
    "You run MindsHub Cowork for a person on a voice call with Anton, their AI coworker. Each user message is "
    "what they asked for, from their speech transcript, so it can be misheard. Do it with the tools, then "
    "answer in one to three short spoken sentences: no lists, links, ids or markdown.\n\n"
    "- Projects hold tasks. A task is a conversation with Anton.\n"
    "- Work (data, files, research, writing, connected apps, websites) goes to Anton with ask_anton, in the "
    "task on screen unless they ask for a new one or a different one. To see or act on the browser, ask Anton "
    "in the task that uses it, and call show_browser so they can watch.\n"
    "- Before starting a new task, know what it is for. When they ask for a new task without saying what it "
    "should do (\"start a new task\", \"open a task for me\"), don't call ask_anton: ask in one short question "
    "what it should be about, and start it once they say.\n"
    "- To stop, cancel or pause work, call stop_task on the running task they mean: the one this call last gave "
    "Anton work in, or else the one on screen. Ask which only when several are running and nothing says which.\n"
    "- Find tasks and projects by name in the snapshot. When a name is ambiguous, ask which one instead of "
    "guessing. Never invent an id.\n"
    "- Anton does one thing at a time for this call. While the snapshot says he is busy, never start other work "
    "beside it. A question about that work (how it is going, what he has found): answer from task_steps. More "
    "work: say in a few words what he is busy with, then either queue it with queue_for_anton when it can wait "
    "or adds to that work, or, when it would replace that work or they sound like it can't wait, ask whether to "
    "stop the current work and do this now; if they say yes, stop_task and then ask_anton.\n"
    "- To change Anton's voice or accent on the call, call change_voice with the voice that fits what they "
    "asked (a British voice: vesper). The call reconnects in that voice, so say nothing after it.\n"
    "- Pass dates and times to Anton as they said them (\"the weekend after Thanksgiving\", \"next Friday\"): "
    "don't turn them into exact dates. When you need a date yourself, count from today's date below.\n"
    "- Say what you did, from the tool results. Never say something is done when a tool failed.\n\n"
)


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False},
        },
    }


_TASK_ID = {"type": "string", "description": "A task id from the snapshot."}

TOOLS: list[dict[str, Any]] = [
    _tool(
        "ask_anton",
        "Give Anton work and wait for the answer. Starts a new task when task_id is 'new'. Long work keeps going "
        "in the task after this returns.",
        {
            "request": {"type": "string", "description": "The request, in the person's words, complete."},
            "task_id": {"type": "string", "description": "A task id from the snapshot, or 'new'."},
            "project": {"type": "string", "description": "For a new task: the project to start it in."},
        },
        ["request", "task_id"],
    ),
    _tool(
        "queue_for_anton",
        "Queue work for Anton behind what he is busy with, in that task or another; it starts when the current "
        "work ends, and the person hears when it is done. Returns at once.",
        {
            "request": {"type": "string", "description": "The request, in the person's words, complete."},
            "task_id": {"type": "string", "description": "A task id from the snapshot: the busy task's, to run right after it."},
        },
        ["request", "task_id"],
    ),
    _tool("create_project", "Create a project.", {"name": {"type": "string"}}, ["name"]),
    _tool(
        "move_task",
        "Move a task, and the files it made, to a project. Creates the project when it doesn't exist.",
        {"task_id": _TASK_ID, "project": {"type": "string", "description": "The project's name."}},
        ["task_id", "project"],
    ),
    _tool("open_task", "Show a task on the screen.", {"task_id": _TASK_ID}, ["task_id"]),
    _tool("open_project", "Show a project on the screen.", {"project": {"type": "string"}}, ["project"]),
    _tool(
        "task_steps",
        "What Anton did and is doing in a task: its steps, newest last, and its latest answer.",
        {"task_id": _TASK_ID},
        ["task_id"],
    ),
    _tool("stop_task", "Stop the work running in a task.", {"task_id": _TASK_ID}, ["task_id"]),
    _tool(
        "show_browser",
        "Show the browser Anton used in a task, so the person sees what Anton sees.",
        {"task_id": _TASK_ID},
        ["task_id"],
    ),
    _tool(
        "change_voice",
        "Switch Anton's voice on this call. Voices: "
        + "; ".join(f"{name} ({style})" for name, style in VOICE_STYLES.items())
        + ".",
        {"voice": {"type": "string", "enum": sorted(VOICE_STYLES)}},
        ["voice"],
    ),
]
TOOL_NAMES = frozenset(t["function"]["name"] for t in TOOLS)


class OperatorError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def chat_url() -> str:
    return f"{minds_chat_base_url(default_turn_minds_api_host())}/chat/completions"


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(_TIMEOUT_S))


def _clean(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The call's history as the web app keeps it: user, assistant and tool turns only."""
    kept: list[dict[str, Any]] = []
    for m in messages[-MAX_MESSAGES:]:
        role = m.get("role")
        if role == "user" or (role == "assistant" and not m.get("tool_calls")):
            kept.append({"role": role, "content": str(m.get("content") or "")})
        elif role == "assistant":
            calls = [
                {"id": str(c.get("id")), "type": "function",
                 "function": {"name": str(c["function"]["name"]), "arguments": str(c["function"].get("arguments") or "{}")}}
                for c in m["tool_calls"]
                if isinstance(c, dict) and isinstance(c.get("function"), dict) and c["function"].get("name") in TOOL_NAMES
            ]
            if calls:
                kept.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": calls})
        elif role == "tool":
            kept.append({"role": "tool", "tool_call_id": str(m.get("tool_call_id")), "content": str(m.get("content") or "")})
    # A window can open on tool results whose call fell off the front.
    while kept and kept[0]["role"] == "tool":
        kept.pop(0)
    return kept


def system_prompt(snapshot: str, tz_name: str | None, now: datetime | None = None) -> str:
    """The instructions, the person's date and time (an IANA zone from their
    browser), and what is on their screen."""
    return (
        f"{OPERATOR_PROMPT}It is now {local_now(tz_name, now)} where the user is.\n\n"
        f"What is on the screen now:\n{snapshot[:MAX_SNAPSHOT_CHARS]}"
    )


def request_body(messages: list[dict[str, Any]], snapshot: str, tz_name: str | None = None) -> dict[str, Any]:
    return {
        "model": get_app_settings().voice_operator_model,
        "messages": [{"role": "system", "content": system_prompt(snapshot, tz_name)}, *_clean(messages)],
        "tools": TOOLS,
        "tool_choice": "auto",
    }


async def operate(bearer: str, messages: list[dict[str, Any]], snapshot: str, tz_name: str | None = None) -> dict[str, Any]:
    """The model's next turn: ``{content, tool_calls}``, with only our tools' calls."""
    if not bearer:
        raise OperatorError(401, "unauthorized", "Sign in to MindsHub.")
    try:
        async with _client() as client:
            response = await asyncio.wait_for(
                client.post(chat_url(), json=request_body(messages, snapshot, tz_name), headers={"Authorization": f"Bearer {bearer}"}),
                _TIMEOUT_S,
            )
    except Exception as exc:
        logger.warning("voice operator unreachable", extra={"event": "voice_operator_unreachable", "error_type": type(exc).__name__})
        raise OperatorError(424, "unreachable", "Couldn't reach MindsHub.") from exc
    if response.status_code >= 400:
        logger.warning("voice operator refused", extra={"event": "voice_operator_refused", "status_code": response.status_code})
        raise OperatorError(424, "refused", "MindsHub couldn't take the request.")
    try:
        message = response.json()["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise OperatorError(424, "bad_answer", "MindsHub answered without a reply.") from exc
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name") not in TOOL_NAMES:
            continue
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except ValueError:
            args = {}
        calls.append({"id": call.get("id"), "name": fn["name"], "arguments": args if isinstance(args, dict) else {}})
    return {"content": message.get("content") or "", "tool_calls": calls}
