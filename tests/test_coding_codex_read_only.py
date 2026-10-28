"""Read only against the real Codex app-server.

The approval tests elsewhere use a fake engine and check the decision the
broker returns. These run the bundled Codex binary, with a scripted
Responses-API model behind the inference URL, and check the files on disk:
Codex runs an accepted request outside the read-only sandbox, so only the
broker's refusal keeps a Read only task from writing.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from coding_service_fakes import CREDS, repository
from cowork.coding.contracts import (
    ApprovalDecision,
    EventType,
    PermissionMode,
    SessionCreateRequest,
    SessionStatus,
)
from cowork.coding.engines import codex_config
from cowork.coding.service import CodingService

PATCH = "*** Begin Patch\n*** Update File: README.md\n@@\n base\n+hello\n*** End Patch\n"


def escalated_append(call_id: str) -> dict:
    return _function_call(call_id, {
        "cmd": "printf 'hello\\n' >> README.md",
        "sandbox_permissions": "require_escalated",
        "justification": "Append hello to README.md",
    })


def rememberable_edit(call_id: str) -> dict:
    # A literal command with a proposed prefix, so the card offers Always allow.
    return _function_call(call_id, {
        "cmd": "sed -i.bak 's/base/base hello/' README.md",
        "sandbox_permissions": "require_escalated",
        "justification": "Append hello to README.md",
        "prefix_rule": ["sed"],
    })


def patch_readme(call_id: str) -> dict:
    return {"type": "custom_tool_call", "id": f"ctc_{call_id}", "call_id": call_id, "name": "apply_patch", "input": PATCH}


def _function_call(call_id: str, arguments: dict) -> dict:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": "exec_command",
        "arguments": json.dumps(arguments),
    }


class ScriptedModel:
    """Answer each Responses request with the next scripted output items."""

    def __init__(self, script: list[list[dict]]) -> None:
        replies = list(script)
        lock = threading.Lock()
        tool_outputs: list[str] = []
        self.tool_outputs = tool_outputs

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802 - the model catalog request
                self._send("application/json", b'{"object":"list","data":[]}')

            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
                with lock:
                    tool_outputs.extend(
                        str(item.get("output")) for item in body.get("input", [])
                        if item.get("type") in {"function_call_output", "custom_tool_call_output"}
                    )
                    items = replies.pop(0) if replies else [{
                        "type": "message", "role": "assistant", "id": "msg_done",
                        "content": [{"type": "output_text", "text": "Finished.", "annotations": []}],
                    }]
                events = [{"type": "response.created", "response": {"id": "resp"}}]
                events += [{"type": "response.output_item.done", "item": item} for item in items]
                events.append({"type": "response.completed", "response": {"id": "resp", "usage": {
                    "input_tokens": 1, "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 2,
                }}})
                self._send("text/event-stream", b"".join(
                    f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events
                ))

            def _send(self, content_type: str, body: bytes) -> None:
                self.send_response(200)
                self.send_header("content-type", content_type)
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def run_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: PermissionMode,
    decision: ApprovalDecision,
    requests: list[dict],
) -> tuple[CodingService, str, Path, list[bool]]:
    """Run one turn whose model makes ``requests``, answering every card with ``decision``."""
    model = ScriptedModel([[item] for item in requests])
    monkeypatch.setattr(codex_config, "local_inference_base_url", lambda _port: model.url)
    service = CodingService(tmp_path / "coding")
    cards: list[bool] = []
    try:
        task = service.create_session(
            SessionCreateRequest(
                path=str(repository(tmp_path)),
                prompt="Add the line hello to the end of README.md",
                permission_mode=mode,
            ),
            CREDS, "codex", "gpt-5.5",
        )
        answered: set[str] = set()
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            session = service.get_session(task.id)
            pending = session.pending_approval
            if pending and pending.id not in answered:
                answered.add(pending.id)
                cards.append(pending.allow_session)
                service.resolve_approval(task.id, pending.id, decision)
            if session.status in {SessionStatus.completed, SessionStatus.failed}:
                break
            time.sleep(0.05)
        session = service.get_session(task.id)
        assert session.status == SessionStatus.completed, session.last_error
        if any("fs sandbox helper failed" in output for output in model.tool_outputs):
            # On Linux, Codex verifies a patch inside its bubblewrap sandbox
            # before asking. Without unprivileged user namespaces (GitHub's
            # Ubuntu runners, Docker) that fails, so no request is ever made.
            pytest.skip("Codex's Linux sandbox cannot start here, so the request never reaches approval")
        return service, task.id, Path(session.workspace_path), cards
    finally:
        service.close_all()
        model.close()


def git_status(workspace: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=workspace, capture_output=True, text=True, check=True,
    ).stdout


@pytest.mark.parametrize("decision, requests", [
    (ApprovalDecision.approve_once, [escalated_append("call-1")]),
    (ApprovalDecision.approve_once, [patch_readme("call-1")]),
    (ApprovalDecision.approve_session, [rememberable_edit("call-1"), rememberable_edit("call-2")]),
], ids=["command, allow once", "patch, allow once", "rememberable command twice, always allow"])
def test_read_only_task_leaves_files_unchanged_whatever_the_answer(tmp_path, monkeypatch, decision, requests):
    service, task_id, workspace, cards = run_task(tmp_path, monkeypatch, PermissionMode.read_only, decision, requests)

    assert cards == []
    assert (workspace / "README.md").read_text() == "base\n"
    assert git_status(workspace) == ""
    notices = [event for event in service.events(task_id).items if event.type == EventType.command_result]
    assert [notice.title for notice in notices] == ["Declined: this task is Read only"] * len(requests)


def test_ask_first_task_still_applies_an_allowed_change(tmp_path, monkeypatch):
    _service, _task_id, workspace, cards = run_task(
        tmp_path, monkeypatch, PermissionMode.supervised, ApprovalDecision.approve_once, [escalated_append("call-1")],
    )

    assert cards == [False]
    assert (workspace / "README.md").read_text() == "base\nhello\n"
