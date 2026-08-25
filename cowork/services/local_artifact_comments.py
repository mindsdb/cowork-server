"""File-backed artifact comments for single-user Desktop workspaces."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from cowork.services.artifact_identity import ArtifactIdentityConflict, resolve_artifact_folder
from cowork.services.artifact_lock import artifact_lock
from cowork.services.artifact_roots import artifacts_sources_for_scan

_VIEWER = {"user_id": "desktop-owner", "email": "You", "role": "owner"}
_CAPABILITIES = {
    "canComment": True,
    "canResolve": True,
    "canAddressWithAgent": True,
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class LocalArtifactComments:
    """Small atomic journal with the same response shape as inference comments."""

    def __init__(self, stable_id: str) -> None:
        try:
            _, folder, _ = resolve_artifact_folder(artifacts_sources_for_scan(), stable_id)
        except ArtifactIdentityConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Artifact not found") from exc
        revisions_dir = Path(folder) / ".revisions"
        revisions_dir.mkdir(parents=True, exist_ok=True)
        self.path = revisions_dir / "comments.json"
        self.folder = Path(folder)
        self.artifact_id = f"artifact/{stable_id}"

    def _read(self) -> list[dict]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="Comment journal is unreadable") from exc
        return raw if isinstance(raw, list) else []

    def _write(self, threads: list[dict]) -> None:
        fd, name = tempfile.mkstemp(prefix="comments-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(threads, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.path)
        finally:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass

    def _mutate(self, mutation):
        with artifact_lock(self.folder):
            threads = self._read()
            result = mutation(threads)
            self._write(threads)
            return result

    @staticmethod
    def _find(threads: list[dict], thread_id: str) -> dict:
        for thread in threads:
            if thread.get("id") == thread_id:
                return thread
        raise HTTPException(status_code=404, detail="Comment thread not found")

    @staticmethod
    def _touch(thread: dict) -> dict:
        thread["version"] = int(thread.get("version") or 0) + 1
        thread["updated_at"] = _now()
        return thread

    def list_payload(self, status_filter: str = "open") -> dict:
        if status_filter not in {"open", "resolved", "dismissed", "all"}:
            raise HTTPException(status_code=422, detail="Invalid status filter")
        threads = self._read()
        if status_filter == "all":
            threads = [item for item in threads if item.get("status") != "dismissed"]
        else:
            threads = [item for item in threads if item.get("status") == status_filter]
        return {
            "threads": threads,
            "viewer": _VIEWER,
            "capabilities": _CAPABILITIES,
            "unreadCount": 0,
        }

    def create(self, body: dict) -> dict:
        text = str(body.get("text") or "").strip()
        if not text or len(text) > 10_000:
            raise HTTPException(status_code=422, detail="Comment text is required")
        selector = body.get("selector")
        if selector is not None and (not isinstance(selector, str) or len(selector) > 2_000):
            raise HTTPException(status_code=422, detail="Invalid comment selector")
        now = _now()
        thread = {
            "id": str(uuid4()),
            "artifact_id": self.artifact_id,
            "selector": selector,
            "status": "open",
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "payload": {
                "author": {"user_id": _VIEWER["user_id"], "email": _VIEWER["email"]},
                "text": text,
                "revision_id": body.get("revisionId"),
                "kind": body.get("kind") if body.get("kind") in {"review", "issue"} else "review",
                "replies": [],
            },
        }
        return self._mutate(lambda threads: threads.append(thread) or thread)

    def reply(self, thread_id: str, body: dict) -> dict:
        text = str(body.get("text") or "").strip()
        if not text or len(text) > 10_000:
            raise HTTPException(status_code=422, detail="Reply text is required")

        def mutation(threads):
            thread = self._find(threads, thread_id)
            replies = thread.setdefault("payload", {}).setdefault("replies", [])
            if len(replies) >= 500:
                raise HTTPException(status_code=409, detail="Reply limit reached")
            replies.append({
                "id": str(uuid4()),
                "author": {"user_id": _VIEWER["user_id"], "email": _VIEWER["email"]},
                "text": text,
                "created_at": _now(),
            })
            return self._touch(thread)

        return self._mutate(mutation)

    def status(self, thread_id: str, body: dict) -> dict:
        next_status = body.get("status")
        if next_status not in {"open", "resolved", "dismissed"}:
            raise HTTPException(status_code=422, detail="Invalid comment status")

        def mutation(threads):
            thread = self._find(threads, thread_id)
            thread["status"] = next_status
            return self._touch(thread)

        return self._mutate(mutation)

    def edit_thread(self, thread_id: str, body: dict) -> dict:
        text = str(body.get("text") or "").strip()
        if not text or len(text) > 10_000:
            raise HTTPException(status_code=422, detail="Comment text is required")

        def mutation(threads):
            thread = self._find(threads, thread_id)
            thread["payload"]["text"] = text
            thread["payload"]["edited_at"] = _now()
            return self._touch(thread)

        return self._mutate(mutation)

    def delete_thread(self, thread_id: str) -> None:
        def mutation(threads):
            thread = self._find(threads, thread_id)
            threads.remove(thread)

        self._mutate(mutation)

    def edit_reply(self, thread_id: str, reply_id: str, body: dict) -> dict:
        text = str(body.get("text") or "").strip()
        if not text or len(text) > 10_000:
            raise HTTPException(status_code=422, detail="Reply text is required")

        def mutation(threads):
            thread = self._find(threads, thread_id)
            reply = next(
                (
                    item
                    for item in thread["payload"].get("replies", [])
                    if item.get("id") == reply_id
                ),
                None,
            )
            if reply is None:
                raise HTTPException(status_code=404, detail="Reply not found")
            reply.update({"text": text, "edited_at": _now()})
            return self._touch(thread)

        return self._mutate(mutation)

    def delete_reply(self, thread_id: str, reply_id: str) -> dict:
        def mutation(threads):
            thread = self._find(threads, thread_id)
            replies = thread["payload"].get("replies", [])
            reply = next((item for item in replies if item.get("id") == reply_id), None)
            if reply is None:
                raise HTTPException(status_code=404, detail="Reply not found")
            replies.remove(reply)
            return self._touch(thread)

        return self._mutate(mutation)


async def handle_local_comments(request: Request, stable_id: str, subpath: str):
    service = LocalArtifactComments(stable_id)
    parts = [part for part in subpath.split("/") if part]
    method = request.method.upper()
    try:
        body = await request.json() if method in {"POST", "PATCH"} else {}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Request body must be an object")

    if parts == ["threads"] and method == "GET":
        return JSONResponse(service.list_payload(request.query_params.get("status", "open")))
    if parts == ["threads"] and method == "POST":
        return JSONResponse(service.create(body))
    if parts == ["read"] and method == "POST":
        return JSONResponse({"ok": True, "unreadCount": 0})
    if len(parts) == 2 and parts[0] == "threads" and method == "PATCH":
        return JSONResponse(service.edit_thread(parts[1], body))
    if len(parts) == 2 and parts[0] == "threads" and method == "DELETE":
        service.delete_thread(parts[1])
        return JSONResponse({"ok": True})
    if len(parts) == 3 and parts[0] == "threads" and parts[2] == "replies" and method == "POST":
        return JSONResponse(service.reply(parts[1], body))
    if len(parts) == 3 and parts[0] == "threads" and parts[2] == "status" and method == "POST":
        return JSONResponse(service.status(parts[1], body))
    if len(parts) == 4 and parts[0] == "threads" and parts[2] == "replies" and method == "PATCH":
        return JSONResponse(service.edit_reply(parts[1], parts[3], body))
    if len(parts) == 4 and parts[0] == "threads" and parts[2] == "replies" and method == "DELETE":
        return JSONResponse(service.delete_reply(parts[1], parts[3]))
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown comment operation")


def local_comments_stream(stable_id: str) -> StreamingResponse:
    service = LocalArtifactComments(stable_id)

    async def events():
        previous = {
            thread["id"]: thread
            for thread in service.list_payload("all")["threads"]
        }
        try:
            previous_mtime = service.path.stat().st_mtime_ns
        except OSError:
            previous_mtime = 0
        while True:
            await asyncio.sleep(1)
            try:
                current_mtime = service.path.stat().st_mtime_ns
            except OSError:
                current_mtime = 0
            if current_mtime == previous_mtime:
                yield ": keepalive\n\n"
                continue
            previous_mtime = current_mtime
            current = {
                thread["id"]: thread
                for thread in service.list_payload("all")["threads"]
            }
            for deleted_id in previous.keys() - current.keys():
                yield f"event: thread.deleted\ndata: {json.dumps({'id': deleted_id})}\n\n"
            for thread_id, thread in current.items():
                if previous.get(thread_id) == thread:
                    continue
                event = "thread.created" if thread_id not in previous else "thread.updated"
                yield f"event: {event}\ndata: {json.dumps(thread)}\n\n"
            previous = current

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})
