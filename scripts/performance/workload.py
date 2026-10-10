"""Bounded answer workload over the same HTTP API the desktop uses."""
from __future__ import annotations

import asyncio
from collections import Counter
import hashlib
import json
import math
import time
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

STAGING_URL = "https://cowork.staging.mindshub.ai"
STAGING_AUTH_URL = "https://auth.staging.mindshub.ai/v1/authenticate/"
BROWSER_UA = "Mozilla/5.0 CoworkPerformance/1.0"
PROMPTS = {
    "text": "Write exactly 80 numbered lines about the integers 1 through 80. Each line should contain the integer and its square. Do not use tools or create files.",
    "scratchpad": "Use the Python scratchpad to calculate sum(i*i for i in range(100)). Print the result from the scratchpad, then explain it in one short sentence. Do not create files or artifacts.",
}


class WorkloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str = STAGING_URL
    model: str = Field(min_length=1, max_length=100)
    scenario: Literal["text", "scratchpad"] = "text"
    concurrency: int = Field(default=1, ge=1, le=8)
    history_turns: int = Field(default=0, ge=0, le=500)
    answers_per_conversation: int = Field(default=5, ge=1, le=50)
    turn_timeout_seconds: int = Field(default=180, ge=1, le=300)
    total_timeout_seconds: int = Field(default=1800, ge=1, le=7200)
    max_stream_bytes: int = Field(default=2_000_000, ge=1024, le=10_000_000)
    max_history_bytes: int = Field(default=20_000_000, ge=1024, le=100_000_000)

    @model_validator(mode="after")
    def bounded_target(self):
        parsed = urlsplit(self.base_url)
        local = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if self.base_url != STAGING_URL and not local:
            raise ValueError("target must be the exact staging origin or a loopback HTTP server")
        if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("target must be an origin without credentials, path, query or fragment")
        if self.concurrency * (self.history_turns + self.answers_per_conversation) > 1200:
            raise ValueError("one run may generate at most 1200 answers, including history setup")
        return self


class AnswerSample(BaseModel):
    index: int
    conversation: str
    elapsed_seconds: float
    first_text_seconds: float | None = None
    completed: bool = False
    error: str | None = None
    delta_count: int = 0
    text_bytes: int = 0
    scratchpad_results: int = 0


class HistorySample(BaseModel):
    conversation: str
    visible_messages: int
    event_rows: int
    text_delta_rows: int
    scratchpad_results: int


class MeasurementFailure(RuntimeError):
    """A sanitized reason that is safe to retain in CI artifacts."""


class SSEParser:
    """Incremental SSE parser with bounded frame and total input sizes."""

    def __init__(self, limit: int):
        self.limit = limit
        self.received = 0
        self.pending = b""
        self.event_name = ""
        self.data: list[str] = []

    def feed(self, chunk: bytes) -> list[dict]:
        self.received += len(chunk)
        if self.received > self.limit:
            raise MeasurementFailure("stream_byte_limit")
        self.pending += chunk
        result = []
        while b"\n" in self.pending:
            line, self.pending = self.pending.split(b"\n", 1)
            line = line.rstrip(b"\r").decode("utf-8", errors="strict")
            if not line:
                if self.data:
                    raw = "\n".join(self.data)
                    if raw != "[DONE]":
                        try:
                            payload = json.loads(raw)
                        except ValueError as exc:
                            raise MeasurementFailure("invalid_sse_json") from exc
                        if not isinstance(payload, dict):
                            raise MeasurementFailure("invalid_sse_payload")
                        if self.event_name and payload.get("type", self.event_name) != self.event_name:
                            raise MeasurementFailure("sse_event_type_mismatch")
                        payload.setdefault("type", self.event_name)
                        result.append(payload)
                self.data = []
                self.event_name = ""
            elif line.startswith("event:"):
                self.event_name = line[6:].lstrip(" ")
            elif line.startswith("data:"):
                self.data.append(line[5:].lstrip(" "))
        return result


async def verify_staging_identity(client: httpx.AsyncClient, *, expected_org: str, expected_user: str) -> None:
    """Check the dedicated standing identity without calling the suite provisioner."""
    UUID(expected_org)
    UUID(expected_user)
    response = await client.get(STAGING_AUTH_URL)
    if response.status_code != 200:
        raise MeasurementFailure(f"identity_http_{response.status_code}")
    data = response.json()
    if not isinstance(data, dict) or data.get("valid") is not True or data.get("auth_method") != "api_key":
        raise MeasurementFailure("invalid_performance_identity")
    if str(data.get("organization_id")) != expected_org or str(data.get("user_id")) != expected_user:
        raise MeasurementFailure("performance_identity_mismatch")
    if str(data.get("email", "")).lower().endswith("@emailsink.dev"):
        raise MeasurementFailure("nightly_test_identity_is_not_a_performance_identity")


def successful_scratchpad_execution(event: dict) -> bool:
    # The formatter uses this result role for other tools and dump/reset too.
    return (event.get("thought_role") == "thought.scratchpad.result"
            and event.get("tool_name") == "scratchpad"
            and event.get("tool_action") == "exec"
            and event.get("cell_status") == "ok")


class Workload:
    def __init__(self, config: WorkloadConfig, client: httpx.AsyncClient):
        self.config = config
        self.client = client
        self.run_id = str(uuid4())
        self.conversations: list[str] = []
        self.cleanup_errors: list[str] = []
        self.history: list[HistorySample] = []

    async def prepare(self) -> None:
        # Create first, then remember only server-confirmed IDs. A cleanup never
        # lists or deletes conversations that preceded this invocation.
        for _ in range(self.config.concurrency):
            response = await self.client.post("/api/v1/conversations/", json={
                "title": f"Performance measurement {self.run_id}", "model": self.config.model,
            })
            if response.status_code != 201:
                raise MeasurementFailure(f"create_http_{response.status_code}")
            conversation = str(UUID(response.json()["id"]))
            self.conversations.append(conversation)

        async def seed(conversation: str) -> None:
            for index in range(self.config.history_turns):
                sample = await self.answer(conversation, index)
                if not sample.completed:
                    raise MeasurementFailure(f"history_setup_{sample.error}")
            if self.config.history_turns:
                self.history.append(await self.inspect_history(conversation))

        await self._gather(seed(conversation) for conversation in self.conversations)

    async def _gather(self, coroutines):
        tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def answer(self, conversation: str, index: int) -> AnswerSample:
        started = time.perf_counter()
        sample = AnswerSample(index=index, conversation=conversation, elapsed_seconds=0)
        parser = SSEParser(self.config.max_stream_bytes)
        try:
            async with asyncio.timeout(self.config.turn_timeout_seconds):
                async with self.client.stream("POST", "/api/v1/responses/", json={
                    "input": PROMPTS[self.config.scenario], "model": self.config.model,
                    "conversation": conversation, "stream": True,
                    "trace_tags": ["cowork-performance", self.run_id],
                }) as response:
                    if response.status_code != 200:
                        raise MeasurementFailure(f"turn_http_{response.status_code}")
                    async for chunk in response.aiter_bytes():
                        for event in parser.feed(chunk):
                            kind = event.get("type")
                            if kind == "response.output_text.delta":
                                delta = event.get("delta")
                                if not isinstance(delta, str):
                                    raise MeasurementFailure("invalid_text_delta")
                                sample.delta_count += 1
                                sample.text_bytes += len(delta.encode("utf-8"))
                                if delta and sample.first_text_seconds is None:
                                    sample.first_text_seconds = time.perf_counter() - started
                            if successful_scratchpad_execution(event):
                                sample.scratchpad_results += 1
                            if kind in {"response.failed", "response.cancelled", "error"}:
                                raise MeasurementFailure("turn_failed" if kind != "response.cancelled" else "turn_cancelled")
                            if kind == "response.completed":
                                sample.completed = True
                                break
                        if sample.completed:
                            break
                if not sample.completed:
                    raise MeasurementFailure("missing_terminal_event")
                if self.config.scenario == "scratchpad" and sample.scratchpad_results == 0:
                    sample.completed = False
                    raise MeasurementFailure("scratchpad_scenario_not_exercised")
        except (MeasurementFailure, httpx.HTTPError, TimeoutError, UnicodeError, ValueError) as exc:
            sample.completed = False
            sample.error = str(exc) if isinstance(exc, MeasurementFailure) else type(exc).__name__
            await self.cancel(conversation)
        sample.elapsed_seconds = time.perf_counter() - started
        return sample

    async def cancel(self, conversation: str) -> None:
        try:
            response = await self.client.post("/api/v1/responses/cancel", json={"conversation_id": conversation})
            if response.status_code not in {200, 404}:
                self.cleanup_errors.append(f"cancel_http_{response.status_code}:{conversation}")
        except httpx.HTTPError:
            self.cleanup_errors.append(f"cancel_transport_error:{conversation}")

    async def measure(self) -> tuple[list[AnswerSample], float]:
        started = time.perf_counter()

        async def lane(conversation: str):
            samples = []
            for index in range(self.config.answers_per_conversation):
                sample = await self.answer(conversation, index)
                samples.append(sample)
                if not sample.completed:
                    break
            return samples

        lanes = await self._gather(lane(conversation) for conversation in self.conversations)
        return [answer for lane in lanes for answer in lane], time.perf_counter() - started

    async def inspect_history(self, conversation: str) -> HistorySample:
        sample = HistorySample(conversation=conversation, visible_messages=0, event_rows=0,
                               text_delta_rows=0, scratchpad_results=0)
        cursor = None
        seen = set()
        consumed = 0
        for _ in range(128):
            params = {"limit": "20"}
            if cursor:
                params["before"] = cursor
            async with self.client.stream("GET", f"/api/v1/conversations/{conversation}/items", params=params) as response:
                if response.status_code != 200:
                    raise MeasurementFailure(f"history_http_{response.status_code}")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    consumed += len(chunk)
                    if consumed > self.config.max_history_bytes:
                        raise MeasurementFailure("history_byte_limit")
                    body.extend(chunk)
            page = json.loads(body)
            if not isinstance(page, dict) or not isinstance(page.get("items"), list) or not isinstance(page.get("hasMore"), bool):
                raise MeasurementFailure("invalid_history_response")
            for item in page["items"]:
                if not isinstance(item, dict) or not isinstance(item.get("events", []), list):
                    raise MeasurementFailure("invalid_history_item")
                sample.visible_messages += 1
                for event in item.get("events", []):
                    if not isinstance(event, dict):
                        continue
                    sample.event_rows += 1
                    sample.text_delta_rows += event.get("type") == "response.output_text.delta"
                    sample.scratchpad_results += successful_scratchpad_execution(event)
            if not page["hasMore"]:
                break
            cursor = page.get("nextBefore")
            if not isinstance(cursor, str) or not cursor or cursor in seen:
                raise MeasurementFailure("invalid_history_cursor")
            seen.add(cursor)
        else:
            raise MeasurementFailure("history_page_limit")
        if sample.event_rows == 0 or (self.config.scenario == "scratchpad" and sample.scratchpad_results == 0):
            raise MeasurementFailure("representative_history_not_persisted")
        return sample

    async def cleanup(self) -> None:
        for conversation in self.conversations:
            await self.cancel(conversation)
            try:
                response = await self.client.delete(f"/api/v1/conversations/{conversation}")
                if response.status_code not in {200, 404}:
                    self.cleanup_errors.append(f"delete_http_{response.status_code}:{conversation}")
            except httpx.HTTPError:
                self.cleanup_errors.append(f"delete_transport_error:{conversation}")


def summarize(samples: list[AnswerSample], elapsed: float) -> dict:
    completed = [sample for sample in samples if sample.completed]
    latencies = sorted(sample.elapsed_seconds for sample in completed)

    def percentile(fraction: float) -> float | None:
        return latencies[max(0, math.ceil(len(latencies) * fraction) - 1)] if latencies else None

    return {
        "attempted": len(samples), "completed": len(completed), "failed": len(samples) - len(completed),
        "errors": dict(Counter(sample.error for sample in samples if sample.error)),
        "elapsed_seconds": elapsed,
        "completed_answers_per_second": len(completed) / elapsed if elapsed > 0 else None,
        "latency_seconds": {"p50": percentile(.5), "p95": percentile(.95), "max": max(latencies, default=None)},
    }


def workload_identity(config: WorkloadConfig) -> str:
    return hashlib.sha256(json.dumps({"config": config.model_dump(), "prompts": PROMPTS}, sort_keys=True).encode()).hexdigest()
