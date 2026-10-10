"""Safety and result contracts of the opt-in answer measurement driver."""
import argparse
import json
import subprocess
import sys
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from scripts.performance.profile_answers import (
    PROCESS_PROBE, compare_reports, cpu_delta, run, select_pods,
)
from scripts.performance.workload import (
    MeasurementFailure, SSEParser, STAGING_URL, Workload, WorkloadConfig,
    summarize, verify_staging_identity,
)


@pytest.mark.parametrize("changes", [
    {"base_url": "https://cowork.mindshub.ai"},
    {"base_url": "https://cowork.staging.mindshub.ai.attacker.test"},
    {"base_url": "http://key@localhost:9010"},
    {"base_url": "http://localhost:9010/redirect"},
    {"concurrency": 9}, {"answers_per_conversation": 51},
    {"history_turns": 500, "concurrency": 3}, {"turn_timeout_seconds": 301},
])
def test_invalid_target_and_unbounded_load_are_rejected_before_io(changes):
    with pytest.raises(ValidationError):
        WorkloadConfig(model="test-model", **changes)


def test_sse_frames_survive_every_byte_boundary_and_crlf():
    parser = SSEParser(1024)
    body = ('event: response.output_text.delta\r\n'
            'data: {"delta": "café"}\r\n\r\n'
            'event: response.completed\n'
            'data: {"type":\n'
            'data: "response.completed"}\n\n').encode()
    events = [event for byte in body for event in parser.feed(bytes([byte]))]
    assert [event["type"] for event in events] == ["response.output_text.delta", "response.completed"]
    assert events[0]["delta"] == "café"


@pytest.mark.parametrize("body,reason", [
    (b"x" * 1025, "stream_byte_limit"),
    (b"data: []\n\n", "invalid_sse_payload"),
    (b'event: response.completed\ndata: {"type":"response.failed"}\n\n', "sse_event_type_mismatch"),
])
def test_bad_or_oversized_sse_is_not_a_completed_answer(body, reason):
    with pytest.raises(MeasurementFailure, match=reason):
        SSEParser(1024).feed(body)


@pytest.mark.parametrize("changes,reason", [
    ({"organization_id": str(uuid4())}, "performance_identity_mismatch"),
    ({"email": "cowork@emailsink.dev"}, "nightly_test_identity"),
    ({"valid": False}, "invalid_performance_identity"),
])
async def test_identity_guard_prevents_using_wrong_or_nightly_tenant(changes, reason):
    org, user = str(uuid4()), str(uuid4())
    identity = {"valid": True, "auth_method": "api_key", "organization_id": org,
                "user_id": user, "email": "cowork-perf@mindshub.ai", **changes}
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=identity)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(MeasurementFailure, match=reason):
            await verify_staging_identity(client, expected_org=org, expected_user=user)
    assert len(requests) == 1 and requests[0].method == "GET"
    assert str(requests[0].url) == "https://auth.staging.mindshub.ai/v1/authenticate/"


def sse(*events):
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()


async def test_real_api_sequence_seeds_measures_pages_and_only_deletes_its_own_conversations():
    ids = [str(uuid4()), str(uuid4())]
    created, deleted, turns, pages = [], [], [], []

    def respond(request):
        path = request.url.path
        if request.method == "POST" and path == "/api/v1/conversations/":
            cid = ids[len(created)]
            created.append(cid)
            return httpx.Response(201, json={"id": cid})
        if path == "/api/v1/responses/":
            body = json.loads(request.content)
            turns.append(body)
            return httpx.Response(200, content=sse(
                {"type": "response.output_text.delta", "delta": "secret answer not in artifacts"},
                {"type": "response.completed"},
            ))
        if path.endswith("/items"):
            pages.append(dict(request.url.params))
            return httpx.Response(200, json={"items": [{"events": [{"type": "response.output_text.delta"}]}],
                                           "hasMore": False, "nextBefore": None})
        if path == "/api/v1/responses/cancel":
            return httpx.Response(404)
        if request.method == "DELETE":
            deleted.append(path.rsplit("/", 1)[1])
            return httpx.Response(200)
        pytest.fail(f"unexpected route: {request.method} {path}")

    config = WorkloadConfig(model="same-model", concurrency=2, history_turns=1, answers_per_conversation=2)
    async with httpx.AsyncClient(base_url=STAGING_URL, transport=httpx.MockTransport(respond)) as client:
        workload = Workload(config, client)
        try:
            await workload.prepare()
            samples, elapsed = await workload.measure()
        finally:
            await workload.cleanup()
    assert created == deleted == ids
    assert len(turns) == 6 and all(turn["model"] == "same-model" for turn in turns)
    assert len(samples) == 4 and summarize(samples, elapsed)["completed"] == 4
    assert all(page["limit"] == "20" for page in pages)
    assert "secret answer" not in json.dumps([sample.model_dump() for sample in samples])
    assert workload.cleanup_errors == []


async def test_failed_turn_stops_its_lane_and_cancels_without_retaining_server_error():
    posted = []
    cid = str(uuid4())

    def respond(request):
        posted.append(request.url.path)
        if request.url.path.endswith("/cancel"):
            return httpx.Response(200)
        return httpx.Response(200, content=sse({"type": "response.failed", "error": "credential should not appear"}))

    async with httpx.AsyncClient(base_url=STAGING_URL, transport=httpx.MockTransport(respond)) as client:
        workload = Workload(WorkloadConfig(model="m", answers_per_conversation=5), client)
        workload.conversations = [cid]
        samples, _ = await workload.measure()
    assert posted == ["/api/v1/responses/", "/api/v1/responses/cancel"]
    assert len(samples) == 1 and samples[0].completed is False
    assert samples[0].error == "turn_failed"


async def test_scratchpad_scenario_rejects_success_without_a_real_tool_result():
    def respond(request):
        return httpx.Response(200, content=sse({"type": "response.completed"}))

    async with httpx.AsyncClient(base_url=STAGING_URL, transport=httpx.MockTransport(respond)) as client:
        workload = Workload(WorkloadConfig(model="m", scenario="scratchpad"), client)
        sample = await workload.answer(str(uuid4()), 0)
    assert not sample.completed and sample.error == "scratchpad_scenario_not_exercised"


@pytest.mark.parametrize("name,action,error,successful", [
    ("other_tool", "exec", "", False),
    ("scratchpad", "dump", "", False),
    ("scratchpad", "reset", "", False),
    ("scratchpad", "exec", "execution failed", False),
    ("scratchpad", "exec", "Cell timed out", False),
    ("scratchpad", "exec", "", True),
])
async def test_scratchpad_measurement_requires_successful_exec_in_live_and_stored_events(name, action, error, successful):
    from anton.core.llm.provider import StreamToolResult
    from cowork.harnesses.anton_harness.stream_formatter import format_responses_stream
    from cowork.schemas.conversations import ConversationItemsPage

    async def provider_events():
        yield StreamToolResult(name=name, action=action,
                               content=json.dumps({"error": error, "stdout": "328350"}), id="tool-1")

    # Use the real formatter and GET /items envelope: nested event dictionaries
    # retain snake_case fields while only the page envelope is camelCase.
    body = "".join([chunk async for chunk in format_responses_stream(provider_events(), model="m")]).encode()
    events = SSEParser(100_000).feed(body)
    page = ConversationItemsPage(items=[{"role": "assistant", "events": events}], has_more=False)

    def respond(request):
        if request.url.path.endswith("/items"):
            return httpx.Response(200, json=page.model_dump(by_alias=True))
        if request.url.path.endswith("/cancel"):
            return httpx.Response(200)
        return httpx.Response(200, content=body)

    async with httpx.AsyncClient(base_url=STAGING_URL, transport=httpx.MockTransport(respond)) as client:
        workload = Workload(WorkloadConfig(model="m", scenario="scratchpad"), client)
        conversation = str(uuid4())
        sample = await workload.answer(conversation, 0)
        assert sample.completed is successful
        assert sample.scratchpad_results == int(successful)
        if successful:
            assert (await workload.inspect_history(conversation)).scratchpad_results == 1
        else:
            assert sample.error == "scratchpad_scenario_not_exercised"
            with pytest.raises(MeasurementFailure, match="representative_history_not_persisted"):
                await workload.inspect_history(conversation)


@pytest.mark.parametrize("oversized", [False, True])
async def test_history_walk_refuses_repeating_cursor_and_limits_total_bytes(oversized):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"items": [], "hasMore": True, "nextBefore": "same",
                                       "padding": "x" * (1100 if oversized else 0)})

    async with httpx.AsyncClient(base_url=STAGING_URL, transport=httpx.MockTransport(respond)) as client:
        workload = Workload(WorkloadConfig(model="m", max_history_bytes=1024), client)
        with pytest.raises(MeasurementFailure, match="history_byte_limit" if oversized else "invalid_history_cursor"):
            await workload.inspect_history(str(uuid4()))
    assert len(calls) == (1 if oversized else 2)


def pod(name, sha="a" * 40):
    return {"metadata": {"name": name, "uid": name + "-uid"},
            "spec": {"containers": [{"name": "cowork-server", "image": f"repo:staging-{sha}",
                                      "resources": {"limits": {"cpu": "1"}}}]},
            "status": {"containerStatuses": [{"name": "cowork-server", "containerID": name + "-container", "ready": True}]}}


def test_pod_selection_requires_every_serving_pod_on_exact_candidate():
    first, second = pod("cowork-server-a"), pod("cowork-server-b")
    assert len(select_pods({"items": [first, second]}, "a" * 40)) == 2
    second["spec"]["containers"][0]["image"] = "repo:staging-" + "b" * 40
    with pytest.raises(MeasurementFailure, match="image_does_not_match"):
        select_pods({"items": [first, second]}, "a" * 40)


def cpu(name, count=1.0):
    return {"name": name, "pod_uid": name, "container_id": name, "pid": 5, "start_ticks": 10,
            "boot_id": "boot", "cpu_seconds": count, "resources": {"limits": {"cpu": "1"}}}


@pytest.mark.parametrize("changed", ["pid", "pod_uid", "container_id", "start_ticks", "boot_id"])
def test_cpu_rejects_restart_instead_of_reporting_a_false_improvement(changed):
    before = cpu("pod")
    after = {**before, "cpu_seconds": 2.0, changed: "changed"}
    with pytest.raises(MeasurementFailure, match="lifetime_changed"):
        cpu_delta([before], [after])


def test_cpu_sums_both_processes_and_refuses_counter_reset():
    assert cpu_delta([cpu("a"), cpu("b")], [cpu("a", 2.5), cpu("b", 3.5)]) == 4.0
    with pytest.raises(MeasurementFailure, match="went_backwards"):
        cpu_delta([cpu("a")], [cpu("a", .5)])


def test_process_probe_rejects_init_wrapper_and_selects_actual_python_process(tmp_path):
    root = tmp_path / "proc"
    (root / "sys/kernel/random").mkdir(parents=True)
    (root / "sys/kernel/random/boot_id").write_text("boot")
    for pid, argv in [(1, ["docker-init", "--", "python", "-m", "uvicorn", "cowork.server:app"]),
                      (20, ["/app/.venv/bin/python", "-m", "uvicorn", "spa_wrapper:app"])]:
        folder = root / str(pid)
        folder.mkdir()
        (folder / "cmdline").write_bytes(b"\0".join(argument.encode() for argument in argv))
        fields = ["S"] + ["0"] * 21
        fields[11], fields[12], fields[19] = "100", "50", "999"
        (folder / "stat").write_text(f"{pid} (python worker) " + " ".join(fields))
    result = subprocess.run([sys.executable, "-c", PROCESS_PROBE, "0", str(root)], capture_output=True, check=True)
    assert json.loads(result.stdout)["pid"] == 20
    rejected = subprocess.run([sys.executable, "-c", PROCESS_PROBE, "1", str(root)], capture_output=True)
    assert rejected.returncode != 0


def report():
    return {"status": "complete", "cleanup_errors": [], "run_id": "run", "workload_id": "same",
            "summary": {"failed": 0, "completed": 10}, "cpu_before": [cpu("a"), cpu("b")],
            "cpu_seconds_per_completed_answer": 1.0, "remote_worker_provenance": "worker@sha256:123 anton:abc",
            "profiler_versions": ["py-spy 0.4.1", "py-spy 0.4.1"]}


@pytest.mark.parametrize("field,value,reason", [
    ("workload_id", "different", "workloads_differ"),
    ("remote_worker_provenance", "unverified", "provenance_is_unverified"),
    ("remote_worker_provenance", "other worker revision", "remote_workers_differ"),
    ("profiler_versions", ["py-spy 0.5.0"], "profiler_versions_differ"),
    ("cleanup_errors", ["delete failed"], "incomplete_run"),
    ("summary", {"failed": 1, "completed": 9}, "failed_answers"),
])
def test_comparison_refuses_uncontrolled_or_failed_measurements(field, value, reason):
    candidate = {**report(), field: value}
    with pytest.raises(MeasurementFailure, match=reason):
        compare_reports(report(), candidate)


def test_full_stack_comparison_is_explicit_and_never_claims_server_isolation():
    candidate = {**report(), "remote_worker_provenance": "other known revision", "cpu_seconds_per_completed_answer": .5}
    compared = compare_reports(report(), candidate, scope="full-stack")
    assert compared["scope"] == "full-stack" and compared["cpu_reduction_fraction"] == .5


async def test_local_staging_invocation_fails_before_kubectl_auth_or_load(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("no process or network access is allowed before the CI guard")

    monkeypatch.setattr(subprocess, "run", forbidden)
    args = argparse.Namespace(output=str(tmp_path / "results"), server_sha="a" * 40, pid=0,
                              remote_worker_provenance="known", debugger_image="")
    result = await run(WorkloadConfig(model="m"), args)
    assert result["status"] == "failed"
    assert result["error"] == "staging_profiling_runs_only_in_the_reviewed_CI_workflow"
    assert (tmp_path / "results/report.json").is_file()


@pytest.mark.parametrize("fail_start", [False, True])
async def test_staging_adapter_cleans_every_owned_debugger_even_when_start_is_ambiguous(tmp_path, monkeypatch, fail_start):
    import types
    from scripts.performance import profile_answers as profiler
    from scripts.performance.workload import AnswerSample, HistorySample

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    for name in ("COWORK_PERF_API_KEY", "COWORK_PERF_ORG_ID", "COWORK_PERF_USER_ID"):
        monkeypatch.setenv(name, str(uuid4()))
    pods = [{"name": "cowork-server-a", "pod_uid": "a", "container_id": "a"},
            {"name": "cowork-server-b", "pod_uid": "b", "container_id": "b"}]
    monkeypatch.setattr(profiler, "staging_pods", lambda _: pods)
    counters = {}
    calls = []

    def sample(target):
        target.pid = 20
        counters[target.name] = counters.get(target.name, 0) + 1
        return {**cpu(target.name, counters[target.name]), "pid": 20}

    def command(target, *args, **kwargs):
        assert kwargs.get("profiler") is True
        assert target.profiler_prefix[-1] == "debugger"
        return b"py-spy 0.4.1" if "--version" in args else b"{}"

    class Debugger:
        def __init__(self, pod, **kwargs):
            self.name = pod["name"]
            self.stopped = False
            self.prefix = ["never-executed", "debugger"]
            calls.append((self.name, "constructed"))

        def start(self, process):
            assert process["pid"] == 20
            calls.append((self.name, "start"))
            if fail_start and self.name.endswith("b"):
                raise MeasurementFailure("ambiguous_patch_timeout")

        def stop(self):
            self.stopped = True
            calls.append((self.name, "stop"))

        def provenance(self):
            return {"name": self.name, "stopped": self.stopped}

    async def verified(*args, **kwargs):
        pass

    async def prepared(workload):
        calls.append(("workload", "prepare"))
        workload.conversations = [str(uuid4())]

    async def measured(workload):
        return [AnswerSample(index=0, conversation=workload.conversations[0], elapsed_seconds=.1, completed=True)], .1

    async def inspected(workload, conversation):
        return HistorySample(conversation=conversation, visible_messages=2, event_rows=2, text_delta_rows=1, scratchpad_results=0)

    async def cleaned(workload):
        calls.append(("workload", "cleanup"))

    monkeypatch.setitem(sys.modules, "scripts.performance.debugger", types.SimpleNamespace(Debugger=Debugger))
    monkeypatch.setattr(profiler.Target, "sample", sample)
    monkeypatch.setattr(profiler.Target, "command", command)
    monkeypatch.setattr(profiler, "verify_staging_identity", verified)
    monkeypatch.setattr(Workload, "prepare", prepared)
    monkeypatch.setattr(Workload, "measure", measured)
    monkeypatch.setattr(Workload, "inspect_history", inspected)
    monkeypatch.setattr(Workload, "cleanup", cleaned)
    monkeypatch.setattr(profiler, "start_profile", lambda *args: types.SimpleNamespace(process=types.SimpleNamespace(poll=lambda: None)))
    monkeypatch.setattr(profiler, "finish_profile", lambda *args: None)
    args = argparse.Namespace(output=str(tmp_path / "results"), server_sha="a" * 40, pid=0,
                              remote_worker_provenance="worker@sha256:123 anton:abc", debugger_image="repo@sha256:123")
    result = await run(WorkloadConfig(model="m"), args)
    assert [call for call in calls if call[1] == "stop"] == [(pod["name"], "stop") for pod in pods]
    assert all(item["stopped"] for item in result["debuggers"])
    if fail_start:
        assert ("workload", "prepare") not in calls
        assert result["error"] == "ambiguous_patch_timeout" and result["status"] == "failed"
    else:
        assert result["status"] == "complete"
        assert result["cpu_seconds"] == result["cpu_seconds_per_completed_answer"] == 2.0
