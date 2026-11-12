"""An export converts in a process the server kills past its limits.

The hostile shapes come from ``scripts/measure_export_capacity.py``: each
takes 25 s or more to convert, so with a short wall clock injected each one
is killed, answers "This file can't be exported.", writes nothing and leaves
no process behind. Ordinary documents export under the same limits, and the
next export after a kill runs, so a kill frees the one conversion slot.
Fake conversion processes, started from ``python -c``, drive each way a
process can end. Every test here runs on Linux, macOS and Windows unless it
says why not.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

import docx
import httpx
import psutil
import pytest

import cowork
from cowork.common.paths import pinned_dir
from cowork.common.settings.app_settings import get_app_settings
from cowork.schemas.artifact_export import InlineLimits, OmittedReason
from cowork.schemas.export_child import AssetReply, AssetRequest, ChildLimits, ConvertRequest
from cowork.server import create_app
from cowork.services import (
    artifact_export,
    artifact_identity,
    artifact_inline,
    docx_writer,
    export_child,
    export_runner,
)
from cowork.services.artifact_export import ExportRefusal, ExportRefused
from cowork.services.artifact_files import ArtifactFileContent, ArtifactFileRefused
from cowork.services.artifact_inline import TRUNCATION_NOTE, folder_reader
from cowork.services.artifacts import ProjectArtifacts
from scripts import measure_export_capacity as capacity
from test_artifact_export_route import _snapshot
from test_artifact_export_service import local_mode  # noqa: F401  (fixture)
from tests._no_fetch import recording_server

ROOT = Path(__file__).resolve().parents[1]
# The wall clock the kill tests run with. Each hostile shape needs far longer.
WALL_SECONDS = 5.0
# A memory limit an ordinary report fits under and a 10 MiB page of one-word
# paragraphs doesn't.
MEMORY_BYTES = 320 * 1024 * 1024
MIB = 1024 * 1024
RUNNER_LOGGER = "cowork.services.export_runner"
LIVE = "/api/v1/health/live"

HOSTILE_TIME_CASES = ("md-unclosed-fences", "md-reference-brackets")


@pytest.fixture
def short_wall(monkeypatch):
    monkeypatch.setattr(artifact_export, "CONVERSION_WALL_SECONDS", WALL_SECONDS)


def _artifact(project: Path, case: capacity.Case) -> tuple[ProjectArtifacts, Path, dict]:
    """An artifact holding *case*'s files, in the general project at *project*."""
    folder = project / ".anton" / "artifacts" / f"report-{uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    case.build(folder)
    metadata = {"id": uuid4().hex, "slug": folder.name, "type": "document", "primary": case.primary}
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    source = ProjectArtifacts(
        base=project / ".anton" / "artifacts",
        project_id=None,
        project_name="general",
        trusted_anchor=project,
        root_parts=(".anton", "artifacts"),
    )
    return source, folder, metadata


def _report(text: str, primary: str = "report.md") -> capacity.Case:
    return capacity.Case(
        primary=primary, build=lambda folder: capacity._write(folder, primary, text), note=""
    )


ORDINARY = _report("# Quarterly\n\nRevenue grew.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")


def _export(source: ProjectArtifacts, folder: Path, metadata: dict, export_format: str = "docx"):
    return artifact_export.export_primary_file(
        source=source,
        folder_name=folder.name,
        metadata=metadata,
        artifact_id=metadata["id"],
        export_format=export_format,
        can_edit=True,
    )


def _descendants() -> set[int]:
    return {process.pid for process in psutil.Process().children(recursive=True)}


def _causes(caplog) -> list[str]:
    return [
        record.cause
        for record in caplog.records
        if record.name == RUNNER_LOGGER and record.getMessage() == "Export conversion stopped"
    ]


def _assert_refused_cleanly(
    *, export, folder: Path, caplog, causes: set[str], within: float
) -> None:
    before = _snapshot(folder)
    children = _descendants()
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        with pytest.raises(ExportRefused) as refused:
            export()
    elapsed = time.monotonic() - started

    assert refused.value.refusal is ExportRefusal.TOO_LARGE
    assert str(refused.value) == "This file can't be exported."
    assert elapsed < within, f"answered after {elapsed:.1f}s"
    assert set(_causes(caplog)) <= causes and _causes(caplog), _causes(caplog)
    assert _snapshot(folder) == before
    assert _descendants() == children


@pytest.mark.parametrize("case_id", HOSTILE_TIME_CASES)
def test_a_hostile_shape_is_killed_at_the_wall_clock_writes_nothing_and_leaves_no_process(
    local_mode, short_wall, tmp_path, caplog, case_id
):
    source, folder, metadata = _artifact(tmp_path / "general", capacity.CASES[case_id])

    _assert_refused_cleanly(
        export=lambda: _export(source, folder, metadata),
        folder=folder,
        caplog=caplog,
        causes={"wall clock"},
        within=WALL_SECONDS + 1,
    )

    # The kill freed the slot: an ordinary report exports under the same limits.
    source, folder, metadata = _artifact(tmp_path / "general", ORDINARY)
    assert _export(source, folder, metadata).filename == "report.docx"


def test_a_conversion_past_its_memory_limit_is_killed_and_answered_the_same_way(
    local_mode, monkeypatch, tmp_path, caplog
):
    """On Linux the process's own address-space limit stops it ("refused");
    on macOS and Windows the server's memory watch does ("memory")."""
    monkeypatch.setattr(artifact_export, "CONVERSION_MEMORY_BYTES", MEMORY_BYTES)
    source, folder, metadata = _artifact(tmp_path / "general", capacity.CASES["html-node-heavy"])

    _assert_refused_cleanly(
        export=lambda: _export(source, folder, metadata),
        folder=folder,
        caplog=caplog,
        causes={"memory", "refused"},
        within=artifact_export.CONVERSION_WALL_SECONDS,
    )

    source, folder, metadata = _artifact(tmp_path / "general", ORDINARY)
    assert _export(source, folder, metadata).filename == "report.docx"


@contextlib.contextmanager
def _unwatched_child(*, limits: ChildLimits, request: ConvertRequest, source: bytes):
    """A conversion process with nothing watching it, so only its own limits apply."""
    command, environment = export_runner.child_command(limits=limits)
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=environment,
    )
    try:
        export_child.write_frame(stream=process.stdin, message=request, payload=source)
        process.stdin.close()
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def _request(*, suffix: str, export_format: str = "docx") -> ConvertRequest:
    return ConvertRequest(
        export_format=export_format,
        suffix=suffix,
        title="report",
        base_parts=(),
        limits=InlineLimits(max_total_bytes=25 * MIB),
    )


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="only Linux has an address-space limit")
def test_on_linux_the_process_stops_itself_at_its_address_space_limit(tmp_path):
    case_folder = tmp_path / "case"
    case_folder.mkdir()
    capacity.CASES["html-node-heavy"].build(case_folder)
    with _unwatched_child(
        limits=ChildLimits(wall_seconds=600.0, memory_bytes=MEMORY_BYTES),
        request=_request(suffix=".html"),
        source=(case_folder / "report.html").read_bytes(),
    ) as process:
        assert process.wait(timeout=120) == export_child.EXIT_REFUSED


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no CPU-time limit")
def test_the_process_stops_at_its_own_cpu_time_limit():
    """The backstop for a server that stopped watching: SIGXCPU, then SIGKILL."""
    with _unwatched_child(
        limits=ChildLimits(wall_seconds=1.0, memory_bytes=2048 * MIB),
        request=_request(suffix=".md"),
        source=("[a][" * 8_000).encode(),
    ) as process:
        assert process.wait(timeout=60) in {-signal.SIGXCPU, -signal.SIGKILL}


def test_a_second_conversion_answers_busy_at_once_while_one_runs(local_mode, short_wall, tmp_path):
    hostile = _artifact(tmp_path / "general", capacity.CASES["md-reference-brackets"])
    ordinary = _artifact(tmp_path / "general", ORDINARY)
    first: list[BaseException] = []
    started = threading.Event()
    convert = export_runner._run

    def signalling_run(**kwargs):
        started.set()
        return convert(**kwargs)

    def run_first():
        try:
            _export(*hostile)
        except BaseException as exc:
            first.append(exc)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(export_runner, "_run", signalling_run)
        thread = threading.Thread(target=run_first)
        thread.start()
        assert started.wait(timeout=10)
        began = time.monotonic()
        with pytest.raises(ExportRefused) as busy:
            _export(*ordinary)
        answered = time.monotonic() - began
        thread.join(timeout=WALL_SECONDS + 10)

    assert busy.value.refusal is ExportRefusal.BUSY
    assert answered < 1.0
    assert [type(exc) for exc in first] == [ExportRefused]
    assert _export(*ordinary).filename == "report.docx"


def test_the_kill_comes_after_the_inliner_and_the_word_writer_have_both_stopped():
    """Each stage stops at its own deadline and still writes the file with a
    note, so the kill must leave room for both, and for Markdown before them."""
    stages = artifact_inline.DEADLINE_SECONDS + docx_writer.DEADLINE_SECONDS
    assert artifact_export.CONVERSION_WALL_SECONDS >= stages + 5


def test_a_document_cut_short_by_the_inliner_still_exports_with_its_note(local_mode, tmp_path):
    nodes = InlineLimits(max_total_bytes=0).max_nodes + 10_000
    page = "<html><body>" + "<p>x</p>" * nodes + "</body></html>"
    source, folder, metadata = _artifact(tmp_path / "general", _report(page, primary="report.html"))

    result = _export(source, folder, metadata)

    written = docx.Document(io.BytesIO((folder / result.filename).read_bytes()))
    assert written.paragraphs[-1].text == TRUNCATION_NOTE


_PEAK_PROBE = """
import json, sys, tempfile
from pathlib import Path
from scripts import measure_export_capacity as capacity
from cowork.services import artifact_export
from cowork.services.artifacts import ProjectArtifacts

artifact_export._org_mode = lambda: False
size = int(sys.argv[1])
with tempfile.TemporaryDirectory() as tmp:
    project = Path(tmp) / "general"
    folder = project / ".anton" / "artifacts" / "report"
    folder.mkdir(parents=True)
    table = "| a | b | c |\\n|---|---|---|\\n" + "| 1 | 2 | 3 |\\n" * (size // 14)
    (folder / "report.md").write_text(table)
    (folder / "metadata.json").write_text(json.dumps({"id": "a" * 32, "primary": "report.md"}))
    source = ProjectArtifacts(base=project / ".anton" / "artifacts", project_id=None,
        project_name="general", trusted_anchor=project, root_parts=(".anton", "artifacts"))
    before = capacity._peak_rss_mib()
    artifact_export.export_primary_file(source=source, folder_name="report",
        metadata={"id": "a" * 32, "primary": "report.md", "type": "document"},
        artifact_id="a" * 32, export_format="docx", can_edit=True)
    print(json.dumps({"server": capacity._peak_rss_mib() - before,
                      "conversion": capacity._peak_rss_mib(who="children")}))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="Windows reports no peak RSS for a finished process")
def test_the_servers_own_peak_memory_does_not_grow_with_the_input(local_mode):
    """Each size runs in a fresh interpreter, so each peak is that export's own."""
    peaks = {}
    for size in (64 * 1024, 500 * 1024):
        finished = subprocess.run(
            [sys.executable, "-c", _PEAK_PROBE, str(size)],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        peaks[size] = json.loads(finished.stdout.strip().splitlines()[-1])

    small, large = peaks[64 * 1024], peaks[500 * 1024]
    assert large["conversion"] - small["conversion"] > 40, peaks
    assert large["server"] - small["server"] < 15, peaks


async def _heartbeat(counter: list[int]) -> None:
    while True:
        await asyncio.sleep(0.01)
        counter[0] += 1


@pytest.fixture
def hostile_desktop_artifact(local_mode, short_wall):
    """A hostile Markdown report in the desktop's general project, which ``local`` resolves."""
    project = Path(get_app_settings().project.root_dir) / "general"
    _source, folder, metadata = _artifact(project, capacity.CASES["md-unclosed-fences"])
    artifact_identity._clear_identity_indexes()
    yield folder, metadata
    shutil.rmtree(folder, ignore_errors=True)
    artifact_identity._clear_identity_indexes()


async def test_liveness_answers_at_once_while_a_hostile_export_converts(
    hostile_desktop_artifact, granted_product_permissions
):
    folder, metadata = hostile_desktop_artifact
    loop = asyncio.get_running_loop()
    ticks = [0]
    latencies: list[float] = []
    transport = httpx.ASGITransport(app=create_app(), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get(LIVE)).status_code == 200
        beat = asyncio.create_task(_heartbeat(ticks))
        # The loop's own tick rate on this machine, with nothing else running.
        await asyncio.sleep(1.0)
        free_rate = ticks[0] / 1.0
        ticks[0] = 0
        started = loop.time()
        export = asyncio.create_task(
            client.post(
                f"/api/v1/artifacts/workspace/local/{metadata['id']}/export", json={"format": "docx"}
            )
        )
        try:
            # Each probe is due every 0.1 s and timed from when it was due, so
            # a loop that wakes late counts against it, not only a slow answer.
            while not export.done():
                due = started + 0.1 * len(latencies)
                await asyncio.sleep(max(0.0, due - loop.time()))
                response = await client.get(LIVE)
                latencies.append(loop.time() - due)
                assert response.status_code == 200
            answered = await export
        finally:
            beat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await beat
        elapsed = loop.time() - started

    assert max(latencies) < 0.1, f"liveness took {max(latencies):.3f}s"
    # A conversion holding the server's GIL starves the loop of its ticks.
    assert ticks[0] >= 0.8 * free_rate * elapsed, f"{ticks[0]} ticks in {elapsed:.1f}s at {free_rate}/s"
    assert answered.status_code == 413, answered.text
    assert answered.json()["detail"] == "This file can't be exported."
    assert elapsed >= WALL_SECONDS - 1, "the export ended before the conversion was killed"


def _fake_child(monkeypatch, code: str) -> None:
    """Start *code* with ``python -I -c`` in place of the conversion process."""
    command, environment = export_runner.child_command(
        limits=ChildLimits(wall_seconds=WALL_SECONDS, memory_bytes=2048 * MIB)
    )
    fake = [command[0], "-I", "-B", "-c", code]
    monkeypatch.setattr(export_runner, "child_command", lambda *, limits: (fake, environment))


def _convert(*, tmp_path: Path, wall_seconds: float = WALL_SECONDS):
    with pinned_dir(tmp_path) as folder:
        return export_runner.convert_in_child(
            request=_request(suffix=".md"),
            source=b"# R\n",
            read_asset=folder_reader(folder=folder, max_bytes=1024),
            limits=ChildLimits(wall_seconds=wall_seconds, memory_bytes=2048 * MIB),
        )


_WRITE_CONVERTED = (
    "import sys\n"
    "from cowork.schemas.export_child import Converted\n"
    "from cowork.services.export_child import write_frame\n"
    "write_frame(stream=sys.stdout.buffer, message=Converted(omitted_count=2,"
    " scripts_dropped=True, denied_events={denied}), payload=b'out')\n"
)


def test_a_fake_result_is_taken_as_the_process_sends_it(monkeypatch, tmp_path):
    _fake_child(monkeypatch, _WRITE_CONVERTED.format(denied=0))

    conversion = _convert(tmp_path=tmp_path)

    assert (conversion.data, conversion.omitted_count, conversion.scripts_dropped) == (b"out", 2, True)


@pytest.mark.parametrize(
    ("code", "outcome"),
    [
        ("import os; os._exit(4)", export_runner.ConversionError),
        ("import os; os._exit(3)", export_runner.ConversionOverrun),
        (_WRITE_CONVERTED.format(denied=1), export_runner.ConversionError),
        # A length past the header cap, from a process that stays: killed, not waited out.
        (
            "import sys, time; sys.stdout.buffer.write(b'\\xff\\xff\\xff\\xff');"
            " sys.stdout.buffer.flush(); time.sleep(60)",
            export_runner.ConversionError,
        ),
        # A stalled process gets the wall clock.
        ("import time; time.sleep(60)", export_runner.ConversionOverrun),
        # Its error output goes nowhere, so a flood of it blocks nothing.
        ("import sys; sys.stderr.write('x' * 1_000_000)", export_runner.ConversionError),
    ],
    ids=[
        "exits-without-a-result",
        "exits-refused",
        "tried-to-connect",
        "oversized-frame",
        "stalls",
        "floods-its-error-output",
    ],
)
def test_each_way_a_process_ends_maps_to_its_answer(monkeypatch, tmp_path, code, outcome):
    _fake_child(monkeypatch, code)
    children = _descendants()
    started = time.monotonic()

    with pytest.raises(outcome):
        _convert(tmp_path=tmp_path, wall_seconds=3.0)

    assert time.monotonic() - started < 3.0 + 1
    assert _descendants() == children


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no signals")
def test_a_process_killed_by_a_signal_is_an_overrun(monkeypatch, tmp_path):
    _fake_child(monkeypatch, "import os, signal; os.kill(os.getpid(), signal.SIGKILL)")

    with pytest.raises(export_runner.ConversionOverrun) as overrun:
        _convert(tmp_path=tmp_path)

    assert overrun.value.cause == f"signal {int(signal.SIGKILL)}"


def _read_answering(answer):
    calls = []

    def read(parts):
        calls.append(parts)
        if isinstance(answer, OmittedReason):
            raise ArtifactFileRefused(answer, "refused")
        return answer

    return read, calls


def test_the_server_answers_only_reads_below_the_sources_folder():
    request = _request(suffix=".md").model_copy(update={"base_parts": ("docs",)})
    read, calls = _read_answering(ArtifactFileContent(data=b"png", device=1, inode=2))

    outside = export_runner._serve(
        asked=AssetRequest(parts=("other", "a.png")), request=request, read_asset=read, served=0
    )
    inside = export_runner._serve(
        asked=AssetRequest(parts=("docs", "a.png")), request=request, read_asset=read, served=0
    )

    assert outside == (AssetReply(refused=OmittedReason.MISSING), b"")
    assert inside == (AssetReply(device=1, inode=2), b"png")
    assert calls == [("docs", "a.png")]


def test_the_server_answers_no_more_reads_than_the_reference_cap():
    request = _request(suffix=".md")
    read, calls = _read_answering(ArtifactFileContent(data=b"png", device=1, inode=2))

    over = export_runner._serve(
        asked=AssetRequest(parts=("a.png",)),
        request=request,
        read_asset=read,
        served=request.limits.max_references,
    )

    assert over == (AssetReply(refused=OmittedReason.BUDGET), b"")
    assert calls == []


def test_a_refused_read_carries_its_reason_to_the_process():
    read, _calls = _read_answering(OmittedReason.SYMLINK)

    refused = export_runner._serve(
        asked=AssetRequest(parts=("a.png",)), request=_request(suffix=".md"), read_asset=read, served=0
    )

    assert refused == (AssetReply(refused=OmittedReason.SYMLINK), b"")


def test_a_name_holding_a_lone_surrogate_crosses_the_pipe(local_mode, tmp_path):
    """A CSS escape can name a file with a lone surrogate; it is left out as
    missing, as in the server's own process, and the export still writes."""
    page = (
        '<html><head><link rel="stylesheet" href="sheet.css"></head>'
        '<body><p style="background:url(a\\D800 .png)">x</p></body></html>'
    )
    case = capacity.Case(
        primary="report.html",
        build=lambda folder: (
            capacity._write(folder, "report.html", page),
            capacity._write(folder, "sheet.css", "p{background:url(b\\DC80 .png)}"),
        ),
        note="",
    )
    source, folder, metadata = _artifact(tmp_path / "general", case)

    result = _export(source, folder, metadata)

    assert result.omitted_count == 2


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="only Linux takes a file name that isn't UTF-8"
)
def test_a_source_in_a_folder_whose_name_isnt_utf8_exports(local_mode, tmp_path):
    name = os.fsdecode(b"caf\xe9")
    case = capacity.Case(
        primary=f"{name}/report.md",
        build=lambda folder: (
            (folder / name).mkdir(),
            capacity._write(folder / name, "report.md", "# R\n"),
        ),
        note="",
    )
    source, folder, metadata = _artifact(tmp_path / "general", case)

    result = _export(source, folder, metadata, export_format="html")

    assert (folder / name / result.filename).read_bytes().startswith(b"<!DOCTYPE html>")


_GUARD_PROBE = """
import socket, subprocess, sys
from cowork.services.export_child import _deny_network_and_processes
denied = _deny_network_and_processes()
for attempt in (
    lambda: socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=2),
    lambda: subprocess.run([sys.executable, "-c", "pass"]),
):
    try:
        attempt()
    except RuntimeError:
        pass
print(denied())
"""


def test_the_process_refuses_a_connection_and_a_process_start():
    with recording_server() as server:
        command, environment = export_runner.child_command(
            limits=ChildLimits(wall_seconds=WALL_SECONDS, memory_bytes=2048 * MIB)
        )
        finished = subprocess.run(
            [command[0], "-I", "-B", "-c", _GUARD_PROBE, str(server.server_port)],
            env=environment, capture_output=True, text=True, check=True, timeout=60,
        )
        hits = server.hits()

    assert finished.stdout.strip() == "2"
    assert hits == []


_CLOSURE_PROBE = """
import json, sys
import cowork, cowork.services.export_child
print(json.dumps({"cowork": cowork.__file__, "modules": sorted(sys.modules)}))
"""

# What the conversion process must never load: the server's settings,
# database, web framework and HTTP client, and the modules that hold them.
_NEVER_LOADED = (
    "cowork.common.settings",
    "cowork.db",
    "cowork.services.artifact_identity",
    "cowork.services.artifact_export",
    "cowork.services.export_runner",
    "fastapi",
    "starlette",
    "sqlalchemy",
    "sqlmodel",
    "httpx",
)


def test_the_process_runs_the_servers_own_cowork_and_loads_none_of_its_server_side():
    command, environment = export_runner.child_command(
        limits=ChildLimits(wall_seconds=WALL_SECONDS, memory_bytes=2048 * MIB)
    )
    finished = subprocess.run(
        [command[0], "-I", "-B", "-c", _CLOSURE_PROBE],
        env=environment, capture_output=True, text=True, check=True, timeout=60,
    )
    probe = json.loads(finished.stdout)

    assert Path(probe["cowork"]).resolve() == Path(cowork.__file__).resolve()
    loaded = [
        name
        for name in probe["modules"]
        if any(name == banned or name.startswith(banned + ".") for banned in _NEVER_LOADED)
    ]
    assert loaded == []
