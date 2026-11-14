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
import statistics
import struct
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
from cowork.common.structured_logging import make_formatter
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

HOSTILE_TIME_CASES = ("md-unclosed-fences", "md-reference-brackets", "md-open-brackets")

# Every test here starts real conversion processes, which import cowork from
# the interpreter's own site-packages.
pytestmark = pytest.mark.usefixtures("child_runs_this_checkout")


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
        record.reason
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
from cowork.schemas.artifact_export import InlineLimits
from cowork.schemas.export_child import ConvertRequest
from cowork.services import artifact_export
from cowork.services.artifacts import ProjectArtifacts
from cowork.services.export_child import convert

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

    def peak_from_here():
        # The peak RSS from now on, in MiB.
        if sys.platform.startswith("linux"):
            # Linux resets the high-water mark on request: an import's peak
            # could otherwise hide the export's.
            Path("/proc/self/clear_refs").write_text("5")
            def read():
                status = Path("/proc/self/status").read_text()
                return int(status.split("VmHWM:")[1].split()[0]) / 1024
        else:
            read = capacity._peak_rss_mib
        start = read()
        return lambda: read() - start

    server = peak_from_here()
    artifact_export.export_primary_file(source=source, folder_name="report",
        metadata={"id": "a" * 32, "primary": "report.md", "type": "document"},
        artifact_id="a" * 32, export_format="docx", can_edit=True)
    server = server()
    # The same conversion in this process, as before it moved out.
    in_process = peak_from_here()
    request = ConvertRequest(export_format="docx", suffix=".md", title="report",
        base_parts=(), limits=InlineLimits(max_total_bytes=25 * 1024 * 1024))
    convert(request=request, source=table.encode(), read_asset=lambda parts: None)
    print(json.dumps({"server": server, "in_process": in_process()}))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="Python reports no peak RSS on Windows")
def test_the_servers_own_peak_memory_does_not_grow_with_the_input(local_mode):
    """Each size runs in a fresh interpreter, so each peak is that export's own.

    The same conversion run in the server's own process afterwards grows with
    the input, which shows the input is large enough to tell the two apart.
    """
    peaks = {}
    for size in (64 * 1024, 500 * 1024):
        finished = subprocess.run(
            [sys.executable, "-c", _PEAK_PROBE, str(size)],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        peaks[size] = json.loads(finished.stdout.strip().splitlines()[-1])

    small, large = peaks[64 * 1024], peaks[500 * 1024]
    assert large["server"] - small["server"] < 15, peaks
    assert large["in_process"] - small["in_process"] > 40, peaks


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

    # A shared CI runner can stall one probe on its own, so the check is the
    # 95th percentile against the ticket's 100 ms, and every probe far under
    # the seconds the in-process conversion held the loop.
    assert statistics.quantiles(latencies, n=20)[-1] < 0.1, sorted(latencies)[-5:]
    assert max(latencies) < 0.5, f"liveness took {max(latencies):.3f}s"
    # A conversion holding the server's GIL starves the loop of its ticks.
    assert ticks[0] >= 0.5 * free_rate * elapsed, f"{ticks[0]} ticks in {elapsed:.1f}s at {free_rate}/s"
    assert answered.status_code == 413, answered.text
    assert answered.json()["detail"] == "This file can't be exported."
    assert elapsed >= WALL_SECONDS - 1, "the export ended before the conversion was killed"


def _fake_child(monkeypatch, code: str) -> None:
    """Start *code* with ``python -I -c`` in place of the conversion process."""
    command = export_runner.child_command(
        limits=ChildLimits(wall_seconds=WALL_SECONDS, memory_bytes=2048 * MIB)
    )
    fake = export_runner.ChildCommand(
        argv=[command.argv[0], "-I", "-B", "-c", code], environment=command.environment
    )
    monkeypatch.setattr(export_runner, "child_command", lambda *, limits: fake)


def _convert(*, tmp_path: Path, wall_seconds: float = WALL_SECONDS):
    with pinned_dir(tmp_path) as folder:
        return export_runner.convert_in_child(
            request=_request(suffix=".md"),
            source=b"# R\n",
            read_asset=folder_reader(folder=folder, max_bytes=1024),
            limits=ChildLimits(wall_seconds=wall_seconds, memory_bytes=2048 * MIB),
        )


def _raw_frame(header: dict) -> str:
    """Code for a process that writes one frame of *header*, valid or not, and exits."""
    encoded = json.dumps(header).encode()
    frame = struct.pack(">I", len(encoded)) + encoded + struct.pack(">I", 0)
    return f"import sys; sys.stdout.buffer.write({frame!r}); sys.stdout.buffer.flush()"


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
        # Its error output is drained as it comes, so a flood of it blocks nothing.
        ("import sys; sys.stderr.write('x' * 1_000_000)", export_runner.ConversionError),
        # A frame the stream cuts short.
        (
            "import sys; sys.stdout.buffer.write(b'\\x00\\x00\\x00\\x0a{}');"
            " sys.stdout.buffer.flush()",
            export_runner.ConversionError,
        ),
        (_raw_frame({"type": "bogus"}), export_runner.ConversionError),
        (
            _raw_frame(
                {
                    "type": "converted",
                    "omitted_count": -1,
                    "scripts_dropped": False,
                    "denied_events": 0,
                }
            ),
            export_runner.ConversionError,
        ),
    ],
    ids=[
        "exits-without-a-result",
        "exits-refused",
        "tried-to-connect",
        "oversized-frame",
        "stalls",
        "floods-its-error-output",
        "cuts-a-frame-short",
        "sends-an-unknown-message",
        "reports-a-negative-count",
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


def test_the_watch_kills_a_process_that_holds_more_than_its_memory_limit(monkeypatch, tmp_path):
    """The server's own watch, with no limit the process sets on itself: the
    only memory limit on macOS and Windows."""
    _fake_child(
        monkeypatch, "import time; held = b'x' * (400 * 1024 * 1024); time.sleep(60)"
    )
    children = _descendants()
    started = time.monotonic()

    with pytest.raises(export_runner.ConversionOverrun) as overrun:
        with pinned_dir(tmp_path) as folder:
            export_runner.convert_in_child(
                request=_request(suffix=".md"),
                source=b"# R\n",
                read_asset=folder_reader(folder=folder, max_bytes=1024),
                limits=ChildLimits(wall_seconds=30.0, memory_bytes=200 * MIB),
            )

    assert overrun.value.cause == "memory"
    assert time.monotonic() - started < 15
    assert _descendants() == children


def test_the_watch_kills_a_process_whose_memory_it_cant_read(monkeypatch, tmp_path):
    """On macOS and Windows the watch is the only memory limit, so a process it
    can't read doesn't run on unwatched, and the wall clock never goes with it."""

    def unreadable(*, watched):
        raise psutil.AccessDenied(watched.pid)

    monkeypatch.setattr(export_runner, "_held_bytes", unreadable)
    _fake_child(monkeypatch, "import time; time.sleep(20)")
    children = _descendants()
    started = time.monotonic()

    with pytest.raises(export_runner.ConversionOverrun) as overrun:
        _convert(tmp_path=tmp_path, wall_seconds=30.0)

    assert overrun.value.cause == "memory unreadable"
    assert time.monotonic() - started < 5
    assert _descendants() == children


def test_an_error_while_talking_to_the_process_kills_it_at_once(monkeypatch, tmp_path):
    """Waited for instead, a busy process would hold the export until its wall clock."""

    def broken(**_kwargs):
        raise RuntimeError("the server failed mid-conversion")

    monkeypatch.setattr(export_runner, "_talk", broken)
    _fake_child(monkeypatch, "import time; time.sleep(20)")
    children = _descendants()
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="the server failed mid-conversion"):
        _convert(tmp_path=tmp_path, wall_seconds=15.0)

    assert time.monotonic() - started < 5
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


def test_a_folder_and_title_holding_a_lone_surrogate_cross_the_pipe(tmp_path):
    """A folder name that isn't UTF-8 reaches the server as a lone surrogate,
    in the source's folder and title alike, and the request still converts."""
    name = "caf\udce9"
    request = _request(suffix=".html").model_copy(update={"base_parts": (name,), "title": name})
    with pinned_dir(tmp_path) as folder:
        conversion = export_runner.convert_in_child(
            request=request,
            source=b'<html><body><p>R</p><img src="missing.png"></body></html>',
            read_asset=folder_reader(folder=folder, max_bytes=1024),
            limits=ChildLimits(wall_seconds=WALL_SECONDS, memory_bytes=2048 * MIB),
        )

    assert conversion.omitted_count == 1
    assert docx.Document(io.BytesIO(conversion.data)).paragraphs


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
            env=environment, capture_output=True, text=True, timeout=60,
        )
        hits = server.hits()

    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip() == "2"
    assert hits == []


_OPENS_A_SOCKET = """
import socket, sys
from cowork.services import export_child

def convert(**_kwargs):
    try:
        socket.socket()
    except RuntimeError:
        pass
    return export_child.Conversion(data=b"out", omitted_count=0, scripts_dropped=False)

export_child.convert = convert
sys.argv = ["export_child", str(2048 * 1024 * 1024), "31"]
export_child.main()
"""


def test_a_conversion_that_opens_a_socket_fails_the_export(monkeypatch, tmp_path):
    """Through ``main``, so the export fails only if ``main`` installs the refusal."""
    _fake_child(monkeypatch, _OPENS_A_SOCKET)

    with pytest.raises(export_runner.ConversionError, match="tried to connect"):
        _convert(tmp_path=tmp_path)


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
        env=environment, capture_output=True, text=True, timeout=60,
    )
    assert finished.returncode == 0, finished.stderr
    probe = json.loads(finished.stdout)

    assert Path(probe["cowork"]).resolve() == Path(cowork.__file__).resolve()
    loaded = [
        name
        for name in probe["modules"]
        if any(name == banned or name.startswith(banned + ".") for banned in _NEVER_LOADED)
    ]
    assert loaded == []


def test_the_shipped_limits_are_thirty_seconds_and_two_gib_or_640_mib_in_org_mode():
    assert artifact_export.conversion_limits(org_mode=False) == ChildLimits(
        wall_seconds=30.0, memory_bytes=2 * 1024 * MIB
    )
    assert artifact_export.conversion_limits(org_mode=True) == ChildLimits(
        wall_seconds=30.0, memory_bytes=640 * MIB
    )


def test_the_process_starts_isolated_with_only_the_variables_that_find_its_home(monkeypatch):
    monkeypatch.setattr(export_runner.site, "ENABLE_USER_SITE", False)
    monkeypatch.setenv("HOME", "home-folder")
    monkeypatch.setenv("COWORK_SECRET_FOR_THIS_TEST", "never-passed")

    command = export_runner.child_command(
        limits=ChildLimits(wall_seconds=30.0, memory_bytes=640 * MIB)
    )

    assert command.argv == [
        sys.executable,
        "-I",
        "-B",
        "-m",
        "cowork.services.export_child",
        str(640 * MIB),
        "31",
    ]
    expected = {"HOME"} | ({"SYSTEMROOT", "USERPROFILE"} if sys.platform == "win32" else set())
    assert set(command.environment) == expected
    assert command.environment["HOME"] == "home-folder"


def test_a_server_in_the_users_site_folder_keeps_that_folder_for_the_process(monkeypatch):
    """``pip install --user`` puts cowork in the user's site folder, which ``-I`` would
    drop; a custom user base comes along too."""
    monkeypatch.setattr(export_runner.site, "ENABLE_USER_SITE", True)
    monkeypatch.setattr(export_runner.site, "getuserbase", lambda: "custom-user-base")

    command = export_runner.child_command(
        limits=ChildLimits(wall_seconds=30.0, memory_bytes=640 * MIB)
    )

    assert command.argv[1:4] == ["-E", "-P", "-B"]
    assert command.environment["PYTHONUSERBASE"] == "custom-user-base"


_HOME_PROBE = """
import pwd

def no_entry(uid):
    raise KeyError(uid)

# A uid the image's /etc/passwd doesn't list, as with docker run --user.
pwd.getpwuid = no_entry
from cowork.common import paths
print(paths._DEFAULT_HOME)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="Windows finds the home folder from USERPROFILE")
def test_the_process_finds_its_home_folder_without_a_passwd_entry(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    command = export_runner.child_command(
        limits=ChildLimits(wall_seconds=WALL_SECONDS, memory_bytes=2048 * MIB)
    )

    finished = subprocess.run(
        [command.argv[0], "-I", "-B", "-c", _HOME_PROBE],
        env=command.environment, capture_output=True, text=True, timeout=60,
    )

    assert finished.returncode == 0, finished.stderr
    assert Path(finished.stdout.strip()) == tmp_path / ".cowork"


_LIMITS_PROBE = """
import json, os, resource, subprocess, sys
from cowork.services.export_child import _limit_self

_limit_self(memory_bytes=2048 * 1024 * 1024, cpu_seconds=31)
seen = {}
try:
    with open(sys.argv[1], "wb") as written:
        written.write(b"x")
    seen["write"] = "allowed"
except OSError:
    seen["write"] = "refused"
try:
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    seen["fork"] = "allowed"
except OSError:
    seen["fork"] = "refused"
try:
    subprocess.run([sys.executable, "-c", "pass"], check=False)
    seen["subprocess"] = "allowed"
except OSError:
    seen["subprocess"] = "refused"
seen["core"] = list(resource.getrlimit(resource.RLIMIT_CORE))
seen["cpu"] = list(resource.getrlimit(resource.RLIMIT_CPU))
seen["nice"] = os.getpriority(os.PRIO_PROCESS, 0)
if sys.platform.startswith("linux"):
    with open("/proc/self/oom_score_adj", encoding="ascii") as score:
        seen["oom_score_adj"] = int(score.read())
    seen["address_space"] = list(resource.getrlimit(resource.RLIMIT_AS))
print(json.dumps(seen))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="Windows sets no limits on the process itself")
@pytest.mark.skipif(
    sys.platform != "win32" and os.geteuid() == 0, reason="root passes the process limit"
)
def test_the_process_limits_itself_before_it_reads_a_document(tmp_path):
    """The limits ``_limit_self`` sets, read back inside a process that set them.

    The probe reports through a pipe: with no room to write a file, a
    file-backed standard output would fail too.
    """
    command = export_runner.child_command(
        limits=ChildLimits(wall_seconds=WALL_SECONDS, memory_bytes=2048 * MIB)
    )
    target = tmp_path / "written.bin"

    finished = subprocess.run(
        [command.argv[0], "-I", "-B", "-c", _LIMITS_PROBE, str(target)],
        env=command.environment, capture_output=True, text=True, timeout=60,
    )

    assert finished.returncode == 0, finished.stderr
    seen = json.loads(finished.stdout)
    assert (seen["write"], seen["fork"], seen["subprocess"]) == ("refused",) * 3, seen
    assert not target.exists() or target.stat().st_size == 0
    assert seen["core"] == [0, 0]
    assert seen["cpu"] == [31, 32]
    if sys.platform.startswith("linux"):
        # GitHub's macOS runners leave a process's priority at 0 after
        # os.nice, so it is checked where the self-hosted stack runs.
        assert seen["nice"] >= 10
        assert seen["oom_score_adj"] == 1000
        assert seen["address_space"] == [2048 * MIB, 2048 * MIB]


def _formatted(record: logging.LogRecord) -> dict:
    """The record as the server's log formatter writes it in production."""
    return json.loads(make_formatter().format(record))


def _runner_records(caplog) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == RUNNER_LOGGER]


def test_the_log_line_says_why_the_server_stopped_a_conversion(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("LOG_FORMAT", "json")
    _fake_child(monkeypatch, "import time; time.sleep(60)")

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        with pytest.raises(export_runner.ConversionOverrun):
            _convert(tmp_path=tmp_path, wall_seconds=1.0)

    [record] = _runner_records(caplog)
    line = _formatted(record)
    assert (line["event"], line["reason"]) == ("export_conversion_stopped", "wall clock")


_FAILS_WITH_A_VALUE = """
import sys
from cowork.services import export_child

def convert(**_kwargs):
    raise ValueError("text from the document")

export_child.convert = convert
sys.argv = ["export_child", str(2048 * 1024 * 1024), "31"]
export_child.main()
"""


def test_a_failed_conversion_logs_where_it_failed_but_not_the_documents_values(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setenv("LOG_FORMAT", "json")
    _fake_child(monkeypatch, _FAILS_WITH_A_VALUE)

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        with pytest.raises(export_runner.ConversionError):
            _convert(tmp_path=tmp_path)

    [record] = _runner_records(caplog)
    line = _formatted(record)
    assert (line["event"], line["error_type"]) == ("export_conversion_failed", "ValueError")
    assert "in convert" in line["stack"]
    assert "text from the document" not in json.dumps(line)


_REFUSES_WITH_A_VALUE = """
import sys
from cowork.services import export_child
from cowork.services.markdown_render import MarkdownRefused

def convert(**_kwargs):
    try:
        raise ValueError("text from the document")
    except ValueError as exc:
        raise MarkdownRefused("Python-Markdown can't convert this document") from exc

export_child.convert = convert
sys.argv = ["export_child", str(2048 * 1024 * 1024), "31"]
export_child.main()
"""


def test_a_refused_conversion_logs_what_refused_it_but_not_the_documents_values(
    monkeypatch, tmp_path, caplog
):
    """A real bug behind a refusal, such as an extension's ``ValueError``, shows
    in the log instead of passing as a silent 413."""
    monkeypatch.setenv("LOG_FORMAT", "json")
    _fake_child(monkeypatch, _REFUSES_WITH_A_VALUE)

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        with pytest.raises(export_runner.ConversionOverrun):
            _convert(tmp_path=tmp_path)

    [record] = _runner_records(caplog)
    line = _formatted(record)
    assert (line["event"], line["reason"]) == ("export_conversion_stopped", "refused")
    assert "MarkdownRefused: [error details omitted]" in line["stack"]
    assert "ValueError: [error details omitted]" in line["stack"]
    assert "in convert" in line["stack"]
    assert "text from the document" not in json.dumps(line)


def test_the_process_starts_in_the_filesystem_root(monkeypatch, tmp_path, caplog):
    """No folder of the server's, which ``-I`` keeps off the import path but not out of reach."""
    monkeypatch.setenv("LOG_FORMAT", "json")
    _fake_child(monkeypatch, "import os, sys; sys.stderr.write(os.getcwd()); sys.exit(1)")

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        with pytest.raises(export_runner.ConversionError):
            _convert(tmp_path=tmp_path)

    [record] = _runner_records(caplog)
    assert _formatted(record)["stack"] == os.path.abspath(os.sep)


def test_a_process_that_fails_before_it_reports_leaves_its_last_error_in_the_log(
    monkeypatch, tmp_path, caplog
):
    """Such as an import that failed: the process never reached its own report."""
    monkeypatch.setenv("LOG_FORMAT", "json")
    _fake_child(
        monkeypatch,
        "import sys; sys.stderr.write('x' * 5000 + 'RuntimeError: the import failed');"
        " sys.stderr.flush(); sys.exit(1)",
    )

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        with pytest.raises(export_runner.ConversionError):
            _convert(tmp_path=tmp_path)

    [record] = _runner_records(caplog)
    line = _formatted(record)
    assert (line["event"], line["reason"]) == ("export_conversion_no_result", "exited 1")
    assert line["stack"].endswith("RuntimeError: the import failed")
    assert len(line["stack"].encode()) <= 2048


_REPORTS_WITHOUT_READING = """
import os, sys
from cowork.schemas.export_child import ConversionFailed
from cowork.services.export_child import write_frame

write_frame(
    stream=sys.stdout.buffer,
    message=ConversionFailed(error_type="OSError", detail='  File "export_child.py", line 1'),
)
os._exit(4)
"""


def test_a_process_that_reports_before_reading_a_large_request_is_logged_as_failed(
    monkeypatch, tmp_path, caplog
):
    """Such as one a host refused a limit: the request fills the pipe and its
    write fails, and the server still reads why."""
    monkeypatch.setenv("LOG_FORMAT", "json")
    _fake_child(monkeypatch, _REPORTS_WITHOUT_READING)

    with caplog.at_level(logging.WARNING, logger=RUNNER_LOGGER):
        with pytest.raises(export_runner.ConversionError):
            with pinned_dir(tmp_path) as folder:
                export_runner.convert_in_child(
                    request=_request(suffix=".md"),
                    source=b"x" * (4 * MIB),
                    read_asset=folder_reader(folder=folder, max_bytes=1024),
                    limits=ChildLimits(wall_seconds=WALL_SECONDS, memory_bytes=2048 * MIB),
                )

    [record] = _runner_records(caplog)
    line = _formatted(record)
    assert (line["event"], line["error_type"]) == ("export_conversion_failed", "OSError")


def test_an_image_name_too_long_to_ask_for_is_left_out_and_the_export_still_writes(tmp_path):
    """Asked for, the name would pass the frame's header bound and fail the export."""
    page = f'<html><body><p>R</p><img src="{"a" * 70_000}.png"></body></html>'
    with pinned_dir(tmp_path) as folder:
        conversion = export_runner.convert_in_child(
            request=_request(suffix=".html"),
            source=page.encode(),
            read_asset=folder_reader(folder=folder, max_bytes=1024),
            limits=ChildLimits(wall_seconds=WALL_SECONDS, memory_bytes=2048 * MIB),
        )

    assert conversion.omitted_count == 1
    assert docx.Document(io.BytesIO(conversion.data)).paragraphs


_OUTPUT_PAST_THE_FRAME_BOUND = """
import sys
from cowork.services import export_child

export_child.MAX_PAYLOAD_BYTES = 16
sys.argv = ["export_child", str(2048 * 1024 * 1024), "31"]
export_child.main()
"""


def test_an_output_past_the_frame_bound_is_refused_as_too_large(monkeypatch, tmp_path):
    _fake_child(monkeypatch, _OUTPUT_PAST_THE_FRAME_BOUND)

    with pytest.raises(export_runner.ConversionOverrun) as overrun:
        _convert(tmp_path=tmp_path)

    assert overrun.value.cause == "refused"


_ASKS_FOR = """
import sys
from cowork.schemas.export_child import AssetRequest, Converted
from cowork.services.export_child import _SERVER_MESSAGE, read_frame, write_frame

frames_in, frames_out = sys.stdin.buffer, sys.stdout.buffer
read_frame(stream=frames_in, adapter=_SERVER_MESSAGE)
answers = []
for parts in {parts!r}:
    write_frame(stream=frames_out, message=AssetRequest(parts=parts))
    reply = read_frame(stream=frames_in, adapter=_SERVER_MESSAGE)
    answers.append(f"{{reply.message.refused}}:{{reply.payload.decode()}}")
write_frame(
    stream=frames_out,
    message=Converted(omitted_count=0, scripts_dropped=False, denied_events=0),
    payload="|".join(answers).encode(),
)
"""


def test_a_process_that_asks_for_files_outside_its_folder_gets_the_pinned_readers_refusals(
    monkeypatch, tmp_path
):
    """A process a document took over asks for what it likes; the server's own
    reader still decides, as it does for the inliner's reads."""
    (tmp_path / "secret.txt").write_text("secret")
    art = tmp_path / "art"
    art.mkdir()
    (art / "a.png").write_text("png")
    asks = [("..", "secret.txt"), ("/etc", "passwd"), ("a.png",)]
    expected = ["traversal:", "traversal:", "None:png"]
    if sys.platform != "win32":
        (art / "link.png").symlink_to(tmp_path / "secret.txt")
        asks.append(("link.png",))
        expected.append("symlink:")
    _fake_child(monkeypatch, _ASKS_FOR.format(parts=tuple(asks)))

    with pinned_dir(art) as folder:
        conversion = export_runner.convert_in_child(
            request=_request(suffix=".md"),
            source=b"# R\n",
            read_asset=folder_reader(folder=folder, max_bytes=1024),
            limits=ChildLimits(wall_seconds=WALL_SECONDS, memory_bytes=2048 * MIB),
        )

    assert conversion.data.decode().split("|") == expected


_STOPS_AT_ONCE = """
import sys
from cowork.services import {stage}, export_child

# Past at once at any clock resolution (Windows' 3.12 monotonic clock ticks every 15.6 ms).
{stage}.DEADLINE_SECONDS = -1.0
sys.argv = ["export_child", str(2048 * 1024 * 1024), "31"]
export_child.main()
"""


@pytest.mark.parametrize("stage", ["artifact_inline", "docx_writer"])
def test_a_stage_that_stops_at_its_deadline_still_writes_the_file_with_its_note(
    monkeypatch, tmp_path, stage
):
    """Through the process: the inliner's and the Word writer's own stops."""
    _fake_child(monkeypatch, _STOPS_AT_ONCE.format(stage=stage))
    page = "<html><body>" + "<p>paragraph</p>" * 200 + "</body></html>"

    with pinned_dir(tmp_path) as folder:
        conversion = export_runner.convert_in_child(
            request=_request(suffix=".html"),
            source=page.encode(),
            read_asset=folder_reader(folder=folder, max_bytes=1024),
            limits=ChildLimits(wall_seconds=WALL_SECONDS, memory_bytes=2048 * MIB),
        )

    written = docx.Document(io.BytesIO(conversion.data))
    assert written.paragraphs[-1].text == TRUNCATION_NOTE
