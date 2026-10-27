"""Measure how long one Word export holds the event loop, and how much memory it takes.

Each case builds an artifact in a temporary folder and exports it to Word at
org caps through ``export_primary_file``, in a worker thread, the way the
export route runs it. Meanwhile a task on the event loop sleeps 10 ms at a time
and records how late it wakes. Pure Python in the worker thread gives the GIL
back every few milliseconds, but one long C call such as a regex match holds
it, and that shows up as one long wake-up.

Every case runs in its own process, so each case's peak RSS is its own. A
replica imports the whole server and warms its agent runtime before it
reports ready, so each case does both before it reads its baseline, and
reads it before it builds the artifact. Peak RSS is then what a replica peaks
at before its database and caches, which only add more. RSS added counts
building the case as well as the export, so it errs high. A case still
running at its time budget stops itself and reports the longest stall it saw.
Only a single stall past the grace period keeps it from doing that; the
parent then kills it and reports the grace as a lower bound.

The cases cover ordinary reports at the org caps (10 MiB inline budget, 10 MiB
HTML and 512 KiB Markdown read bounds), the heaviest HTML the caps allow, and
Markdown shapes that slow Python-Markdown down much faster than their size.

Run it with 1 CPU and 1 GiB of memory, with the repository mounted read-only:

    docker run --rm --cpus=1 --memory=1g -v "$PWD":/src:ro -w /src \\
      -e UV_PROJECT_ENVIRONMENT=/opt/venv -e PYTHONPATH=/src \\
      ghcr.io/astral-sh/uv:0.6.14-python3.12-bookworm-slim \\
      sh -c 'uv sync --frozen --group dev --no-install-project \\
        && uv run --no-sync python scripts/measure_export_capacity.py'

It prints one Markdown table row per case.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import anyio
from pydantic import BaseModel

MIB = 1024 * 1024
# Just under the export's read bounds, so every case is one the route accepts.
HTML_BYTES = 10 * MIB - 4096
MARKDOWN_BYTES = 512 * 1024 - 4096
# Just under the inliner's 2 MiB per-asset cap.
IMAGE_BYTES = 2 * MIB - 4096
WATCH_INTERVAL_SECONDS = 0.01
# How long past its budget a case may stay silent before the parent kills it.
GRACE_SECONDS = 30.0


class Row(BaseModel):
    """One case's measurement."""

    case: str
    wall_seconds: float
    longest_stall_ms: float
    baseline_rss_mib: float
    peak_rss_mib: float
    rss_added_mib: float
    result: str


@dataclass(frozen=True)
class Case:
    """An artifact to export: its primary file's name and a builder for its folder."""

    primary: str
    build: Callable[[Path], None]
    note: str


def _png(*, filler_bytes: int) -> bytes:
    """A valid 64 x 64 PNG padded with a text chunk to about *filler_bytes*."""

    def chunk(*, kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    header = struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)
    pixels = zlib.compress(b"".join(b"\x00" + b"\x80" * 64 * 3 for _ in range(64)))
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(kind=b"IHDR", body=header),
            chunk(kind=b"tEXt", body=b"filler\x00" + b"x" * filler_bytes),
            chunk(kind=b"IDAT", body=pixels),
            chunk(kind=b"IEND", body=b""),
        )
    )


def _report_markdown(*, target_bytes: int, images: int) -> str:
    """A long agent report: headings, paragraphs, tables, lists and a few images."""
    section = (
        "## Section {n}\n\n"
        + "Revenue grew in every region this quarter, led by the north, "
        "while costs held flat and margins widened by two points. " * 6
        + "\n\n| Region | Q1 | Q2 | Q3 | Q4 |\n|---|---|---|---|---|\n"
        + "| North | 10 | 12 | 14 | 16 |\n" * 10
        + "\n* first point\n* second point\n    * nested point\n\n"
    )
    pieces = ["# Quarterly report\n\n"]
    pieces += [f"![chart {n}](chart{n}.png)\n\n" for n in range(images)]
    size = sum(len(piece) for piece in pieces)
    n = 0
    while size < target_bytes:
        piece = section.format(n=n)
        pieces.append(piece)
        size += len(piece)
        n += 1
    return "".join(pieces)[:target_bytes]


def _report_html(*, target_bytes: int, images: int) -> str:
    row = "<tr><td>North</td><td>10</td><td>12</td><td>14</td><td>16</td></tr>"
    section = (
        "<h2>Section</h2><p>"
        + "Revenue grew in every region this quarter, led by the north. " * 6
        + "</p><table>"
        + row * 10
        + "</table><ul><li>first</li><li>second<ul><li>nested</li></ul></li></ul>"
    )
    head = "<!DOCTYPE html><html><head><title>Quarterly report</title></head><body><h1>Quarterly report</h1>"
    head += "".join(f'<img src="chart{n}.png" alt="chart {n}">' for n in range(images))
    tail = "</body></html>"
    body = section * ((target_bytes - len(head) - len(tail)) // len(section))
    return head + body + tail


def _write(folder: Path, name: str, data: str | bytes) -> None:
    path = folder / name
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)


def _with_images(folder: Path, *, count: int) -> None:
    image = _png(filler_bytes=IMAGE_BYTES)
    for n in range(count):
        _write(folder, f"chart{n}.png", image)


def _sized(*, unit: str, target_bytes: int) -> str:
    return unit * (target_bytes // len(unit))


def _markdown_case(text: Callable[[], str], note: str) -> Case:
    return Case(primary="report.md", build=lambda folder: _write(folder, "report.md", text()), note=note)


CASES: dict[str, Case] = {
    "md-report-with-images": Case(
        primary="report.md",
        build=lambda folder: (
            _with_images(folder, count=5),
            _write(folder, "report.md", _report_markdown(target_bytes=MARKDOWN_BYTES, images=5)),
        ),
        note="512 KiB report, five 2 MiB images (the inline budget)",
    ),
    "md-report-small": Case(
        primary="report.md",
        build=lambda folder: (
            _with_images(folder, count=2),
            _write(folder, "report.md", _report_markdown(target_bytes=200_000, images=2)),
        ),
        note="200 KB report, two 2 MiB images",
    ),
    "html-report-with-images": Case(
        primary="report.html",
        build=lambda folder: (
            _with_images(folder, count=5),
            _write(folder, "report.html", _report_html(target_bytes=HTML_BYTES, images=5)),
        ),
        note="10 MiB page, five 2 MiB images",
    ),
    "html-css-heavy": Case(
        primary="report.html",
        build=lambda folder: (
            _write(folder, "a.png", _png(filler_bytes=16)),
            _write(
                folder,
                "report.html",
                "<html><head><style>"
                + _sized(unit=".c{background:url(a.png)}", target_bytes=HTML_BYTES - 200)
                + "</style></head><body><p class=c>x</p></body></html>",
            ),
        ),
        note="10 MiB style element of url() declarations",
    ),
    "html-images-over-budget": Case(
        primary="report.html",
        build=lambda folder: (
            _with_images(folder, count=128),
            _write(
                folder,
                "report.html",
                "<html><body>" + "".join(f'<img src="chart{n}.png">' for n in range(128)) + "</body></html>",
            ),
        ),
        note="128 distinct 2 MiB images, 5 fit the budget",
    ),
    "html-css-at-cap": Case(
        primary="report.html",
        build=lambda folder: _write(
            folder,
            "report.html",
            "<html><head><style>"
            + _sized(unit=".a>b{c:d}", target_bytes=MIB - 64)
            + "</style></head><body><p>x</p></body></html>",
        ),
        note="a style element just under the 1 MiB CSS cap",
    ),
    "html-node-heavy": Case(
        primary="report.html",
        build=lambda folder: _write(
            folder,
            "report.html",
            "<html><body>" + _sized(unit="<p>x</p>", target_bytes=HTML_BYTES - 100) + "</body></html>",
        ),
        note="10 MiB of one-word paragraphs, past the node cap",
    ),
    "md-paragraphs": _markdown_case(
        lambda: _sized(unit="A plain sentence of ordinary report text.\n\n", target_bytes=MARKDOWN_BYTES),
        "512 KiB of plain paragraphs",
    ),
    "md-table": _markdown_case(
        lambda: "| a | b | c |\n|---|---|---|\n"
        + _sized(unit="| 1 | 2 | 3 |\n", target_bytes=MARKDOWN_BYTES - 64),
        "512 KiB table",
    ),
    "md-headings": _markdown_case(
        lambda: _sized(unit="# a\n\n", target_bytes=256 * 1024),
        "256 KiB of one repeated heading",
    ),
    "md-backtick-run": _markdown_case(lambda: "`" * 2_000, "2 KB of backticks"),
    "md-code-spans": _markdown_case(lambda: "`a` " * 20_000, "80 KB of inline code spans"),
    "md-reference-brackets": _markdown_case(lambda: "[a][" * 8_000, "32 KB of '[a]['"),
    "md-image-brackets": _markdown_case(lambda: "![" * 8_000, "16 KB of '!['"),
    "md-fence-ladder": _markdown_case(
        lambda: "\n".join("`" * (3 + n) for n in range(1_000)), "1,000 fence lines of growing length"
    ),
    "md-tag-runs": _markdown_case(lambda: "<a b" * 8_000, "32 KB of '<a b'"),
    "md-emphasis": _markdown_case(lambda: "***" + "a*" * 8_000, "16 KB of emphasis markers"),
    "md-deep-list": _markdown_case(
        lambda: "".join("    " * depth + "* x\n" for depth in range(400)), "a list nested 400 levels"
    ),
}


def _peak_rss_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return peak / (MIB if sys.platform == "darwin" else 1024)


def _replica_baseline_mib() -> float:
    """Peak RSS with the server imported and its agent runtime warmed, as a
    replica holds it once it reports ready."""
    import cowork.server

    cowork.server._warm_agent_runtime()
    return _peak_rss_mib()


@dataclass
class _Watch:
    longest: float = 0.0


async def _watch_loop(*, watch: _Watch, stop: anyio.Event, on_overrun: Callable[[], None]) -> None:
    while not stop.is_set():
        start = time.perf_counter()
        await anyio.sleep(WATCH_INTERVAL_SECONDS)
        watch.longest = max(watch.longest, time.perf_counter() - start - WATCH_INTERVAL_SECONDS)
        on_overrun()


def _measure_case(*, case_id: str, budget_seconds: float) -> Row:
    # Read before the case is built: building holds the source and its images
    # in memory once, as the export's read does.
    baseline = _replica_baseline_mib()
    from cowork.services import artifact_export
    from cowork.services.artifacts import ProjectArtifacts

    # The org inline budget, without an org deployment's database and identity,
    # with org export on: this measures what turning it on would cost.
    artifact_export._org_mode = lambda: True
    artifact_export.ORG_EXPORT_ENABLED = True
    case = CASES[case_id]
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "general"
        folder = project / ".anton" / "artifacts" / "report"
        folder.mkdir(parents=True)
        case.build(folder)
        metadata = {"id": uuid4().hex, "slug": "report", "type": "document", "primary": case.primary}
        _write(folder, "metadata.json", json.dumps(metadata))
        source = ProjectArtifacts(
            base=project / ".anton" / "artifacts",
            project_id=None,
            project_name="general",
            trusted_anchor=project,
            root_parts=(".anton", "artifacts"),
        )
        watch = _Watch()
        started = time.perf_counter()

        def row(*, outcome: str) -> Row:
            peak = _peak_rss_mib()
            return Row(
                case=case_id,
                wall_seconds=round(time.perf_counter() - started, 2),
                longest_stall_ms=round(watch.longest * 1000, 1),
                baseline_rss_mib=round(baseline, 1),
                peak_rss_mib=round(peak, 1),
                rss_added_mib=round(peak - baseline, 1),
                result=outcome,
            )

        def on_overrun() -> None:
            # The worker thread can't be stopped, so the process reports and ends.
            if time.perf_counter() - started > budget_seconds:
                print(row(outcome=f"still running after {budget_seconds:.0f} s").model_dump_json(), flush=True)
                os._exit(0)

        async def run() -> str:
            stop = anyio.Event()
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(lambda: _watch_loop(watch=watch, stop=stop, on_overrun=on_overrun))
                try:
                    result = await anyio.to_thread.run_sync(
                        lambda: artifact_export.export_primary_file(
                            source=source,
                            folder_name="report",
                            metadata=metadata,
                            artifact_id=metadata["id"],
                            export_format="docx",
                            can_edit=True,
                        )
                    )
                finally:
                    stop.set()
            return f"ok, {result.omitted_count} omitted"

        try:
            outcome = anyio.run(run)
        except Exception as exc:  # the row records any failure; the next case still runs
            outcome = f"error: {type(exc).__name__}"
        return row(outcome=outcome)


def _run_isolated(*, case_id: str, budget_seconds: float) -> Row:
    try:
        finished = subprocess.run(
            [sys.executable, __file__, "--case", case_id, "--budget", str(budget_seconds)],
            capture_output=True,
            text=True,
            timeout=budget_seconds + GRACE_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Row(
            case=case_id,
            wall_seconds=budget_seconds + GRACE_SECONDS,
            longest_stall_ms=GRACE_SECONDS * 1000,
            baseline_rss_mib=0.0,
            peak_rss_mib=0.0,
            rss_added_mib=0.0,
            result=f"killed: one stall outlasted the {GRACE_SECONDS:.0f} s grace",
        )
    lines = finished.stdout.strip().splitlines()
    if finished.returncode != 0 or not lines:
        tail = (finished.stderr.strip().splitlines() or ["no output"])[-1]
        return Row(
            case=case_id,
            wall_seconds=0.0,
            longest_stall_ms=0.0,
            baseline_rss_mib=0.0,
            peak_rss_mib=0.0,
            rss_added_mib=0.0,
            result=f"exit {finished.returncode}: {tail[:120]}",
        )
    return Row.model_validate_json(lines[-1])


def _limits() -> str:
    """The CPU and memory the process may use, as cgroup v2 reports them on Linux."""
    found = []
    for name in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            found.append(f"{name}={path.read_text().strip()}")
    return ", ".join(found) or "no cgroup limits found"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=sorted(CASES), help="measure one case in this process")
    parser.add_argument("--only", nargs="*", choices=sorted(CASES), help="cases to run (default: all)")
    parser.add_argument("--budget", type=float, default=120.0, help="seconds a case may run")
    args = parser.parse_args()
    if args.case:
        print(_measure_case(case_id=args.case, budget_seconds=args.budget).model_dump_json())
        return
    print(f"Python {sys.version.split()[0]} on {sys.platform}; {_limits()}")
    print()
    print(
        "| Case | What it holds | Wall s | Longest stall ms | Baseline RSS MiB | Peak RSS MiB "
        "| RSS added MiB | Result |"
    )
    print("| -- | -- | -- | -- | -- | -- | -- | -- |")
    for case_id in args.only or list(CASES):
        row = _run_isolated(case_id=case_id, budget_seconds=args.budget)
        print(
            f"| {row.case} | {CASES[case_id].note} | {row.wall_seconds} | {row.longest_stall_ms} "
            f"| {row.baseline_rss_mib} | {row.peak_rss_mib} | {row.rss_added_mib} | {row.result} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
