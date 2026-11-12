"""scripts/measure_export_capacity.py measures RSS the way a replica holds it.

A replica imports the whole server and warms its agent runtime before it
reports ready, so a case does both before it reads its baseline, and reads it
before it builds the artifact, whose source and images the build holds in
memory once.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from cowork.services import artifact_export
from scripts import measure_export_capacity as capacity

ROOT = Path(__file__).resolve().parents[1]


def test_a_case_reads_its_baseline_before_it_builds_the_artifact(monkeypatch):
    # The script turns org mode and org export on in its own process; put
    # both back for the tests that follow.
    monkeypatch.setattr(artifact_export, "_org_mode", artifact_export._org_mode)
    monkeypatch.setattr(artifact_export, "ORG_EXPORT_ENABLED", artifact_export.ORG_EXPORT_ENABLED)
    steps: list[str] = []

    def baseline() -> float:
        steps.append("baseline")
        return 200.0

    def build(folder: Path) -> None:
        steps.append("build")
        capacity._write(folder, "report.md", "# R\n")

    monkeypatch.setattr(capacity, "_replica_baseline_mib", baseline)
    monkeypatch.setattr(capacity, "_peak_rss_mib", lambda: 260.0)
    monkeypatch.setitem(capacity.CASES, "probe", capacity.Case(primary="report.md", build=build, note="probe"))

    row = capacity._measure_case(case_id="probe", budget_seconds=60.0)

    assert steps == ["baseline", "build"]
    assert row.result.startswith("ok"), row.result
    assert (row.baseline_rss_mib, row.peak_rss_mib, row.rss_added_mib) == (200.0, 260.0, 60.0)


def test_the_baseline_is_read_with_the_server_imported_and_its_agent_runtime_warmed():
    # The warm-up swallows import failures, so the probe checks sys.modules.
    probe = (
        "import sys\n"
        "from scripts import measure_export_capacity as capacity\n"
        "before = set(sys.modules)\n"
        "capacity._replica_baseline_mib()\n"
        "after = set(sys.modules)\n"
        "import cowork.server\n"
        "runtime = cowork.server.AGENT_RUNTIME_MODULES\n"
        "print('cowork.server' in before, 'cowork.server' in after)\n"
        "print(all(n in before for n in runtime), all(n in after for n in runtime))\n"
    )

    finished = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True
    )

    server, runtime = finished.stdout.strip().splitlines()[-2:]
    assert server.split() == ["False", "True"]
    assert runtime.split() == ["False", "True"]
