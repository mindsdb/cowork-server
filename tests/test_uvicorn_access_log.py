"""Uvicorn's access line keeps each request's path and redacts its query."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._uvicorn_harness import Launch, run_uvicorn_app

PRIVATE = "access_query_private_marker"

_APP = """
    from fastapi import FastAPI
    from cowork.common.logger import setup_logging

    # cowork.server runs this at import, under every launcher.
    setup_logging()
    app = FastAPI()

    @app.get("/ok")
    def ok():
        return {}
"""


@pytest.mark.parametrize("launch", ["module", "run"])
def test_uvicorns_access_line_keeps_the_path_and_redacts_the_query(tmp_path: Path, launch: Launch) -> None:
    run = run_uvicorn_app(tmp_path=tmp_path, app_source=_APP, paths=(f"/ok?code={PRIVATE}", "/ok"), launch=launch)
    assert run.statuses == (200, 200)
    events = [json.loads(line) for line in run.stderr.splitlines()]
    access = [event for event in events if event.get("event") == "http_access"]
    assert len(access) == 2
    assert all(event["status_code"] == 200 and event["method"] == "GET" for event in access)
    assert "/ok" not in run.stdout + run.stderr
    assert "Finished server process" not in run.stderr
    assert PRIVATE not in run.stdout + run.stderr
