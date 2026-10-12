"""Uvicorn's access line keeps each request's path and redacts its query."""
from __future__ import annotations

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
    assert '"GET /ok?[redacted] HTTP/1.1" 200' in run.stdout, run.stdout
    assert '"GET /ok HTTP/1.1" 200' in run.stdout, run.stdout
    assert "Finished server process" in run.stderr
    assert PRIVATE not in run.stdout + run.stderr
