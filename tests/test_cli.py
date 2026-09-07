from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from cowork import cli


def test_server_shutdown_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    # An open event stream must not keep a terminated sidecar alive forever.
    monkeypatch.setattr(cli, "get_app_settings", lambda: SimpleNamespace(port=27966, host="127.0.0.1"))
    run = mock.Mock()
    monkeypatch.setattr(cli.uvicorn, "run", run)

    cli.main()

    run.assert_called_once()
    assert run.call_args.kwargs["timeout_graceful_shutdown"] == cli.SHUTDOWN_GRACE_SECONDS
    assert 0 < cli.SHUTDOWN_GRACE_SECONDS <= 10
    assert run.call_args.kwargs["host"] == "127.0.0.1"
    assert run.call_args.kwargs["port"] == 27966
