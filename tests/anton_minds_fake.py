"""Replace anton's publisher transport with a test fake, for any anton.

Older anton's publisher sends its upload through ``minds_request`` (returns the
body); newer anton sends it through ``minds_request_with_status`` (returns
``(status, body)``). A test that fakes only one of them lets the other reach the
network. ``fake_minds_requests`` takes one fake written against the
``minds_request`` shape and installs it behind both names, wherever the
installed anton has them.
"""
from __future__ import annotations

from collections.abc import Callable


def fake_minds_requests(monkeypatch, fake: Callable[..., bytes | str]) -> None:
    import anton.publisher

    monkeypatch.setattr("anton.publisher.minds_request", fake)
    if hasattr(anton.publisher, "minds_request_with_status"):

        def with_status(url, api_key, **kwargs):
            body = fake(url, api_key, **kwargs)
            return 200, body.encode() if isinstance(body, str) else body

        monkeypatch.setattr("anton.publisher.minds_request_with_status", with_status)
