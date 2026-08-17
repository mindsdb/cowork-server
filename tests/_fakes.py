"""Shared test doubles (importing conftest directly double-loads it)."""


class FakeRequest:
    """Minimal Request stub for calling route handlers directly. `headers` is
    empty unless a test sets it (only the org-mode bearer path reads it)."""

    def __init__(self, headers: dict | None = None) -> None:
        self.headers = headers or {}
