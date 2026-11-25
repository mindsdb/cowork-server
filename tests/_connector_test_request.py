"""Test requests for handcrafted connector forms in tests.

A single-user install refuses a handcrafted form without one
(``check_test_requests``), so every fixture form that is meant to pass carries
one built here.
"""
from __future__ import annotations

from typing import Any


def bearer_test_request(field: str = "token", url: str = "https://httpbin.org/bearer") -> dict[str, Any]:
    """A data-only test request that sends ``field`` as a bearer token."""
    return {"data": {"method": "GET", "url": url, "headers": {"Authorization": f"Bearer {{{{{field}}}}}"}}}


def with_test_request(form: dict[str, Any], field: str = "token") -> dict[str, Any]:
    """``form`` with a bearer test request at the top level, or on each method."""
    if form.get("methods"):
        return {**form, "methods": [{**m, "test_request": bearer_test_request(field)} for m in form["methods"]]}
    return {**form, "test_request": bearer_test_request(field)}
