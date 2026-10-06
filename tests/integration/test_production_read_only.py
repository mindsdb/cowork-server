"""Read-only smoke against the production Cowork deployment.

This module is selected explicitly by the production nightly. The broad
integration target excludes it so staging and post-deploy runs cannot change
its target or identity mode by accident.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import httpx
import pytest

from tests.integration.prereq import missing_prerequisite
from tests.integration.test_post_deploy import (
    PROD_COWORK_BASE_URL,
    _headers,
    _verified_prod_standing_identity,
)

pytestmark = [pytest.mark.postdeploy, pytest.mark.production_read_only]


@dataclass(frozen=True)
class ReadOnlyEndpoint:
    """A GET whose response proves one deployed read path is available."""

    name: str
    path: str
    collection_key: str | None = None
    expected_status: str | None = None


class ReadResponse(Protocol):
    """Response fields consumed by the read-only assertions."""

    status_code: int
    text: str

    def json(self) -> object: ...


READ_ONLY_ENDPOINTS = (
    ReadOnlyEndpoint(
        name="service health",
        path="/api/v1/health/",
        expected_status="ok",
    ),
    ReadOnlyEndpoint(
        name="conversation listing",
        path="/api/v1/conversations/?project=all&limit=1",
        collection_key="conversations",
    ),
    ReadOnlyEndpoint(
        name="schedule listing",
        path="/api/v1/schedules/",
        collection_key="schedules",
    ),
    ReadOnlyEndpoint(
        name="file listing",
        path="/api/v1/files/",
        collection_key="data",
    ),
    ReadOnlyEndpoint(
        name="pin listing",
        path="/api/v1/pins/",
        collection_key="pins",
    ),
)


def _production_identity() -> dict[str, str]:
    """Resolve the standing identity only in the explicit production mode."""
    if os.environ.get("COWORK_TEST_IDENTITY_MODE") != "standing":
        missing_prerequisite(
            "the production read-only smoke requires "
            "COWORK_TEST_IDENTITY_MODE=standing"
        )
    return _verified_prod_standing_identity()


def _assert_read_response(
    response: ReadResponse,
    endpoint: ReadOnlyEndpoint,
) -> None:
    """Require a successful JSON response with the endpoint's stable shape."""
    assert response.status_code == 200, (
        f"{endpoint.name} returned HTTP {response.status_code}: "
        f"{response.text[:300]}"
    )
    try:
        payload = response.json()
    except ValueError:
        pytest.fail(f"{endpoint.name} returned a non-JSON response")
    if not isinstance(payload, dict):
        pytest.fail(f"{endpoint.name} returned {type(payload).__name__}, not an object")
    if endpoint.expected_status is not None:
        assert payload.get("status") == endpoint.expected_status, payload
    if endpoint.collection_key is not None:
        assert isinstance(payload.get(endpoint.collection_key), list), payload


@pytest.fixture(scope="session")
def production_api() -> Iterator[httpx.Client]:
    """Authenticated client that refuses redirects."""
    identity = _production_identity()
    with httpx.Client(
        base_url=PROD_COWORK_BASE_URL,
        headers=_headers(identity),
        follow_redirects=False,
        timeout=30.0,
    ) as client:
        yield client


@pytest.mark.parametrize(
    "endpoint",
    READ_ONLY_ENDPOINTS,
    ids=lambda endpoint: endpoint.name,
)
def test_production_read_path(
    production_api: httpx.Client,
    endpoint: ReadOnlyEndpoint,
) -> None:
    """Each production assertion is one explicit GET with no cleanup write."""
    response = production_api.get(endpoint.path)
    _assert_read_response(response, endpoint)
