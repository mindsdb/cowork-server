from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

logger = logging.getLogger(__name__)

# httpx timeouts apply per network operation, so a trickled body could hold
# startup open indefinitely. The deadline bounds the whole fetch; the shorter
# per-operation timeout bounds how far one read can overrun it.
_FETCH_DEADLINE_SECONDS = 15.0
_FETCH_OPERATION_TIMEOUT_SECONDS = 5.0


def _fetch_models(endpoint: str, token: str, client_version: str) -> object:
    deadline = time.monotonic() + _FETCH_DEADLINE_SECONDS
    with httpx.Client(timeout=_FETCH_OPERATION_TIMEOUT_SECONDS) as client:
        with client.stream(
            "GET",
            f"{endpoint.rstrip('/')}/models",
            headers={
                "Authorization": f"Bearer {token}",
                "originator": "codex_mindshub_cowork",
            },
            params={"client_version": client_version},
        ) as response:
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes():
                if time.monotonic() > deadline:
                    raise httpx.ReadTimeout("Model catalog fetch exceeded its deadline", request=response.request)
                body += chunk
    return json.loads(body)


@contextmanager
def model_catalog(
    endpoint: str,
    token: str,
    client_version: str,
    model: str,
) -> Iterator[Path | None]:
    """Load native metadata through the scoped proxy for one app-server lifetime.

    Yields None when the selected model has no usable row, so Codex starts on
    its bundled fallback metadata as it did before the catalog existed. MindsHub
    lists only moving aliases, and pinned versions must keep working.
    """
    try:
        payload = _fetch_models(endpoint, token, client_version)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Codex model catalog unavailable; using fallback metadata for %s: %s", model, exc)
        yield None
        return

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("slug"), str) for row in models
    ):
        logger.warning("MindsHub returned no native Codex model catalog; using fallback metadata for %s", model)
        yield None
        return
    # Hidden rows override bundled models MindsHub does not serve, so they
    # carry no metadata worth running on.
    if not any(row["slug"] == model and row.get("visibility") == "list" for row in models):
        logger.warning("Codex model catalog has no listed row for %s; using fallback metadata", model)
        yield None
        return
    for row in models:
        row.setdefault("supports_parallel_tool_calls", False)

    with TemporaryDirectory(prefix="cowork-codex-models-") as directory:
        path = Path(directory) / "models.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        yield path
