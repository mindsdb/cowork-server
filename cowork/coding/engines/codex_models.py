from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

# httpx timeouts apply per network operation, so a trickled body could hold
# startup open indefinitely. The deadline bounds the whole fetch; the shorter
# per-operation timeout bounds how far one read can overrun it.
_FETCH_DEADLINE_SECONDS = 15.0
_FETCH_OPERATION_TIMEOUT_SECONDS = 5.0


@contextmanager
def model_catalog(
    endpoint: str,
    token: str,
    client_version: str,
    model: str,
) -> Iterator[Path]:
    """Load native metadata through the scoped proxy for one app-server lifetime."""
    deadline = time.monotonic() + _FETCH_DEADLINE_SECONDS
    try:
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
        payload = json.loads(body)
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError("Unable to load Codex model metadata from MindsHub. Retry the task.") from exc

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list) or not models or any(
        not isinstance(row, dict) or not isinstance(row.get("slug"), str) for row in models
    ):
        raise RuntimeError("MindsHub did not return a native Codex model catalog. Check the inference server version.")
    # Hidden rows override bundled models MindsHub does not serve; Codex still
    # starts a thread on one, so the failure would surface on the first turn.
    if not any(row["slug"] == model and row.get("visibility") == "list" for row in models):
        raise RuntimeError("The selected model is missing from the Codex model catalog. Choose another model or retry.")
    for row in models:
        row.setdefault("supports_parallel_tool_calls", False)

    with TemporaryDirectory(prefix="cowork-codex-models-") as directory:
        path = Path(directory) / "models.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        yield path
