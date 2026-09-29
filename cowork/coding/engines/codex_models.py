from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx


@contextmanager
def model_catalog(
    endpoint: str,
    token: str,
    client_version: str,
    model: str,
) -> Iterator[Path]:
    """Load native metadata through the scoped proxy for one app-server lifetime."""
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(
                f"{endpoint.rstrip('/')}/models",
                headers={
                    "Authorization": f"Bearer {token}",
                    "originator": "codex_mindshub_cowork",
                },
                params={"client_version": client_version},
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError("Unable to load Codex model metadata from MindsHub. Retry the task.") from exc

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list) or not models or any(
        not isinstance(row, dict) or not isinstance(row.get("slug"), str) for row in models
    ):
        raise RuntimeError("MindsHub did not return a native Codex model catalog. Check the inference server version.")
    if not any(row["slug"] == model for row in models):
        raise RuntimeError("The selected model is missing from the Codex model catalog. Choose another model or retry.")
    for row in models:
        row.setdefault("supports_parallel_tool_calls", False)

    with TemporaryDirectory(prefix="cowork-codex-models-") as directory:
        path = Path(directory) / "models.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        yield path
