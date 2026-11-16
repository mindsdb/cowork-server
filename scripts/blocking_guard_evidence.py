"""Versioned sidecars for composing the blocking guard across broad shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from scripts.test_inventory import Revision


class Leaf(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    path: str
    function: str
    kind: str
    leaf: str

    def key(self) -> tuple[str, str, str, str]:
        """Return the original guard's site identity for exact set composition."""
        return self.path, self.function, self.kind, self.leaf


class GuardEvidence(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: Literal[1]
    mode: Literal["collection", "execution"]
    cohort: Literal["broad", "broad-shard-0", "broad-shard-1", "broad-shard-2", "broad-shard-3"]
    revision: Revision
    revision_end: Revision
    run_id: str | None
    run_attempt: str | None
    required: list[Leaf]
    reached: list[Leaf]
    required_files: list[str]
    collected_files: list[str]
    workers: list[str]
    workers_lost: list[str]
    ineligible_because: str | None


def sidecar(inventory: Path) -> Path:
    """Keep different inventory basenames distinct even in one shared directory."""
    return inventory.with_name(inventory.name + ".blocking-guard.json")


def read_evidence(path: Path) -> GuardEvidence:
    """Reject malformed or contradictory evidence without echoing artifact contents."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if type(raw.get("schema_version")) is not int:
            raise ValueError("Invalid blocking guard version")
        evidence = GuardEvidence.model_validate(raw)
    except (ValidationError, json.JSONDecodeError, AttributeError) as exc:
        raise ValueError("Invalid blocking guard structure") from exc
    required = [leaf.key() for leaf in evidence.required]
    reached = [leaf.key() for leaf in evidence.reached]
    sequences = [required, reached, evidence.required_files, evidence.collected_files,
                 evidence.workers, evidence.workers_lost]
    if any(sequence != sorted(set(sequence)) for sequence in sequences):
        raise ValueError("Duplicate or unordered blocking guard evidence")
    if not required or not evidence.required_files or not set(reached) <= set(required):
        raise ValueError("Invalid blocking guard declaration or reaches")
    for leaf in evidence.required:
        if not all(leaf.key()) or (leaf.leaf != "*" and (*leaf.key()[:3], "*") in required):
            raise ValueError("Invalid blocking guard wildcard declaration")
    if evidence.mode == "collection" and (reached or evidence.workers or evidence.workers_lost):
        raise ValueError("Collection cannot attest blocking guard execution")
    if evidence.revision != evidence.revision_end:
        raise ValueError("Blocking guard source changed during execution")
    identity = (evidence.run_id, evidence.run_attempt)
    if identity != (None, None) and not all(
        value is not None and value.isascii() and value.isdecimal() and str(int(value)) == value and int(value) > 0
        for value in identity
    ):
        raise ValueError("Invalid blocking guard producer identity")
    return evidence
