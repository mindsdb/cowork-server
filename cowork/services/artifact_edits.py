"""AI edit pipeline: propose a reviewable diff, then accept it with OCC.

This service implements the M1 "edit pipeline":

1. ``propose_edit`` — a dry-run that validates an ``old_text → new_text`` rewrite
   against the live artifact (without mutating anything) and returns a structured
   diff the client can render for review.
2. ``accept_edit`` — applies the rewrite with optimistic-concurrency control
   (a base-version compare-and-swap). If the artifact has moved since the client
   loaded ``base_version_id`` the accept raises :class:`EditConflict`, which the
   API maps to HTTP 409. Otherwise the rewrite is applied through the existing
   snapshot/patch machinery and a new ``ai_edit`` version is recorded.

It REUSES the existing primitives in ``cowork.services.artifact_versions`` rather
than re-implementing snapshotting or patching:

- ``_artifact_from_identifier_or_path`` — resolve an artifact by path.
- ``_normalize_proposed_patch`` / ``_apply_patch_operations`` — build + apply a
  ``replace_text`` patch op the same way accepted suggestions do.
- ``ArtifactVersionService.snapshot_artifact`` — content-addressed snapshot that
  advances ``artifact.current_version_id`` and records activity.
- ``_external_artifact_id`` / ``version_to_dict`` — response shaping.

The OCC guard mirrors the existing "suggestion was created for an older artifact
version" stale-guard used by ``apply_comment_patch``/``preview_comment_patch`` —
the difference is that here the base version is supplied by the client per-request
rather than stored on a comment row.

TODO(model): generating ``new_text`` from a natural-language instruction is out of
scope for M1 — the client supplies ``new_text`` directly. Wire an LLM rewrite here
(read ``old_text`` from ``target``, call the model, return the draft) so the
frontend's ``proposeEdit({ target, instruction })`` no longer has to send the
rewrite itself.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from sqlmodel import Session

from cowork.models.artifact import Artifact, ArtifactVersion
from cowork.services.artifact_versions import (
    ArtifactVersionService,
    _apply_patch_operations,
    _artifact_from_identifier_or_path,
    _external_artifact_id,
    _normalize_proposed_patch,
    version_to_dict,
)


class EditConflict(Exception):
    """Raised when ``accept_edit`` loses the base-version compare-and-swap.

    Carries the artifact's *current* version so the API layer can return it in
    the 409 body (the frontend captures ``currentVersionId`` to drive its
    "Merge & keep" retry).
    """

    def __init__(
        self,
        *,
        artifact: Artifact,
        base_version_id: str | None,
        current_version_id: str | None,
        current_version: ArtifactVersion | None = None,
        session: Session | None = None,
        message: str | None = None,
    ) -> None:
        self.artifact = artifact
        self.base_version_id = base_version_id
        self.current_version_id = current_version_id
        self.current_version = current_version
        self.session = session
        # Default copy matches the frontend's expected conflict message
        # (editIntegrationNotes.md), overridable by callers.
        self.message = message or "This changed since you started — Anton can merge your edit"
        super().__init__(self.message)

    def current_version_dict(self) -> dict | None:
        """Serialize the current version for the 409 body, when available."""
        if self.current_version is None:
            return None
        return version_to_dict(
            self.current_version,
            session=self.session,
            artifact_external_id=_external_artifact_id(self.artifact, self.session),
        )


def _replace_text_operations(*, target: str | None, old_text: str, new_text: str) -> dict:
    """Build a normalized single ``replace_text`` patch for ``target``.

    ``target`` is the file path within the artifact folder to edit. The patch is
    validated/normalized by the existing ``_normalize_proposed_patch`` so it stays
    compatible with ``_apply_patch_operations`` (which is what enforces that
    ``old_text`` is actually present in the file).
    """
    rel_path = (target or "").strip()
    if not rel_path:
        raise ValueError("target (the file path to edit) is required")
    if not isinstance(old_text, str) or not old_text:
        raise ValueError("old_text must be a non-empty string")
    if not isinstance(new_text, str):
        raise ValueError("new_text must be a string")
    return _normalize_proposed_patch(
        {
            "operations": [
                {
                    "type": "replace_text",
                    "path": rel_path,
                    "find": old_text,
                    "replace": new_text,
                }
            ]
        }
    )


def _version_id_str(version_id: object) -> str | None:
    if version_id is None:
        return None
    return str(version_id)


def propose_edit(
    session: Session,
    *,
    path: str,
    target: str,
    old_text: str,
    new_text: str,
    base_version_id: str | None = None,
) -> dict:
    """Dry-run an AI edit and return a structured diff WITHOUT mutating anything.

    Resolves the artifact by ``path``, then validates that ``old_text`` is present
    in ``target`` by applying the rewrite against a throwaway copy of the live
    folder (the same validation ``preview_comment_patch`` performs). Nothing on
    disk or in the database changes.

    Returns a diff dict::

        {
            "target": "<file path>",
            "old": "<old_text>",
            "new": "<new_text>",
            "baseVersionId": "<the version the client is looking at, echoed back>",
            "currentVersionId": "<artifact.current_version_id>",
            "applies": True,           # whether old_text was found and the patch is clean
            "oldText": "...",          # aliases for the frontend proposeEdit() contract
            "newText": "...",
            "artifactId": "...",
            "artifactPath": "...",
        }

    ``applies`` is ``False`` (rather than raising) when ``old_text`` cannot be
    located, so the client can show a "could not draft a change" affordance.

    TODO(model): when the model is wired, ``new_text`` will be generated here from
    an ``instruction`` instead of being supplied by the caller.
    """
    artifact = _artifact_from_identifier_or_path(session, artifact_id=None, path=path)
    patch = _replace_text_operations(target=target, old_text=old_text, new_text=new_text)

    applies = True
    apply_error: str | None = None
    live_root = Path(artifact.path).expanduser().resolve(strict=False)
    with TemporaryDirectory(prefix="cowork-edit-propose-") as tmp:
        preview_root = Path(tmp) / live_root.name
        shutil.copytree(live_root, preview_root)
        try:
            _apply_patch_operations(preview_root, patch)
        except (ValueError, FileNotFoundError) as exc:
            applies = False
            apply_error = str(exc)

    artifact_external_id = _external_artifact_id(artifact, session)
    result = {
        "target": target,
        "old": old_text,
        "new": new_text,
        "oldText": old_text,
        "newText": new_text,
        "baseVersionId": _version_id_str(base_version_id),
        "currentVersionId": _version_id_str(artifact.current_version_id),
        "applies": applies,
        "artifactId": artifact_external_id,
        "artifactPath": artifact.path,
    }
    if apply_error is not None:
        result["error"] = apply_error
    return result


def accept_edit(
    session: Session,
    *,
    path: str,
    target: str,
    old_text: str,
    new_text: str,
    base_version_id: str | None,
    actor_name: str | None = None,
    actor_email: str | None = None,
    actor_subject: str | None = None,
) -> dict:
    """Apply an AI edit with optimistic-concurrency control.

    OCC contract: the edit is applied only if ``base_version_id`` still equals the
    artifact's ``current_version_id``. If they differ — the artifact moved since the
    client loaded it — :class:`EditConflict` is raised carrying the current version
    (the API maps it to HTTP 409). This is the compare-and-swap the frontend's
    ``commitEdit`` relies on.

    On success the rewrite is applied via the existing ``_apply_patch_operations``
    machinery against a staged copy of the live folder, the folder is atomically
    swapped, and a new ``ai_edit`` snapshot is recorded (advancing
    ``current_version_id``). Returns::

        {
            "ok": True,
            "versionId": "<new version id>",
            "version": { ...version_to_dict... },
            "changedPaths": ["<target>"],
            "artifactId": "...",
            "artifactPath": "...",
            "previousVersionId": "<base version that was swapped from>",
        }

    TODO(merge): same-``target`` collisions correctly 409 today. A disjoint-edit
    3-way auto-merge (base/ours/theirs) belongs here so non-overlapping concurrent
    edits to the *same* file can land without a conflict, and so "Merge & keep" can
    resolve prose overlaps server-side.
    """
    artifact = _artifact_from_identifier_or_path(session, artifact_id=None, path=path)

    current_version_id = artifact.current_version_id
    base_uuid = _coerce_version_uuid(base_version_id)
    current_str = _version_id_str(current_version_id)

    # Compare-and-swap on the base version. When the artifact has no versions yet
    # (current_version_id is None) we only proceed if the caller also passed no
    # base — anything else is a stale/mismatched base and must conflict.
    if base_uuid != current_version_id:
        current_version = (
            session.get(ArtifactVersion, current_version_id) if current_version_id is not None else None
        )
        raise EditConflict(
            artifact=artifact,
            base_version_id=_version_id_str(base_version_id),
            current_version_id=current_str,
            current_version=current_version,
            session=session,
        )

    patch = _replace_text_operations(target=target, old_text=old_text, new_text=new_text)

    service = ArtifactVersionService(session)
    root = Path(artifact.path).expanduser().resolve(strict=False)

    backup: Path | None = None
    with TemporaryDirectory(prefix="cowork-edit-apply-", dir=root.parent) as tmp:
        staged_root = Path(tmp) / root.name
        shutil.copytree(root, staged_root)
        # Raises ValueError/FileNotFoundError if old_text is no longer present;
        # nothing has been swapped yet, so the live folder is untouched.
        changed_paths = _apply_patch_operations(staged_root, patch)
        backup = service._replace_artifact_folder(root, staged_root, keep_backup=True)

    try:
        applied = service.snapshot_artifact(
            root,
            artifact_id=artifact.id,
            label="AI edit",
            operation_type="ai_edit",
            prompt=f"AI edit: {target}",
            preview_status="ready",
        )
    except Exception:
        # Roll the live folder back to its pre-edit state on any failure.
        session.rollback()
        if backup is not None and backup.exists():
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
            os.replace(backup, root)
        raise
    finally:
        if backup is not None and backup.exists():
            shutil.rmtree(backup, ignore_errors=True)

    artifact_external_id = _external_artifact_id(session.get(Artifact, applied.artifact_id), session)
    return {
        "ok": True,
        "versionId": str(applied.id),
        "version": version_to_dict(applied, session=session, artifact_external_id=artifact_external_id),
        "changedPaths": changed_paths,
        "artifactId": artifact_external_id,
        "artifactPath": artifact.path,
        "previousVersionId": current_str,
    }


def _coerce_version_uuid(version_id: str | UUID | None) -> UUID | None:
    """Parse a client-supplied base version id into a UUID (or None).

    A malformed, non-empty value can never equal a real ``current_version_id``, so
    it is surfaced as ``ValueError`` by the API (400) rather than silently treated
    as "no base", which would let a garbage base slip past the CAS.
    """
    if version_id is None:
        return None
    if isinstance(version_id, UUID):
        return version_id
    text = str(version_id).strip()
    if not text:
        return None
    try:
        return UUID(text)
    except ValueError as exc:
        raise ValueError(f"Invalid base_version_id: {version_id!r}") from exc
