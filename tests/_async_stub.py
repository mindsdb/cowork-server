"""Coroutine stand-ins built from plain functions."""
from __future__ import annotations

from cowork.api.v1.artifact_scope import ResolvedArtifact, ReviewedRead
from cowork.db.units import run_efs


def awaitable(fn):
    """``fn`` as a coroutine function, called when awaited, for a stub of an
    async function such as the artifact resolver or the owner gate."""

    async def stub(*args, **kwargs):
        return fn(*args, **kwargs)

    return stub


def reviewed_and_read(review):
    """``review``, a stand-in for review_artifact_for_request, as one for
    review_artifact_and_read: the route's read runs on what ``review``
    returned, in a file store thread as the real lookup runs it, and never
    when ``review`` raised."""

    async def review_and_read(scope, project_ref, artifact_id, *, read):
        artifact = ResolvedArtifact(*review(scope, project_ref, artifact_id))
        return await run_efs(lambda: ReviewedRead(artifact, read(artifact)), scope=scope)

    return review_and_read
