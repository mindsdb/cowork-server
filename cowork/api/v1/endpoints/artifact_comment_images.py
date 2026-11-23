"""Images in artifact comments (ENG-4410).

``POST /api/v1/artifacts/workspace/{project_ref}/{artifact_id}/comment-images``
stores one image (multipart ``file``) and answers ``{"id", "mediaType",
"size"}``; the app puts the id in the comment's text
(services/comment_images.py). ``GET .../comment-images/{image_id}`` answers
the image.

Both are for whoever may review the artifact, the owner included: the people
who see its comments. Another person's artifact answers 404, as its comments
do. An image is checked by its bytes (PNG, JPEG, GIF or WebP, up to 5 MB):
anything else answers 415, a larger one 413. It is answered with the type its
bytes say, ``nosniff`` and a sandboxing CSP, so it can only ever be an image.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status

from cowork.api.v1.artifact_scope import review_artifact_and_read
from cowork.api.v1.endpoints.artifact_workspace import ArtifactIdDep, ScopeDep
from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.services.comment_images import (
    IMAGE_RESPONSE_HEADERS,
    MAX_IMAGE_BYTES,
    ImageRefused,
    read_comment_image,
    save_comment_image,
)

router = APIRouter(dependencies=[Depends(require(AuthenticatedInOrgMode))])

_REFUSAL_STATUS = {
    "too_large": status.HTTP_413_CONTENT_TOO_LARGE,
    "unsupported": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
}


@router.post("/workspace/{project_ref}/{artifact_id}/comment-images")
async def upload_comment_image(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    scope: ScopeDep,
    file: UploadFile = File(...),
):
    # One byte past the cap is enough to refuse it; the rest is never read.
    data = await file.read(MAX_IMAGE_BYTES + 1)
    try:
        reviewed = await review_artifact_and_read(
            scope, project_ref, artifact_id, read=lambda found: save_comment_image(found.folder, data),
        )
    except ImageRefused as exc:
        raise HTTPException(status_code=_REFUSAL_STATUS[exc.reason], detail=str(exc)) from exc
    return reviewed.value


@router.get("/workspace/{project_ref}/{artifact_id}/comment-images/{image_id}")
async def comment_image(
    project_ref: str,
    artifact_id: ArtifactIdDep,
    image_id: str,
    scope: ScopeDep,
):
    def read(found):
        try:
            return read_comment_image(found.folder, image_id)
        except FileNotFoundError:
            return None

    reviewed = await review_artifact_and_read(scope, project_ref, artifact_id, read=read)
    if reviewed.value is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    data, media_type = reviewed.value
    return Response(content=data, media_type=media_type, headers=dict(IMAGE_RESPONSE_HEADERS))

