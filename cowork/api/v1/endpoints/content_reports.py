from functools import partial
from uuid import UUID
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy.exc import IntegrityError
from cowork.api.v1.endpoints.push import _require_org_mode, _caller
from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import scope_from_principal
from cowork.db.units import run_db
from cowork.principal import Principal
from cowork.services.content_reports import submit_report

router = APIRouter(dependencies=[Depends(_require_org_mode), Depends(require(AuthenticatedInOrgMode))])

class ContentReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation_id: UUID
    message_id: UUID

@router.post("/", status_code=201)
async def report_content(body: ContentReportRequest, principal: Principal = Depends(_caller)):
    operation = partial(submit_report, conversation_id=body.conversation_id, message_id=body.message_id)
    scope = scope_from_principal(principal)
    try:
        return await run_db(operation, scope=scope)
    except IntegrityError:
        # A retry racing the first request returns the same receipt.
        return await run_db(operation, scope=scope)

class ArtifactReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: UUID
    artifact_id: UUID

@router.post("/artifact", status_code=201)
async def report_artifact(body: ArtifactReportRequest, principal: Principal = Depends(_caller)):
    from cowork.api.v1.artifact_scope import review_artifact_for_request
    from cowork.services.content_reports import record_report
    import json
    scope = scope_from_principal(principal)
    # The same visibility check used by the artifact viewer. A private draft
    # without a review grant remains indistinguishable from a missing one.
    _source, _folder, metadata, _is_own = await review_artifact_for_request(
        scope, str(body.project_id), body.artifact_id.hex,
    )
    # Retain the authorized reference and metadata, not arbitrary files/URLs
    # supplied by the reporter. Operators inspect the artifact in its store.
    payload = metadata.model_dump(mode="json") if hasattr(metadata, "model_dump") else metadata
    operation = partial(record_report, target_key=f"artifact:{body.project_id}:{body.artifact_id}",
        project_id=body.project_id, artifact_id=body.artifact_id,
        evidence=json.dumps(payload, default=str, ensure_ascii=False)[:65536])
    try:
        return await run_db(operation, scope=scope)
    except IntegrityError:
        return await run_db(operation, scope=scope)
