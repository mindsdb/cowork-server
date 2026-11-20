import hashlib
import json
from functools import partial
from uuid import UUID

from anton.core.artifacts.store import iter_content_files
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy.exc import IntegrityError

from cowork.api.v1.artifact_scope import ResolvedArtifact, review_artifact_and_read
from cowork.api.v1.endpoints.push import _require_org_mode, _caller
from cowork.api.v1.permissions import AuthenticatedInOrgMode, require
from cowork.db.scoped import scope_from_principal
from cowork.db.units import run_db
from cowork.principal import Principal
from cowork.services.content_reports import record_report, submit_report

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

def _artifact_evidence(found: ResolvedArtifact) -> tuple[str, str]:
    metadata = found.metadata.model_dump(mode="json")
    # No journal operation here: capturing a new text revision can mutate
    # another person's draft. File generations cover binary and auxiliary
    # assets too, and timestamps are taken without opening their contents.
    files = sorted(
        (str(path.relative_to(found.folder)), st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        for path, st in iter_content_files(found.folder)
    )
    generation = hashlib.sha256(json.dumps([metadata, files], sort_keys=True).encode()).hexdigest()
    return generation, json.dumps({"generation": generation, "metadata": metadata})


@router.post("/artifact", status_code=201)
async def report_artifact(body: ArtifactReportRequest, principal: Principal = Depends(_caller)):
    scope = scope_from_principal(principal)
    _, (generation, evidence) = await review_artifact_and_read(
        scope, str(body.project_id), body.artifact_id.hex,
        read=_artifact_evidence,
    )
    # All three identifiers are also available separately on the row or in its
    # evidence. Hash the composite to fit the queue's 128-character target key.
    identity = f"{body.project_id}:{body.artifact_id}:{generation}"
    key = hashlib.sha256(identity.encode()).hexdigest()
    operation = partial(record_report, target_key=f"artifact:{key}",
        project_id=body.project_id, artifact_id=body.artifact_id, evidence=evidence)
    try:
        return await run_db(operation, scope=scope)
    except IntegrityError:
        return await run_db(operation, scope=scope)
