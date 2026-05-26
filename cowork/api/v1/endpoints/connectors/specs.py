from fastapi import APIRouter, HTTPException, status

from cowork.schemas.connectors import (
    ConnectorMetadataResponse,
    ConnectorSpecResponse,
    MatchRequest,
    MatchResponse,
)
from cowork.services.connectors.specs._registry import registry

router = APIRouter()


@router.get("/", response_model=list[ConnectorMetadataResponse])
def list_connector_specs():
    return registry.list_connectors()


@router.get("/{connector_id}", response_model=ConnectorSpecResponse)
def get_connector_spec(connector_id: str):
    spec = registry.get_connector(connector_id)
    if not spec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connector not found.")
    return spec


@router.post("/match", response_model=MatchResponse)
def match_connector_spec(req: MatchRequest) -> MatchResponse:
    return registry.match_connector(req.query, req.max_candidates)