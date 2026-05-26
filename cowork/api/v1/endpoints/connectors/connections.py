from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from cowork.schemas.connectors import ConnectionDetailResponse, ConnectionSummaryResponse
from cowork.services.connectors.connections import service

router = APIRouter()


@router.get("/", response_model=list[ConnectionSummaryResponse])
def list_connections():
    return service.list()


@router.get("/{engine}/{name}", response_model=ConnectionDetailResponse)
def get_connection(engine: str, name: str):
    record = service.get(engine, name)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")
    return record


@router.delete("/{engine}/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_connection(engine: str, name: str):
    if not service.delete(engine, name):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found.")
