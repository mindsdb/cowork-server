import re

from fastapi import APIRouter, HTTPException, status

from cowork.schemas.connectors import (
    ConnectorMetadataResponse,
    ConnectorSpecResponse,
    MatchRequest,
    MatchResponse,
    MatchCandidate,
)
from cowork.services.connectors import registry

router = APIRouter()


@router.get("/", response_model=list[ConnectorMetadataResponse])
def list_connector_specs():
    return registry.list_summaries()


@router.get("/{connector_id}", response_model=ConnectorSpecResponse)
def get_connector_spec(connector_id: str):
    spec = registry.get_connector(connector_id)
    if not spec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connector not found.")
    return spec


# ─── Matching helpers ──────────────────────────────────────────────


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _exact_match(query: str) -> str | None:
    nq = _normalize(query)
    if not nq:
        return None
    for c in registry.all_connectors().values():
        if _normalize(c.get("id", "")) == nq:
            return c["id"]
        for alias in c.get("aliases", []):
            if _normalize(alias) == nq:
                return c["id"]
    return None


def _token_score(query: str, c: dict) -> float:
    q_tokens = set(_normalize(query).split())
    if not q_tokens:
        return 0.0

    label_tokens = set(_normalize(c.get("label", "")).split())
    alias_tokens: set[str] = set()
    for alias in c.get("aliases", []):
        alias_tokens.update(_normalize(alias).split())
    keyword_tokens = set(_normalize(" ".join(c.get("keywords", []))).split())
    desc_tokens = set(_normalize(c.get("description", "")).split())

    score = 0.0
    score += 3.0 * len(q_tokens & label_tokens)
    score += 2.5 * len(q_tokens & alias_tokens)
    score += 1.0 * len(q_tokens & keyword_tokens)
    score += 0.4 * len(q_tokens & desc_tokens)
    return score


@router.post("/match", response_model=MatchResponse)
def match_connector_spec(req: MatchRequest) -> MatchResponse:
    exact_id = _exact_match(req.query)
    if exact_id:
        return MatchResponse(
            candidates=[MatchCandidate(id=exact_id, confidence=1.0)],
            needs_clarification=False,
            stage="exact",
        )

    scored: list[tuple[float, str]] = []
    for c in registry.all_connectors().values():
        s = _token_score(req.query, c)
        if s > 0:
            scored.append((s, c["id"]))
    scored.sort(reverse=True)

    if not scored:
        return MatchResponse(
            candidates=[],
            needs_clarification=True,
            stage="no-match",
            question="I don't recognise that one — try the search box?",
        )

    top_score = scored[0][0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0

    if runner_up == 0.0 or top_score >= runner_up * 2:
        return MatchResponse(
            candidates=[MatchCandidate(id=scored[0][1], confidence=0.85)],
            needs_clarification=False,
            stage="scored-single",
        )

    n = min(req.max_candidates, len(scored))
    return MatchResponse(
        candidates=[
            MatchCandidate(id=cid, confidence=round(s / top_score, 3))
            for s, cid in scored[:n]
        ],
        needs_clarification=True,
        stage="scored-multi",
        question="Which one did you mean?",
    )