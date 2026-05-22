from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cowork.schemas.connectors import MatchCandidate, MatchResponse


class ConnectorSpecRegistry:
    def __init__(self, specs_dir: Path | None = None):
        self._specs_dir = specs_dir or Path(__file__).parent
        self._cache: dict[str, dict] | None = None

    def _load_all(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if not self._specs_dir.is_dir():
            return out
        for path in sorted(self._specs_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            cid = data.get("id") or path.stem
            data["id"] = cid
            out[cid] = data
        return out

    def all_connectors(self) -> dict[str, dict]:
        if self._cache is None:
            self._cache = self._load_all()
        return self._cache

    def get_connector(self, cid: str) -> dict | None:
        return self.all_connectors().get(cid)

    def list_summaries(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for c in self.all_connectors().values():
            out.append({
                "id": c.get("id"),
                "label": c.get("label", c.get("id")),
                "description": c.get("description", ""),
                "category": c.get("category", "other"),
                "logo": c.get("logo"),
                "logo_url": c.get("logo_url"),
                "logo_color": c.get("logo_color"),
                "aliases": c.get("aliases", []),
                "featured": c.get("featured", False),
            })
        out.sort(key=lambda x: (x.get("label") or "").lower())
        return out

    def reload(self) -> None:
        self._cache = None

    @staticmethod
    def _normalize(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

    def _exact_match(self, query: str) -> str | None:
        nq = self._normalize(query)
        if not nq:
            return None
        for c in self.all_connectors().values():
            if self._normalize(c.get("id", "")) == nq:
                return c["id"]
            for alias in c.get("aliases", []):
                if self._normalize(alias) == nq:
                    return c["id"]
        return None

    def _token_score(self, query: str, c: dict) -> float:
        q_tokens = set(self._normalize(query).split())
        if not q_tokens:
            return 0.0

        label_tokens = set(self._normalize(c.get("label", "")).split())
        alias_tokens: set[str] = set()
        for alias in c.get("aliases", []):
            alias_tokens.update(self._normalize(alias).split())
        keyword_tokens = set(self._normalize(" ".join(c.get("keywords", []))).split())
        desc_tokens = set(self._normalize(c.get("description", "")).split())

        score = 0.0
        score += 3.0 * len(q_tokens & label_tokens)
        score += 2.5 * len(q_tokens & alias_tokens)
        score += 1.0 * len(q_tokens & keyword_tokens)
        score += 0.4 * len(q_tokens & desc_tokens)
        return score

    def match_connector(self, query: str, max_candidates: int = 3) -> MatchResponse:
        exact_id = self._exact_match(query)
        if exact_id:
            return MatchResponse(
                candidates=[MatchCandidate(id=exact_id, confidence=1.0)],
                needs_clarification=False,
                stage="exact",
            )

        scored: list[tuple[float, str]] = []
        for c in self.all_connectors().values():
            s = self._token_score(query, c)
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

        n = min(max_candidates, len(scored))
        return MatchResponse(
            candidates=[
                MatchCandidate(id=cid, confidence=round(s / top_score, 3))
                for s, cid in scored[:n]
            ],
            needs_clarification=True,
            stage="scored-multi",
            question="Which one did you mean?",
        )


registry = ConnectorSpecRegistry()