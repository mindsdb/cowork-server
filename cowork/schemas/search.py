from pydantic import BaseModel


class SearchResultResponse(BaseModel):
    type: str
    id: str
    title: str
    subtitle: str
    route: str
    score: int


class SearchResponse(BaseModel):
    results: list[SearchResultResponse]
