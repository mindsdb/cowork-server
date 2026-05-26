from typing import Any

from pydantic import BaseModel, Field


class ConnectorField(BaseModel):
    name: str
    label: str
    type: str
    required: bool = False
    secret: bool = False
    placeholder: str | None = None
    description: str | None = None
    default: Any = None
    options: list[dict[str, Any]] | None = None


class OAuthConfig(BaseModel):
    auth_url: str
    token_url: str
    scopes: list[str] = []
    extra_auth_params: dict[str, str] = {}


class ConnectorMethod(BaseModel):
    id: str
    label: str
    description: str | None = None
    recommended: bool = False
    hidden: bool = False
    submit_action: str | None = None
    oauth: OAuthConfig | None = None
    how_to: str | None = None
    help_url: str | None = None
    fields: list[ConnectorField] = []


class ConnectorForm(BaseModel):
    form_id: str
    title: str
    subtitle: str | None = None
    logo: str | None = None
    logo_color: str | None = None
    methods: list[ConnectorMethod] | None = None
    fields: list[ConnectorField] | None = None


class ConnectorMetadataResponse(BaseModel):
    id: str
    label: str
    description: str
    category: str
    logo: str | None = None
    logo_url: str | None = None
    logo_color: str | None = None
    aliases: list[str] = []
    featured: bool = False


class ConnectorSpecResponse(ConnectorMetadataResponse):
    keywords: list[str] = []
    form: ConnectorForm


class MatchRequest(BaseModel):
    query: str
    max_candidates: int = Field(default=3, ge=1, le=5)


class MatchCandidate(BaseModel):
    id: str
    confidence: float


class MatchResponse(BaseModel):
    candidates: list[MatchCandidate]
    needs_clarification: bool
    stage: str
    question: str | None = None


class SubmitFormRequest(BaseModel):
    connector_id: str
    method: str | None = None
    name: str = ""
    conversation_id: str | None = None
    values: dict[str, Any] = Field(default_factory=dict)
    skipped: list[str] = Field(default_factory=list)


class SaveConnectionResponse(BaseModel):
    status: str
    submission_id: str
    engine: str
    name: str
    method: str | None