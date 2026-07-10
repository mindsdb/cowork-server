from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
    connector_id: str | None = None
    method: str | None = None
    name: str = ""
    conversation_id: str | None = None
    values: dict[str, Any] = Field(default_factory=dict)
    skipped: list[str] = Field(default_factory=list)
    # Compat: the current client sends form_id + form_spec instead of
    # connector_id + method. Derive from these if connector_id is absent.
    form_id: str | None = None
    form_spec: dict[str, Any] | None = None

    def resolve_connector_id(self) -> str:
        if self.connector_id:
            return self.connector_id
        if self.form_spec and self.form_spec.get("_connector_id"):
            return self.form_spec["_connector_id"]
        if self.form_id:
            return self.form_id.removesuffix("-connector")
        raise ValueError("connector_id is required")

    def resolve_method(self) -> str | None:
        if self.method:
            return self.method
        if self.form_spec:
            return self.form_spec.get("selected_method") or self.form_spec.get("auth_method")
        return None


class SaveConnectionResponse(BaseModel):
    status: str
    submission_id: str
    engine: str
    name: str
    method: str | None


class ConnectionSummaryResponse(BaseModel):
    engine: str
    name: str
    # Human-facing name for the card (label or derived identity); falls back to
    # `name` (the slug) client-side when null.
    display_name: str | None = None
    created_at: str | None = None
    label: str | None = None
    logo: str | None = None
    logo_color: str | None = None
    # "needs_reconnect" when the connection's token was lost/revoked; absent
    # when healthy. Lets the catalogue card show a warning without requiring
    # the client to fetch each connection's full detail first.
    status: str | None = None


class ConnectionDetailResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    engine: str
    name: str
    display_name: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    connector_id: str | None = None
    method: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    # Names of fields in `fields` that were masked with VAULT_KEEP_SENTINEL —
    # lets the client show a "saved" indicator instead of the raw sentinel.
    secure_keys: list[str] = Field(default_factory=list, serialization_alias="secureKeys")


class DirectSaveRequest(BaseModel):
    connector_id: str
    method: str | None = None
    name: str = ""
    values: dict[str, Any] = Field(default_factory=dict)


class OAuthStartRequest(BaseModel):
    client_id: str = ""
    client_secret: str = ""
    extra_fields: dict[str, str] = Field(default_factory=dict)


class OAuthStartResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    auth_url: str = Field(serialization_alias="authUrl")
    redirect_uri: str = Field(serialization_alias="redirectUri")
    started_at: str = Field(serialization_alias="startedAt")
    state: str


class DisabledConnection(BaseModel):
    engine: str
    name: str
