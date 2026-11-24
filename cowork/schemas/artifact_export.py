"""Models for exporting a document artifact to HTML, Word or PDF.

The export route, the inliner, the Word writer and the render-service PDF
client all speak these models. ``ExportRequest``, ``ExportResult`` and
``ExportFormat`` are the wire the Cowork client reads; ``RenderPdfRequest``
and ``RenderPdfResult`` are the wire of a render service's ``/v1/pdf``, pinned
by ``tests/fixtures/render_pdf/``.
"""
from enum import StrEnum
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, field_validator

from cowork.schemas.base import CamelRequest, CamelResponse

ExportFormat = Literal["html", "docx", "pdf"]
PdfEngine = Literal["render"]

# An omitted reference keeps at most this many characters, so one hostile
# attribute can't grow the record without bound.
OMITTED_REFERENCE_MAX_CHARS = 200


class ExportRequest(CamelRequest):
    """Body of ``POST /api/v1/artifacts/workspace/{project_ref}/{artifact_id}/export``.

    The route exports the artifact's primary file only, so the body never
    names a file.
    """

    model_config = ConfigDict(extra="forbid")

    format: ExportFormat


class ExportResult(CamelResponse):
    """Answer to a successful export: the new file sits beside the source."""

    filename: str
    draft_url: str
    serve_url: str  # "" in org mode
    omitted_count: int
    scripts_dropped: bool


class InlineMode(StrEnum):
    HTML = "html"
    WORD = "word"
    PDF = "pdf"


class InlineLimits(BaseModel, frozen=True):
    """Caps the inliner enforces; going over one drops an asset, never the export.

    ``max_total_bytes`` counts what the output inlines, at source size: each
    use of an asset at its decoded size, and each spliced stylesheet at its
    size as read. It bounds neither the output nor the reading. The HTML
    written is larger, since base64 adds a third and escaping can make a
    stylesheet up to four times its size. An asset read once counts at each
    use, and a file is read before it is counted, so one past the budget is
    read and then left out; ``max_references`` and ``max_asset_bytes`` bound
    the reading.
    """

    max_asset_bytes: int = 2 * 1024 * 1024
    max_references: int = 256
    max_total_bytes: int  # caller: about 5 MiB for PDF; 25 MiB for Word, 10 MiB in org mode
    max_image_pixels: int = 40_000_000
    max_import_depth: int = 3
    max_nodes: int = 50_000


class OmittedReason(StrEnum):
    SCHEME = "scheme"
    ABSOLUTE = "absolute"
    TRAVERSAL = "traversal"
    HOUSEKEEPING = "housekeeping"
    SYMLINK = "symlink"
    REPARSE_POINT = "reparse_point"
    UNC_PATH = "unc_path"
    MISSING = "missing"
    TOO_LARGE = "too_large"
    BUDGET = "budget"
    UNSUPPORTED_TYPE = "unsupported_type"


class OmittedReference(BaseModel):
    """A reference the inliner left out of the document, and why."""

    reference: str
    reason: OmittedReason

    @field_validator("reference")
    @classmethod
    def _cap_reference(cls, value: str) -> str:
        return value[:OMITTED_REFERENCE_MAX_CHARS]


class InlinedDocument(BaseModel):
    """Self-contained HTML: every asset it keeps is a ``data:`` URI."""

    html: str
    omitted: list[OmittedReference]
    inlined_bytes: int  # what max_total_bytes counted: each inlined use at its source size
    scripts_dropped: bool


class RenderPdfRequest(BaseModel, extra="forbid", strict=True):
    html: str
    scripts: bool
    paper: Literal["letter", "a4"] = "letter"
    landscape: bool = False


class RenderPdfRefusal(BaseModel):
    url: str
    reason: str


class RenderPdfResult(BaseModel):
    pdf_base64: str
    refused: list[RenderPdfRefusal]


class ExportPdfSettings(BaseModel, frozen=True):
    """PDF export settings, read from the process environment only.

    ``COWORK_EXPORT_PDF_ENGINE`` and ``COWORK_EXPORT_RENDER_URL``; when the
    engine is unset, no deployment offers PDF.
    """

    engine: PdfEngine | None
    render_url: AnyHttpUrl | None
    timeout_seconds: float = 60.0
    max_pdf_bytes: int = 48 * 1024 * 1024
    max_pages: int = 500
