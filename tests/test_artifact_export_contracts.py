"""The export contract: the wire models, the render-service fixtures and the helpers' signatures."""
import base64
import inspect
import typing
from pathlib import Path

import pytest
from pydantic import AnyHttpUrl, BaseModel, ValidationError
from pydantic_core import PydanticUndefined

from cowork.common.paths import PinnedDir
from cowork.schemas.artifact_export import (
    ExportFormat,
    ExportPdfSettings,
    ExportRequest,
    ExportResult,
    InlinedDocument,
    InlineLimits,
    InlineMode,
    OmittedReason,
    OmittedReference,
    PdfEngine,
    RenderPdfRefusal,
    RenderPdfRequest,
    RenderPdfResult,
)
from cowork.services.artifact_export import export_formats
from cowork.services.artifact_files import read_artifact_file
from cowork.services.artifact_inline import inline_artifact_html
from cowork.services.docx_writer import write_docx
from cowork.services.markdown_render import render_markdown

FIXTURES = Path(__file__).parent / "fixtures" / "render_pdf"


@pytest.mark.parametrize("fmt", ["html", "docx", "pdf"])
def test_export_request_takes_each_format(fmt):
    assert ExportRequest.model_validate({"format": fmt}).format == fmt


@pytest.mark.parametrize(
    "body",
    [
        {"format": "docx", "path": "/etc/passwd"},
        {"format": "rtf"},
        {"format": "DOCX"},
        {},
    ],
)
def test_export_request_refuses_any_other_body(body):
    with pytest.raises(ValidationError):
        ExportRequest.model_validate(body)


def test_export_result_answers_the_five_camel_case_fields():
    result = ExportResult(
        filename="report.docx",
        draft_url="/api/v1/artifacts/drafts/local/a1/report.docx",
        serve_url="",
        omitted_count=2,
        scripts_dropped=True,
    )
    assert result.model_dump(by_alias=True) == {
        "filename": "report.docx",
        "draftUrl": "/api/v1/artifacts/drafts/local/a1/report.docx",
        "serveUrl": "",
        "omittedCount": 2,
        "scriptsDropped": True,
    }


def test_omitted_reference_keeps_at_most_200_characters():
    omitted = OmittedReference(reference="x" * 500, reason=OmittedReason.SCHEME)
    assert omitted.reference == "x" * 200


def test_render_pdf_request_fixture_validates():
    request = RenderPdfRequest.model_validate_json((FIXTURES / "request.json").read_bytes())
    assert request.scripts is False


def test_render_pdf_request_refuses_an_extra_field():
    with pytest.raises(ValidationError):
        RenderPdfRequest.model_validate({"html": "<p>x</p>", "scripts": False, "url": "http://x"})


def test_render_pdf_response_fixture_validates_and_holds_a_pdf():
    result = RenderPdfResult.model_validate_json((FIXTURES / "response.json").read_bytes())
    assert base64.b64decode(result.pdf_base64, validate=True).startswith(b"%PDF-")


@pytest.mark.parametrize(
    ("helper", "names"),
    [
        (render_markdown, ["text"]),
        (read_artifact_file, ["folder", "parts", "max_bytes"]),
        (inline_artifact_html, ["html", "folder", "base_parts", "mode", "limits"]),
        (write_docx, ["document", "title"]),
        (export_formats, ["source_name", "is_backend", "can_edit", "pdf"]),
    ],
)
def test_export_helpers_take_keyword_arguments_only(helper, names):
    parameters = inspect.signature(helper).parameters.values()
    assert [p.name for p in parameters] == names
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters)


@pytest.mark.parametrize(
    ("helper", "hints"),
    [
        (render_markdown, {"text": str, "return": str}),
        (
            read_artifact_file,
            {"folder": PinnedDir, "parts": tuple[str, ...], "max_bytes": int, "return": bytes},
        ),
        (
            inline_artifact_html,
            {
                "html": bytes,
                "folder": PinnedDir,
                "base_parts": tuple[str, ...],
                "mode": InlineMode,
                "limits": InlineLimits,
                "return": InlinedDocument,
            },
        ),
        (write_docx, {"document": InlinedDocument, "title": str, "return": bytes}),
        (
            export_formats,
            {
                "source_name": str,
                "is_backend": bool,
                "can_edit": bool,
                "pdf": ExportPdfSettings,
                "return": list[ExportFormat],
            },
        ),
    ],
)
def test_export_helpers_keep_their_parameter_and_return_types(helper, hints):
    assert typing.get_type_hints(helper) == hints


def _fields(model: type[BaseModel]) -> dict[str, tuple[object, object]]:
    """Each field's type and default, in order; a required field's default is PydanticUndefined."""
    return {name: (field.annotation, field.default) for name, field in model.model_fields.items()}


@pytest.mark.parametrize(
    ("model", "fields"),
    [
        (
            InlinedDocument,
            {
                "html": (str, PydanticUndefined),
                "omitted": (list[OmittedReference], PydanticUndefined),
                "inlined_bytes": (int, PydanticUndefined),
                "scripts_dropped": (bool, PydanticUndefined),
            },
        ),
        (
            RenderPdfRefusal,
            {"url": (str, PydanticUndefined), "reason": (str, PydanticUndefined)},
        ),
        (
            RenderPdfResult,
            {
                "pdf_base64": (str, PydanticUndefined),
                "refused": (list[RenderPdfRefusal], PydanticUndefined),
            },
        ),
        (
            ExportPdfSettings,
            {
                "engine": (PdfEngine | None, PydanticUndefined),
                "render_url": (AnyHttpUrl | None, PydanticUndefined),
                "timeout_seconds": (float, 60.0),
                "max_pdf_bytes": (int, 48 * 1024 * 1024),
                "max_pages": (int, 500),
            },
        ),
    ],
    ids=["InlinedDocument", "RenderPdfRefusal", "RenderPdfResult", "ExportPdfSettings"],
)
def test_export_models_keep_their_fields_and_defaults(model, fields):
    assert _fields(model) == fields


def test_omitted_reasons_and_inline_modes_keep_their_values():
    assert [reason.value for reason in OmittedReason] == [
        "scheme",
        "absolute",
        "traversal",
        "housekeeping",
        "symlink",
        "reparse_point",
        "unc_path",
        "missing",
        "too_large",
        "budget",
        "unsupported_type",
    ]
    assert [mode.value for mode in InlineMode] == ["html", "word", "pdf"]
