"""The export contract: the wire models, the render-service fixtures and the helpers' signatures."""
import base64
import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

from cowork.schemas.artifact_export import (
    ExportRequest,
    ExportResult,
    OmittedReason,
    OmittedReference,
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
