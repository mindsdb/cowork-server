"""Regression: attachment uploads must not crash for long project names.

The attachment `purpose` tag is "attachment:{project}:{session}". The session
UUID is 36 chars, so the old `File.purpose` width of 64 left only ~16 chars
for the project name. A longer name (e.g. "Catana-Outbound-email") overflowed,
failed `File` model validation, and crashed the upload endpoint with a 500.
`File.purpose` is now 255; these tests pin that it accepts the long tag.
"""
from __future__ import annotations

from cowork.models.file import File
from cowork.services.files import attachment_purpose


def test_attachment_purpose_for_long_project_exceeds_old_limit():
    purpose = attachment_purpose(
        "Catana-Outbound-email", "d6ad2000-915b-4915-baf4-369e2db05f17"
    )
    # This is exactly the case that used to crash: > 64 chars.
    assert len(purpose) > 64


def test_file_model_accepts_long_purpose():
    purpose = attachment_purpose(
        "Catana-Outbound-email", "d6ad2000-915b-4915-baf4-369e2db05f17"
    )
    # Must not raise pydantic's "String should have at most 64 characters".
    f = File(
        filename="report.pdf",
        content_type="application/pdf",
        size=337438,
        purpose=purpose,
        path="/tmp/report.pdf",
    )
    assert f.purpose == purpose


def test_file_model_accepts_a_very_long_project_name():
    # Even a much longer project name should fit comfortably under 255.
    purpose = attachment_purpose("A" * 120, "d6ad2000-915b-4915-baf4-369e2db05f17")
    f = File(filename="x", content_type="text/plain", size=1, purpose=purpose, path="/tmp/x")
    assert f.purpose == purpose
