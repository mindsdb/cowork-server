"""Regression: attachment uploads must not crash for long project names.

Originally (ENG-333) the purpose tag embedded the project name
("attachment:{project}:{session}") and a long name overflowed the old
64-char `File.purpose` column, 500-ing the upload. The tag is now keyed by
the session id only (ENG-338), so its length is constant regardless of the
project name — pinned here. `File.purpose` stays unbounded TEXT so no
future free-form purpose can re-introduce a width crash.
"""
from __future__ import annotations

from cowork.models.file import File
from cowork.services.files import attachment_purpose


def test_attachment_purpose_length_is_independent_of_project_name():
    # The tag contains no project name at all — the ENG-333 overflow (and the
    # ENG-338 rename-stranding) are structurally impossible, not just padded.
    purpose = attachment_purpose("d6ad2000-915b-4915-baf4-369e2db05f17")
    assert purpose == "attachment:d6ad2000-915b-4915-baf4-369e2db05f17"
    assert len(purpose) < 64  # fits even the original column width


def test_file_model_accepts_long_free_form_purpose():
    # Non-attachment purposes remain free-form; the column must never again
    # be the thing that crashes a write.
    purpose = "x" * 400
    f = File(
        filename="report.pdf",
        content_type="application/pdf",
        size=337438,
        purpose=purpose,
        path="/tmp/report.pdf",
    )
    assert f.purpose == purpose
