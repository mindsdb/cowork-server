"""`attachment_disposition` builds the only download header that carries a file name."""
import os
from urllib.parse import unquote

import pytest

from cowork.api.v1.content_disposition import attachment_disposition

RFC5987_PREFIX = "filename*=UTF-8''"


def _parts(header: str) -> tuple[str, str]:
    """Return (ASCII filename, decoded filename*) from a header."""
    assert header.startswith('attachment; filename="')
    ascii_part, _, encoded = header.partition(f'"; {RFC5987_PREFIX}')
    return ascii_part.removeprefix('attachment; filename="'), unquote(encoded)


@pytest.mark.parametrize(
    ("name", "ascii_name", "exact_name"),
    [
        ("report.pdf", "report.pdf", "report.pdf"),
        ("Отчёт.pdf", "download.pdf", "Отчёт.pdf"),
        ("報告.xlsx", "download.xlsx", "報告.xlsx"),
        ("plan – v2.docx", "plan  v2.docx", "plan – v2.docx"),
        ("café.pdf", "caf.pdf", "café.pdf"),
        # macOS screenshot names put U+202F before "PM".
        (
            "Screenshot 2026-10-27 at 1.37.00\u202fPM.png",
            "Screenshot 2026-10-27 at 1.37.00PM.png",
            "Screenshot 2026-10-27 at 1.37.00\u202fPM.png",
        ),
        ("a\u00a0b.txt", "ab.txt", "a\u00a0b.txt"),
        (".env", ".env", ".env"),
        (".gitignore", ".gitignore", ".gitignore"),
        ("x\r\ny.txt", "xy.txt", "xy.txt"),
        # A bidi override would make the save dialog show a fake extension.
        ("invoice\u202efdp.exe", "invoicefdp.exe", "invoicefdp.exe"),
        # Undecodable bytes in an on-disk name arrive as lone surrogates.
        (os.fsdecode(b"\xff.pdf"), "download.pdf", ".pdf"),
        # ZWJ joins emoji sequences and must survive.
        ("\U0001f468\u200d\U0001f4bb.png", "download.png", "\U0001f468\u200d\U0001f4bb.png"),
    ],
)
def test_ascii_fallback_and_exact_name(name, ascii_name, exact_name):
    header = attachment_disposition(name)

    assert "\r" not in header and "\n" not in header
    assert _parts(header) == (ascii_name, exact_name)


def test_latin1_ascii_name_keeps_its_previous_filename():
    assert attachment_disposition("report.pdf") == (
        "attachment; filename=\"report.pdf\"; filename*=UTF-8''report.pdf"
    )


def test_cyrillic_name_literal():
    assert attachment_disposition("Отчёт.pdf") == (
        "attachment; filename=\"download.pdf\"; "
        "filename*=UTF-8''%D0%9E%D1%82%D1%87%D1%91%D1%82.pdf"
    )


def test_header_is_latin1_encodable():
    # Starlette encodes header values as Latin-1; the original bug was a 500 here.
    for name in ("Отчёт.pdf", "報告.xlsx", "plan – v2.docx", "café.pdf"):
        attachment_disposition(name).encode("latin-1")
