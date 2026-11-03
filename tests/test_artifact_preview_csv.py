"""Row counting for the CSV artifact preview."""

import random

import pytest

from cowork.services.artifacts import count_csv_rows


def _reference_count(text: str) -> int:
    """Character-loop port of the renderer's countCsvRows (artifactPreviewUtils.js)."""
    if text.startswith("\ufeff"):
        text = text[1:]
    rows = 0
    in_quotes = False
    at_field_start = True
    i = 0
    end = len(text)
    while i < end:
        c = text[i]
        if in_quotes:
            if c == '"':
                if i + 1 < end and text[i + 1] == '"':
                    i += 1
                else:
                    in_quotes = False
        elif c == '"' and at_field_start:
            in_quotes = True
            at_field_start = False
        elif c == ",":
            at_field_start = True
        elif c in "\n\r":
            if c == "\r" and i + 1 < end and text[i + 1] == "\n":
                i += 1
            rows += 1
            at_field_start = True
        else:
            at_field_start = False
        i += 1
    if text and text[-1] not in "\n\r":
        rows += 1
    return rows


# Keep in sync with SHARED_CASES in the cowork renderer's
# src/renderer/cowork/components/artifact/artifactPreviewUtils.test.js.
SHARED_CASES = [
    ("", 0),
    ("id,name", 1),
    ("id,name\n", 1),
    ("id,name\n1,Ada\n2,Grace\n", 3),
    ("id,name\n1,Ada", 2),
    ("\ufeffid,name\n1,Ada\n", 2),
    ('\ufeff"a,b",c\n1,2\n', 2),
    ('id,note\n1,"two\nlines"\n', 2),
    ('id,note\n1,"say ""hi""\nthere"\n', 2),
    ('id,note\n1,part"ial\n2,fine\n', 3),
    ("id,name\r1,Ada\r2,Grace\r", 3),
    ("id,name\r\n1,Ada\r\n", 2),
    ('"abc\n', 0),
]


@pytest.mark.parametrize(("text", "expected"), SHARED_CASES)
def test_count_csv_rows_matches_the_renderer(text, expected):
    assert count_csv_rows(text) == expected


def test_count_csv_rows_agrees_with_the_character_loop():
    rng = random.Random(1234)
    for _ in range(5000):
        text = "".join(rng.choice('ab,"\n\r') for _ in range(rng.randint(0, 16)))
        if rng.random() < 0.1:
            text = "\ufeff" + text
        assert count_csv_rows(text) == _reference_count(text), repr(text)
