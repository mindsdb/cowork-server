"""The doc format: the model's validation, the Markdown dialect both ways, and the page it renders as."""
from __future__ import annotations

import json
import re

import pytest

from cowork.docs.markdown import blocks_to_markdown, parse_markdown
from cowork.docs.model import (
    MAX_CHART_SERIES,
    MAX_METRICS,
    SCHEMA,
    DocError,
    image_src_ok,
    is_number,
    normalize_block,
    normalize_doc,
    to_iso_date,
    to_number,
)
from cowork.docs.render import doc_html, inline_html

from tests._doc_samples import PLAN


def _format_doc(markdown: str = PLAN, title: str = "Q3 launch plan"):
    """A stored doc holding ``markdown``'s blocks, read as the format reads one (no edit ops)."""
    return normalize_doc({"schema": SCHEMA, "title": title, "rev": 1,
                          "tabs": [{"id": "t_1", "title": title, "blocks": parse_markdown(markdown, 1)}]})


def test_markdown_round_trips():
    blocks = parse_markdown(PLAN)
    again = parse_markdown(blocks_to_markdown(blocks))
    strip = lambda bs: [{k: v for k, v in b.items() if k not in ("id", "v")} for b in bs]  # noqa: E731
    # Table/row/column ids are minted fresh on every parse; compare shapes.
    for a, b in zip(strip(blocks), strip(again), strict=True):
        if a["type"] == "table":
            assert [c["name"] for c in a["columns"]] == [c["name"] for c in b["columns"]]
            assert [list(r["cells"].values()) for r in a["rows"]] == [list(r["cells"].values()) for r in b["rows"]]
        elif a["type"] == "chart":
            assert a["chart"] == b["chart"]
        else:
            assert a == b

@pytest.mark.parametrize("line,text", [
    ("## Learning C#", "Learning C#"),
    ("## F# and C# ##", "F# and C#"),
    ("### Issue #12", "Issue #12"),
    ("## Closed ##   ", "Closed"),
])
def test_a_heading_keeps_a_trailing_hash_that_is_part_of_a_word(line, text):
    assert [(b["type"], b["text"]) for b in parse_markdown(line)] == [("heading", text)]


@pytest.mark.parametrize("text", ["Learning C#", "Ticket #", "Ends in hashes ##"])
def test_a_heading_reads_back_as_it_was_written(text):
    (heading,) = parse_markdown("## x")
    heading["text"] = text
    assert [b["text"] for b in parse_markdown(blocks_to_markdown([heading]))] == [text]


@pytest.mark.parametrize("code", [
    "```\nnested\n```",
    "before\n````js\nlonger\n````\nafter",
    "inline `code` and ``more``",
    "~~~\ntilde\n~~~",
])
def test_a_code_block_holding_a_fence_reads_back_as_one_block(code):
    (block,) = parse_markdown("```text\nx\n```")
    block["code"] = code
    again = parse_markdown(blocks_to_markdown([block]))
    assert [(b["type"], b["language"], b["code"]) for b in again] == [("code", "text", code)]


def test_a_fence_is_closed_only_by_one_as_long_alone_on_its_line():
    md = "````\n```\n```python\n````\n\nAfter."
    assert [(b["type"], b.get("code") or b.get("text")) for b in parse_markdown(md)] == [
        ("code", "```\n```python"), ("paragraph", "After.")]
    assert [b["type"] for b in parse_markdown("```\nx\n`````\n\nAfter.")] == ["code", "paragraph"]


def test_a_metrics_block_past_its_limit_is_refused_naming_it_not_cut_short():
    tiles = [{"label": f"M{n}", "value": str(n)} for n in range(MAX_METRICS + 1)]
    assert len(normalize_block({"type": "metrics", "items": tiles[:MAX_METRICS]})["items"]) == MAX_METRICS
    with pytest.raises(DocError) as err:
        normalize_block({"type": "metrics", "items": tiles})
    assert err.value.code == "bad_block" and f"at most {MAX_METRICS} tiles" in err.value.message


def test_a_chart_past_its_series_limit_is_refused_naming_it_not_cut_short():
    series = [{"name": f"S{n}", "values": [n]} for n in range(MAX_CHART_SERIES + 1)]
    chart = {"type": "chart", "chart": "line", "x": ["Jan"]}
    assert len(normalize_block({**chart, "series": series[:MAX_CHART_SERIES]})["series"]) == MAX_CHART_SERIES
    with pytest.raises(DocError) as err:
        normalize_block({**chart, "series": series})
    assert err.value.code == "bad_chart" and f"at most {MAX_CHART_SERIES} series" in err.value.message


# The app's docModel.toNumber reads by the same rule, against the same vectors.
@pytest.mark.parametrize("text,value", [
    ("1,200", 1200),
    ("$1.5M", 1_500_000),
    ("-3.2%", -3.2),
    ("12k", 12_000),
    ("\u22124", -4),
    ("2B", 2_000_000_000),
    ("500 MB", None),
    ("1e6", None),
    ("12-15", None),
    ("5pm", None),
    ("1,20", None),
    ("", None),
    ("  42 ", 42),
    ("**1,200**", 1200),
    ("_$1.5M_", 1_500_000),
    ("**12-15**", None),
    ("*5*%", None),
    ("** 42 **", 42),
    ("$1.1M", 1_100_000),
    ("0.1234567", 0.1234567),
])
def test_a_cell_reads_as_a_number_only_when_it_is_written_as_one(text, value):
    assert to_number(text) == value
    assert is_number(text) is (value is not None)


def test_a_column_is_numbers_only_when_every_value_reads_as_one():
    (table,) = parse_markdown("| Size | Raise | Range |\n| --- | --- | --- |\n"
                              "| 500 MB | $1.5M | 12-15 |\n| 2 GB | 12k | 20-25 |")
    assert [c["kind"] for c in table["columns"]] == ["text", "number", "text"]


def test_a_charts_values_are_read_by_the_same_rule():
    chart = normalize_block({"type": "chart", "chart": "bar", "x": ["a", "b", "c"],
                             "series": [{"name": "s", "values": ["1,200", "500 MB", 7]}]})
    assert chart["series"][0]["values"] == [1200, None, 7]


def test_bad_chart_json_is_explained():
    with pytest.raises(DocError) as err:
        parse_markdown("```chart\n{not json}\n```")
    assert err.value.code == "bad_markdown"
    with pytest.raises(DocError) as err:
        parse_markdown('```chart\n{"type":"bar","x":["a","b"],"series":[{"name":"s","values":[1]}]}\n```')
    assert "1 values for 2 x labels" in err.value.message

def test_a_stored_doc_is_validated_on_load():
    doc = _format_doc()
    assert normalize_doc(doc)["tabs"][0]["blocks"][0]["id"] == doc["tabs"][0]["blocks"][0]["id"]
    with pytest.raises(DocError):
        normalize_doc({"schema": "something-else"})

def test_charts_read_back_with_the_key_the_agent_writes():
    md = blocks_to_markdown([b for b in parse_markdown(PLAN) if b["type"] == "chart"])
    assert '"type": "bar"' in md and '"chart"' not in md

@pytest.mark.parametrize("src", [
    "https://example.com/a.png", "http://example.com/a.png", "data:image/png;base64,iVBORw0KGgo=",
    "chart.png", "media/chart.png",
])
def test_an_image_may_point_at_the_web_or_beside_the_doc(src):
    assert image_src_ok(src)

@pytest.mark.parametrize("src", [
    "/Users/me/Pictures/passport.png", "~/Desktop/scan.png", "../../secret.png", "media/../../x.png",
    "file:///etc/hosts", "javascript:alert(1)", "C:\\Users\\me\\scan.png", "\\\\server\\share\\x.png",
    "data:text/html;base64,PHNjcmlwdD4=", "",
])
def test_an_image_may_not_point_elsewhere_on_the_machine(src):
    assert not image_src_ok(src)

def test_an_unusable_image_stays_as_its_text_on_import():
    blocks = parse_markdown("![scan](/Users/me/Pictures/passport.png)\n\n![logo](https://example.com/logo.png)")
    assert [b["type"] for b in blocks] == ["paragraph", "image"]
    assert "passport.png" in blocks[0]["text"]

@pytest.mark.parametrize("fence", ["5", '"tiles"', "true"])
def test_a_metrics_fence_that_isnt_tiles_is_refused(fence):
    with pytest.raises(DocError) as err:
        parse_markdown(f"```metrics\n{fence}\n```")
    assert err.value.code == "bad_markdown"

@pytest.mark.parametrize("raw", [
    {"schema": "cowork.doc/1", "rev": "seven", "tabs": []},
    {"schema": "cowork.doc/1", "rev": 1, "tabs": ["not a tab"]},
])
def test_a_malformed_doc_on_disk_is_refused_as_bad(raw):
    with pytest.raises(DocError) as err:
        normalize_doc(raw)
    assert err.value.code == "bad_doc"

def test_a_pasted_table_with_many_distinct_values_parses_quickly():
    import time

    rows = "".join(f"| row {i} | {i} |\n" for i in range(2000))
    start = time.perf_counter()
    blocks = parse_markdown("| Name | N |\n| --- | --- |\n" + rows)
    assert time.perf_counter() - start < 1.0
    assert [c["kind"] for c in blocks[0]["columns"]] == ["text", "number"]

def test_inline_markup_leaves_link_addresses_whole():
    out = inline_html("See [the **plan**](https://x.com/_p_/?a=1&b=2) and _this_.")
    assert '<a href="https://x.com/_p_/?a=1&amp;b=2">the <strong>plan</strong></a>' in out
    assert "<em>this</em>" in out

def test_text_never_becomes_markup_on_the_page():
    hostile = "<script>alert(1)</script> [x](javascript:alert(1)) \x005\x00"
    doc = _format_doc(f"{hostile}\n\n| Col <i> |\n| --- |\n| {hostile} |", title="<b>Title</b>")
    page = doc_html(doc)
    assert "<script>alert" not in page and "<i>" not in page and "<b>Title" not in page
    assert "&lt;script&gt;" in page and 'href="javascript' not in page


def _chart_page(values: list[float], labels: list[str]) -> str:
    chart = ('```chart\n{"type": "line", "title": "Weekly value", "x": %s, '
             '"series": [{"name": "Value", "values": %s}]}\n```' % (
                 json.dumps(labels), json.dumps(values)))
    return doc_html(_format_doc(chart, "Chart"))


def test_a_pages_chart_reads_its_axis_on_round_ticks_none_cut_off():
    # Quarters of the range ("468,202.50") ran past the fixed margin and lost their first digits.
    svg = _chart_page([553750, 618550, 624270], ["w1", "w2", "w3"])
    ticks = re.findall(r'text-anchor="end" fill="#777">([^<]+)<', svg)
    assert ticks == ["0", "200K", "400K", "600K", "800K"]
    margin = float(re.search(r'<line x1="([\d.]+)"', svg).group(1))
    assert margin >= 6 * max(map(len, ticks)) + 6


def test_a_pages_chart_labels_every_nth_category_when_they_would_collide():
    labels = [f"2026-11-{d:02d}" for d in range(1, 14)]
    svg = _chart_page(list(range(13)), labels)
    shown = re.findall(r'<text x="([\d.]+)" y="[\d.]+" font-size="10" text-anchor="middle"', svg)
    xs = [float(x) for x in shown]
    assert 1 < len(xs) < 13 and all(b - a >= 64 for a, b in zip(xs, xs[1:]))
    assert "2026-11-01" in svg and "2026-11-02" not in svg
    assert _chart_page([1, 2, 3], ["a", "b", "c"]).count('text-anchor="middle"') == 3


# The same vectors as the editor's toIsoDate (docModel.test.js).
@pytest.mark.parametrize(("text", "iso"), [
    ("2026-11-04", "2026-11-04"),
    ("November 4, 2026", "2026-11-04"),
    ("Nov 4, 2026", "2026-11-04"),
    ("Nov. 4 2026", "2026-11-04"),
    ("Wednesday, November 4, 2026", "2026-11-04"),
    ("November 4th, 2026", "2026-11-04"),
    ("4 November 2026", "2026-11-04"),
    ("4 Nov, 2026", "2026-11-04"),
    ("**December 1, 2026**", "2026-12-01"),
    ("sept 30, 2026", "2026-09-30"),
    ("February 30, 2026", None),
    ("2026-13-01", None),
    ("11/04/2026", None),
    ("November 2026", None),
    ("next week", None),
    ("TBD", None),
    ("", None),
])
def test_a_cell_reads_as_a_date_only_when_it_is_written_as_a_whole_date(text, iso):
    assert to_iso_date(text) == iso


def test_a_column_of_written_out_dates_is_a_date_column_stored_as_iso():
    (table,) = parse_markdown("| Milestone | Date |\n| --- | --- |\n| Scope | November 4, 2026 |\n"
                              "| Launch | Dec 2, 2026 |")
    assert [c["kind"] for c in table["columns"]] == ["text", "date"]
    date_col = table["columns"][1]["id"]
    assert [r["cells"][date_col] for r in table["rows"]] == ["2026-11-04", "2026-12-02"]
