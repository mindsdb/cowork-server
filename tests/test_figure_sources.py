from __future__ import annotations

import json

from cowork.harnesses.anton_harness.figure_sources import (
    FigureOrigin,
    SourceKind,
    TurnStep,
    find_figure_sources,
    question_text,
)


def _cell(tool_use_id: str, *, code: str, stdout: str, description: str = "") -> TurnStep:
    return TurnStep(
        tool_use_id=tool_use_id,
        tool_name="scratchpad",
        action="exec",
        tool_input=json.dumps({"action": "exec", "code": code}),
        content=json.dumps({
            "code": code, "stdout": stdout, "stderr": "", "error": None,
            "description": description, "estimated_time": "", "logs": "",
        }),
    )


READ_STOCK = _cell(
    "toolu_read",
    code="print(open('/workspace/stock.csv').read())",
    stdout="part,site,qty\nP100,S1,250\nP100,S2,90\n",
    description="Read stock.csv",
)


def _figure(figures, text):
    return next(f for f in figures if f.text == text)


class TestSourcedFigures:
    def test_a_value_a_step_printed_names_that_step_and_its_file(self):
        figures = find_figure_sources(
            answer="Site S1 holds 250 units of P100.", question="How many units of P100 are at site S1?",
            steps=[READ_STOCK],
        )

        assert [f.text for f in figures] == ["250"]
        assert figures[0].origin is FigureOrigin.step
        [source] = figures[0].sources
        assert (source.tool_use_id, source.kind, source.label) == ("toolu_read", SourceKind.file, "stock.csv")

    def test_a_sum_a_later_step_printed_names_the_summing_step(self):
        total = _cell("toolu_sum", code="import pandas as pd\nprint(pd.read_csv('/workspace/stock.csv').qty.sum())", stdout="340\n")

        figures = find_figure_sources(
            answer="There are 340 units across S1 and S2.", question="", steps=[READ_STOCK, total],
        )

        assert [s.tool_use_id for s in _figure(figures, "340").sources] == ["toolu_sum"]

    def test_a_value_the_answer_rounds_still_matches(self):
        step = _cell("toolu_q", code="cur.execute('select sum(cost) from orders')", stdout="1234567.89\n")

        figures = find_figure_sources(answer="Orders cost $1.2M in total.", question="", steps=[step])

        assert _figure(figures, "$1.2M").origin is FigureOrigin.step
        assert _figure(figures, "$1.2M").sources[0].kind is SourceKind.query

    def test_a_percent_matches_the_fraction_a_step_printed(self):
        step = _cell("toolu_q", code="print(late / total)", stdout="0.125\n")

        figures = find_figure_sources(answer="12.5% of orders shipped late.", question="", steps=[step])

        assert _figure(figures, "12.5%").origin is FigureOrigin.step

    def test_a_pdf_figure_carries_the_page_the_step_printed(self):
        step = _cell(
            "toolu_pdf",
            code="from pypdf import PdfReader\nfor i, p in enumerate(PdfReader('rates.pdf').pages):\n    print(f'--- page {i+1} ---')\n    print(p.extract_text())",
            stdout="--- page 1 ---\nOvertime on Line 7 costs $42 an hour.\n",
        )

        figures = find_figure_sources(
            answer="rates.pdf says overtime on Line 7 costs $42 an hour.",
            question="What does rates.pdf say an hour of overtime on Line 7 costs?",
            steps=[step],
        )

        [source] = _figure(figures, "$42").sources
        assert (source.kind, source.label, source.page) == (SourceKind.file, "rates.pdf", 1)

    def test_a_connected_data_query_is_a_query(self):
        step = _cell(
            "toolu_m",
            code="import os, requests\nr = requests.get(os.environ['DS_ERP_URL'] + '/inventory')\nprint(r.json())",
            stdout="[{'part': 'A2210', 'qty': 150, 'site': 'W3001'}]",
            description="Query inventory for supplier S005",
        )

        figures = find_figure_sources(answer="S005 holds 150 units.", question="", steps=[step])

        [source] = _figure(figures, "150").sources
        assert (source.kind, source.label) == (SourceKind.query, "Query inventory for supplier S005")


class TestFiguresWithoutASource:
    def test_a_figure_no_step_returned_has_no_source(self):
        figures = find_figure_sources(
            answer="An hour of overtime costs $30.", question="What does an hour of overtime cost?",
            steps=[READ_STOCK],
        )

        assert _figure(figures, "$30").origin is FigureOrigin.none
        assert _figure(figures, "$30").sources == []

    def test_a_cell_that_prints_a_literal_it_was_given_is_not_a_source(self):
        echo = _cell("toolu_echo", code="cost = 30\nprint(cost)", stdout="30\n")

        figures = find_figure_sources(answer="Overtime costs $30 an hour.", question="", steps=[echo])

        assert _figure(figures, "$30").origin is FigureOrigin.none

    def test_a_memorize_call_is_not_a_source(self):
        memorize = TurnStep(
            tool_use_id="toolu_mem", tool_name="memorize", action="",
            tool_input=json.dumps({"text": "Overtime costs 30 dollars"}), content="Saved: Overtime costs 30 dollars",
        )

        figures = find_figure_sources(answer="Overtime costs $30.", question="", steps=[memorize])

        assert _figure(figures, "$30").origin is FigureOrigin.none

    def test_a_non_exec_scratchpad_action_is_not_a_source(self):
        dump = TurnStep(tool_use_id="toolu_dump", tool_name="scratchpad", action="dump", tool_input="{}", content="250")

        figures = find_figure_sources(answer="S1 holds 250 units.", question="", steps=[dump])

        assert _figure(figures, "250").origin is FigureOrigin.none

    def test_a_figure_the_question_states_is_the_questions(self):
        figures = find_figure_sources(
            answer="I found no stock data for the next 3 weeks.", question="What ships in the next 3 weeks?", steps=[],
        )

        assert _figure(figures, "3").origin is FigureOrigin.question


class TestWhatIsNotAFigure:
    def test_identifiers_dates_list_markers_links_and_code_are_skipped(self):
        answer = (
            "1. Part A2210 (32GB) at W3001, question Q-03, on 2026-10-14 at 09:30, 14 October.\n"
            "2. See [the doc](https://example.com/page/42) and `qty = 99`.\n"
            "```\nprint(77)\n```\n"
            "Version 1.2.3 of P100 at S1."
        )

        assert find_figure_sources(answer=answer, question="", steps=[READ_STOCK]) == []

    def test_an_answer_with_no_numbers_has_no_figures(self):
        assert find_figure_sources(answer="The data doesn't hold that.", question="", steps=[READ_STOCK]) == []


class TestOffsets:
    def test_offsets_count_utf16_units_so_javascript_slices_the_figure(self):
        answer = "📦 S1 holds 250 units."

        [figure] = find_figure_sources(answer=answer, question="", steps=[READ_STOCK])

        utf16 = answer.encode("utf-16-le")
        assert utf16[figure.start * 2:figure.end * 2].decode("utf-16-le") == "250"
        assert figure.start == answer.index("250") + 1


class TestQuestionText:
    def test_joins_the_text_parts_of_a_multi_part_question(self):
        content = [
            {"type": "input_text", "text": "How many at S1?"},
            {"type": "input_file", "file_id": "f1"},
            {"type": "text", "text": "Within 2 days."},
        ]

        assert question_text(content) == "How many at S1?\nWithin 2 days."

    def test_a_plain_question_is_its_own_text(self):
        assert question_text("How many?") == "How many?"
