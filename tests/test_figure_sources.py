from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from uuid import uuid4

import pytest
from anton.cloud_turn.__main__ import _clip_result_content
from anton.core.backends.base import Cell
from anton.core.llm.provider import StreamToolResult, ToolCall
from anton.core.mcp.names import namespaced_tool_name
from anton.core.session import _discarded_tool_call_events
from anton.core.tools import text_file

from cowork.harnesses.anton_harness import figure_sources as figure_sources_mod
from cowork.harnesses.anton_harness.agent_files import AgentFiles, Attachment, NamedFile
from cowork.harnesses.anton_harness.figure_sources import (
    MAX_EARLIER_CELLS,
    MAX_FIGURE_SOURCES,
    MAX_LABELED_DOCUMENTS,
    MAX_QUERY_CHARS,
    MAX_REQUEST_CHARS,
    Figure,
    FigureOrigin,
    FigureSource,
    FigureSources,
    FigureStep,
    SourceKind,
    ToolArguments,
    ToolInput,
    TurnStep,
    find_figure_sources,
    mark_repeated_figures,
)
from cowork.schemas.figure_sources import EarlierAnswer
from tests._hubspot_records import HUBSPOT_DEALS, SEARCH_DEALS

# The current code reads each timed input in about a second or less on a
# laptop under coverage, so a CI runner several times slower still passes; the
# slow form each test guards against takes 15 s or more on the same laptop.
TIME_BOUND_SECONDS = 5.0


def _cell(tool_use_id: str, *, code: str, stdout: str, description: str = "", pad: str = "main") -> TurnStep:
    return TurnStep(
        tool_use_id=tool_use_id,
        tool_name="scratchpad",
        action="exec",
        tool_input=ToolInput.from_text(text=json.dumps({"action": "exec", "name": pad, "code": code})),
        content=json.dumps(asdict(Cell(code=code, stdout=stdout, stderr="", error=None, description=description))),
    )


def _tool(tool_use_id: str, name: str, tool_input: dict, content: str) -> TurnStep:
    return TurnStep(
        tool_use_id=tool_use_id, tool_name=name, action="", tool_input=ToolInput.from_text(text=json.dumps(tool_input)),
        content=content,
    )


def _whole_file_read(*, path: str, body: str) -> str:
    """read_text_file's result for a file it read whole, as anton writes it."""
    lines = body.count("\n")
    return text_file.TextRange(path=path, total_lines=lines, first=1, last=lines, body=body, stop="done").render()


def _scratchpad(tool_use_id: str, *, action: str, content: str, pad: str = "main", code: str = "") -> TurnStep:
    """A scratchpad call other than a cell that ran, with the result anton
    gives the model for it."""
    tool_input = {"action": action, "name": pad, **({"code": code} if code else {})}
    return TurnStep(
        tool_use_id=tool_use_id, tool_name="scratchpad", action=action,
        tool_input=ToolInput.from_text(text=json.dumps(tool_input)), content=content,
    )


def _earlier(step: TurnStep) -> TurnStep:
    return replace(step, earlier_turn=True)


def _discarded(tool_use_id: str, *, action: str, code: str = "") -> TurnStep:
    """A scratchpad call anton refused to run, with the result it streams for
    it in the call's place."""
    tool_input = {"action": action, "name": "main", **({"code": code} if code else {})}
    events = _discarded_tool_call_events(ToolCall(id=tool_use_id, name="scratchpad", input=tool_input, repaired=True))
    [result] = [event for event in events if isinstance(event, StreamToolResult)]
    return TurnStep(
        tool_use_id=tool_use_id, tool_name="scratchpad", action=action,
        tool_input=ToolInput.from_text(text=json.dumps(tool_input)), content=result.content,
    )


READ_STOCK = _cell(
    "toolu_read",
    code="print(open('/workspace/stock.csv').read())",
    stdout="part,site,qty\nP100,S1,250\nP100,S2,90\n",
    description="Read stock.csv",
)


def _figure(sources: FigureSources, text: str) -> Figure:
    return next(f for f in sources.figures if f.text == text)


def _only_step(cell: TurnStep, *, answer: str, files: AgentFiles | None = None) -> FigureStep:
    [step] = find_figure_sources(answer=answer, question="", steps=[cell], files=files).steps
    return step


class TestSourcedFigures:
    def test_a_value_a_step_printed_names_that_step_and_its_file(self):
        sources = find_figure_sources(
            answer="Site S1 holds 250 units of P100.", question="How many units of P100 are at site S1?",
            steps=[READ_STOCK],
        )

        assert [f.text for f in sources.figures] == ["250"]
        assert sources.figures[0].origin is FigureOrigin.step
        assert sources.figures[0].sources == [FigureSource(tool_use_id="toolu_read", page=None)]
        assert sources.steps == [FigureStep(tool_use_id="toolu_read", kind=SourceKind.file, label="stock.csv")]

    def test_a_sum_a_later_step_printed_names_only_the_summing_step(self):
        total = _cell("toolu_sum", code="import pandas as pd\nprint(pd.read_csv('/workspace/stock.csv').qty.sum())", stdout="340\n")

        sources = find_figure_sources(
            answer="There are 340 units across S1 and S2.", question="", steps=[READ_STOCK, total],
        )

        assert [s.tool_use_id for s in _figure(sources, "340").sources] == ["toolu_sum"]
        assert [step.tool_use_id for step in sources.steps] == ["toolu_sum"]

    def test_a_value_the_answer_rounds_still_matches(self):
        step = _cell("toolu_q", code="cur.execute('select sum(cost) from orders')", stdout="1234567.89\n")

        sources = find_figure_sources(answer="Orders cost $1.2M in total.", question="", steps=[step])

        assert _figure(sources, "$1.2M").origin is FigureOrigin.step
        assert sources.steps[0].kind is SourceKind.query
        assert sources.steps[0].queries == ["select sum(cost) from orders"]

    def test_a_percent_matches_the_fraction_a_step_printed(self):
        step = _cell("toolu_q", code="print(late / total)", stdout="0.125\n")

        sources = find_figure_sources(answer="12.5% of orders shipped late.", question="", steps=[step])

        assert _figure(sources, "12.5%").origin is FigureOrigin.step

    def test_a_pdf_figure_carries_the_page_the_step_printed(self):
        step = _cell(
            "toolu_pdf",
            code="from pypdf import PdfReader\nfor i, p in enumerate(PdfReader('rates.pdf').pages):\n    print(f'--- page {i+1} ---')\n    print(p.extract_text())",
            stdout="--- page 1 ---\nOvertime on Line 7 costs $42 an hour.\n",
        )

        sources = find_figure_sources(
            answer="rates.pdf says overtime on Line 7 costs $42 an hour.",
            question="What does rates.pdf say an hour of overtime on Line 7 costs?",
            steps=[step],
        )

        assert _figure(sources, "$42").sources == [FigureSource(tool_use_id="toolu_pdf", page=1)]
        assert (sources.steps[0].kind, sources.steps[0].label) == (SourceKind.file, "rates.pdf")

    def test_a_request_to_a_saved_connection_is_a_query(self):
        step = _cell(
            "toolu_m",
            code="import os, requests\nr = requests.get(os.environ['DS_ERP_URL'] + '/inventory')\nprint(r.json())",
            stdout="[{'part': 'A2210', 'qty': 150, 'site': 'W3001'}]",
            description="Query inventory for supplier S005",
        )

        assert _only_step(step, answer="S005 holds 150 units.") == FigureStep(
            tool_use_id="toolu_m", kind=SourceKind.query, label="Query inventory for supplier S005",
            requests=["GET {DS_ERP_URL}/inventory"],
        )

    def test_a_figure_in_hours_matches_a_step_that_names_hours_among_other_units(self):
        convert = _cell(
            "toolu_hours",
            code="days = stockout_days()\nprint(f'days={days:.4f} hours={days * 24:.2f}')",
            stdout="days=0.9158 hours=21.98\n",
            description="Convert the earliest stockout time to hours",
        )

        sources = find_figure_sources(answer="SKU-0127 runs out in about 22 hours.", question="", steps=[convert])

        assert _figure(sources, "22").origin is FigureOrigin.step
        assert _figure(sources, "22").sources == [FigureSource(tool_use_id="toolu_hours")]

    def test_a_figure_in_hours_names_only_the_step_that_names_hours_when_a_days_column_also_prints_it(self):
        """The days column drops out on its own: the step that names hours
        still holds the figure, so it keeps its source."""
        sample = _cell(
            "toolu_sample",
            code=(
                "cur.execute('SELECT * FROM inventory LIMIT 3')\n"
                "print([d.name for d in cur.description])\n"
                "for row in cur.fetchall():\n    print(row)"
            ),
            stdout="['sku', 'count_in_stock', 'lead_time_days']\n('SKU-0028', 313, 22)\n",
        )
        convert = _cell(
            "toolu_hours",
            code="days = stockout_days()\nprint(f'days={days:.4f} hours={days * 24:.2f}')",
            stdout="days=0.9158 hours=21.98\n",
        )

        sources = find_figure_sources(
            answer="SKU-0127 runs out in about 22 hours.", question="", steps=[sample, convert],
        )

        assert _figure(sources, "22").origin is FigureOrigin.step
        assert _figure(sources, "22").sources == [FigureSource(tool_use_id="toolu_hours")]
        assert [step.tool_use_id for step in sources.steps] == ["toolu_hours"]

    def test_a_figure_in_hours_matches_a_step_that_names_no_time_unit(self):
        step = _cell("toolu_q", code="print(round(stock / rate, 2))", stdout="21.98\n")

        sources = find_figure_sources(answer="SKU-0127 runs out in about 22 hours.", question="", steps=[step])

        assert _figure(sources, "22").sources == [FigureSource(tool_use_id="toolu_q")]

    @pytest.mark.parametrize(("body", "answer", "figure"), [
        (
            "Each production run takes a 36-hour cycle.\nThe supplier ships within 5 business days.\n",
            "Each run takes 36 hours.", "36",
        ),
        (
            "Returns are accepted within a 30-day window.\nThe warranty runs 2 years from delivery.\n",
            "You have 30 days to return it.", "30",
        ),
    ], ids=["36-hour", "30-day"])
    def test_a_figure_matches_a_document_that_writes_its_unit_right_after_a_number(
        self, body: str, answer: str, figure: str,
    ):
        """A unit right after a number names that unit, singular or not, so
        the document names the figure's unit beside the plural of another."""
        read = _tool("toolu_doc", "read_text_file", {"path": "ops.txt"}, _whole_file_read(path="ops.txt", body=body))

        sources = find_figure_sources(answer=answer, question="", steps=[read])

        assert _figure(sources, figure).sources == [FigureSource(tool_use_id="toolu_doc")]

    def test_a_figure_in_hours_matches_a_cell_that_prints_hours_glued_to_a_number(self):
        cell = _cell(
            "toolu_c",
            code="row = fetch()\nprint(f'lead: {row.lead_time_days}d, cycle: {row.cycle}h')",
            stdout="lead: 5d, cycle: 36h\n",
        )

        sources = find_figure_sources(answer="Each run takes 36 hours.", question="", steps=[cell])

        assert _figure(sources, "36").sources == [FigureSource(tool_use_id="toolu_c")]

    def test_a_figure_in_days_matches_a_step_that_names_a_unit_only_as_a_date_part(self):
        """A singular unit names a date part here, not the unit of the
        values printed beside it."""
        step = _cell(
            "toolu_q",
            code="cur.execute('SELECT month, AVG(lead_time) FROM orders GROUP BY month')\nprint(cur.fetchall())",
            stdout="[('March', 12.4)]\n",
        )

        sources = find_figure_sources(answer="Lead time averaged 12.4 days in March.", question="", steps=[step])

        assert _figure(sources, "12.4").sources == [FigureSource(tool_use_id="toolu_q")]

    @pytest.mark.parametrize(("answer", "figure", "stdout"), [
        ("Line 7 overtime costs $42 per hour.", "$42", "{'line': 7, 'overtime_rate': 42, 'shift_days': 5}\n"),
        ("SKU-0127 sells 10 units/day.", "10", "{'sku': 'SKU-0127', 'daily_units': 10, 'lead_time_hours': 6}\n"),
    ], ids=["per-hour", "slash-day"])
    def test_a_rate_keeps_its_source_in_a_step_that_names_another_time_unit(
        self, answer: str, figure: str, stdout: str,
    ):
        """A rate's time unit follows another word or a slash, so its figure
        is written in no time unit and any step that prints it holds it."""
        step = _cell("toolu_q", code="print(fetch_row())", stdout=stdout)

        sources = find_figure_sources(answer=answer, question="", steps=[step])

        assert _figure(sources, figure).sources == [FigureSource(tool_use_id="toolu_q")]

    @pytest.mark.parametrize(("answer", "figure", "stdout"), [
        ("We run 3 D printers.", "3", "{'printers': 3, 'uptime_hours': 20}\n"),
        ("Stock 22 H-beams.", "22", "{'h_beams': 22, 'lead_time_days': 5}\n"),
    ], ids=["d", "h"])
    def test_a_single_letter_after_a_figure_is_not_its_time_unit(self, answer: str, figure: str, stdout: str):
        """A lone D or H after a space starts a word, so the figure keeps its
        source in a step that names another time unit."""
        step = _cell("toolu_q", code="print(fetch_row())", stdout=stdout)

        sources = find_figure_sources(answer=answer, question="", steps=[step])

        assert _figure(sources, figure).sources == [FigureSource(tool_use_id="toolu_q")]

    @pytest.mark.parametrize(("code", "description", "stdout", "answer", "figure"), [
        (
            "cur.execute(\"SELECT AVG(lead_time) FROM orders WHERE order_date > now() - interval '12 months'\")\n"
            "print(cur.fetchone())",
            "", "(22.3,)\n", "Average lead time was 22.3 days.", "22.3",
        ),
        (
            "cur.execute(\"SELECT AVG(EXTRACT(EPOCH FROM closed_at - opened_at) / 3600) FROM tickets"
            " WHERE opened_at > now() - interval '30 days'\")\nprint(cur.fetchone())",
            "", "(4.2,)\n", "Tickets took 4.2 hours on average.", "4.2",
        ),
        (
            "print(average_lead_time())", "Average lead time over the last 12 months", "22.3\n",
            "Lead time was 22.3 days.", "22.3",
        ),
    ], ids=["sql-months", "sql-days", "label-months"])
    def test_a_figure_keeps_a_step_whose_code_or_label_names_another_time_unit_as_its_window(
        self, code: str, description: str, stdout: str, answer: str, figure: str,
    ):
        """A window the step filters on names its period, not its output's
        unit, so only the output's units rule a step out."""
        step = _cell("toolu_q", code=code, stdout=stdout, description=description)

        sources = find_figure_sources(answer=answer, question="", steps=[step])

        assert _figure(sources, figure).origin is FigureOrigin.step
        assert _figure(sources, figure).sources == [FigureSource(tool_use_id="toolu_q")]


class TestTheStepsList:
    def test_each_named_step_is_listed_once_in_step_order(self):
        first = _cell("toolu_1", code="print(a)", stdout="250\n")
        unnamed = _cell("toolu_2", code="print(b)", stdout="999\n")
        third = _cell("toolu_3", code="print(c)", stdout="90\n340\n")

        sources = find_figure_sources(
            answer="S2 holds 90, S1 holds 250, 340 in all; 90 of them are P100.",
            question="", steps=[first, unnamed, third],
        )

        assert [step.tool_use_id for step in sources.steps] == ["toolu_1", "toolu_3"]
        named = {source.tool_use_id for figure in sources.figures for source in figure.sources}
        assert named == {step.tool_use_id for step in sources.steps}

    def test_a_figure_two_steps_hold_names_both_in_step_order(self):
        again = _cell("toolu_again", code="print(df.qty.max())", stdout="250\n")

        sources = find_figure_sources(answer="S1 holds 250 units.", question="", steps=[READ_STOCK, again])

        assert [s.tool_use_id for s in _figure(sources, "250").sources] == ["toolu_read", "toolu_again"]


class TestStepKinds:
    def test_a_comment_or_a_ds_named_variable_does_not_make_a_query(self):
        step = _cell(
            "toolu_c",
            code="# select the rows from df\nds_total = df.qty.sum()\nprint(ds_total)",
            stdout="340\n",
        )

        assert _only_step(step, answer="340 units in all.") == FigureStep(
            tool_use_id="toolu_c", kind=SourceKind.step, label="",
        )

    def test_the_data_helper_is_a_query_even_when_its_sql_is_out_of_sight(self):
        step = _cell("toolu_q", code="print(query_minds_data(3, build_sql()))", stdout="250\n", description="Stock at S1")

        assert (_only_step(step, answer="S1 holds 250.").kind, _only_step(step, answer="S1 holds 250.").label) == (
            SourceKind.query, "Stock at S1",
        )

    def test_reading_a_saved_connections_variable_is_a_query(self):
        step = _cell(
            "toolu_sdk", code="import os\nclient = Erp(token=os.environ['DS_ERP_TOKEN'])\nprint(client.stock('S1'))",
            stdout="250\n",
        )

        assert _only_step(step, answer="S1 holds 250.").kind is SourceKind.query

    def test_a_web_search_cell_is_web_with_the_url_it_printed(self):
        step = _cell(
            "toolu_web", code="print(web_search('steel price per ton'))",
            stdout="Steel trades at $640 a ton (https://example.com/rates?src=search#top)\n",
        )

        step_out = _only_step(step, answer="Steel trades at $640 a ton.")

        assert (step_out.kind, step_out.url, step_out.label) == (
            SourceKind.web, "https://example.com/rates", "example.com/rates",
        )

    def test_a_request_to_a_literal_url_is_web_without_its_query_string(self):
        step = _cell(
            "toolu_api",
            code="r = requests.get('https://example.com/rates?region=eu', headers={'X-Api-Key': 'k-123'})\nprint(r.json()['rate'])",
            stdout="42\n",
        )

        assert _only_step(step, answer="The rate is 42.") == FigureStep(
            tool_use_id="toolu_api", kind=SourceKind.web, label="example.com/rates",
            url="https://example.com/rates", requests=["GET https://example.com/rates"],
        )

    def test_a_web_fetch_tool_is_web_without_its_credentials_or_query_string(self):
        step = _tool("toolu_f", "web_fetch", {"url": "https://erp:pw@example.com/rates?key=abc#x"}, "Rate: 42")

        step_out = _only_step(step, answer="The rate is 42.")

        assert (step_out.kind, step_out.url, step_out.label) == (
            SourceKind.web, "https://example.com/rates", "example.com/rates",
        )

    def test_a_query_inside_an_fstring_placeholder_makes_a_query(self):
        step = _cell(
            "toolu_q",
            code="""print(f"Total stock: {query_minds_data('SELECT SUM(qty) FROM stock').iloc[0, 0]}")""",
            stdout="Total stock: 250",
        )

        step_out = _only_step(step, answer="Total stock is 250.")

        assert (step_out.kind, step_out.queries) == (SourceKind.query, ["SELECT SUM(qty) FROM stock"])

    def test_a_request_through_a_name_an_earlier_cell_bound_to_a_saved_connection_is_a_query(self):
        bind = _cell("toolu_a", code="import os\nBASE = os.environ['DS_ERP_URL']", stdout="")
        fetch = _cell(
            "toolu_b", code='r = requests.get(f"{BASE}/inventory")\nprint(r.json()[\'qty\'])', stdout="250\n",
        )

        [step] = find_figure_sources(answer="The ERP holds 250.", question="", steps=[bind, fetch]).steps

        assert step == FigureStep(
            tool_use_id="toolu_b", kind=SourceKind.query, label="", requests=["GET {DS_ERP_URL}/inventory"],
        )

    def test_a_printed_label_that_reads_like_sql_leaves_the_cell_a_file_step(self):
        step = _cell(
            "toolu_f",
            code='df = pd.read_csv("stock.csv"); print("With rows from stock.csv:", int(df.qty.sum()))',
            stdout="With rows from stock.csv: 250",
        )

        step_out = _only_step(step, answer="250")

        assert (step_out.kind, step_out.label, step_out.queries) == (SourceKind.file, "stock.csv", [])

    @pytest.mark.parametrize(("name", "tool_input", "kind", "label"), [
        ("web_search", {"query": "overtime rate"}, SourceKind.web, "overtime rate"),
        ("web_fetch", {"url": "https://example.com/rates"}, SourceKind.web, "example.com/rates"),
        ("read_text_file", {"path": "docs/rates.md"}, SourceKind.file, "rates.md"),
    ])
    def test_a_tool_that_reads_the_web_or_a_file_is_a_source(
        self, name: str, tool_input: dict, kind: SourceKind, label: str,
    ):
        step = _tool("toolu_t", name, tool_input, "Overtime costs 42 an hour.")

        sources = find_figure_sources(answer="Overtime costs $42 an hour.", question="", steps=[step])

        assert _origins(sources) == [("$42", FigureOrigin.step)]
        assert [(s.kind, s.label) for s in sources.steps] == [(kind, label)]

    @pytest.mark.parametrize("name", [
        # Earlier answers, verbatim.
        "recall_history", "recall",
        # The agent's own artifacts and skills.
        "open_artifact", "list_artifacts", "create_artifact", "update_artifact", "generate_artifact",
        "recall_skill", "create_skill_draft", "publish",
        # What the agent wrote, or the user answered.
        "memorize", "ask_user", "select_path",
        # A line about the image, not what it shows.
        "read_image",
        # Connection setup.
        "connect_datasource", "lookup_connector",
    ])
    def test_any_other_tools_result_is_not_a_source(self, name: str):
        step = _tool("toolu_t", name, {"query": "overtime"}, "Overtime typically costs $42 an hour for 12 people.")

        sources = find_figure_sources(answer="Overtime costs $42 an hour for 12 people.", question="", steps=[step])

        assert _origins(sources) == [("$42", FigureOrigin.none), ("12", FigureOrigin.none)]
        assert sources.steps == []

    def test_a_cell_with_no_description_sends_an_empty_label(self):
        """The client words an empty label by the kind alone, so a
        placeholder would read 'From a query: a query'."""
        query = _cell("toolu_q", code="print(query_minds_data('SELECT qty FROM stock'))", stdout="250\n")
        web = _cell("toolu_w", code="print(web_search('steel price'))", stdout="Steel is $640 a ton.\n")
        calc = _cell("toolu_c", code="print(df.qty.max())", stdout="90\n")

        sources = find_figure_sources(answer="250 units, $640 a ton, 90 at most.", question="", steps=[query, web, calc])

        assert [(s.kind, s.label) for s in sources.steps] == [
            (SourceKind.query, ""), (SourceKind.web, ""), (SourceKind.step, ""),
        ]


class TestReadTextFile:
    def _read(self, *, header: str, body: str = "part,site,qty\nP100,S1,250\nP100,S2,90\n", **arguments) -> TurnStep:
        return _tool("toolu_t", "read_text_file", {"path": "data/stock.csv", **arguments}, f"{header}\n{body}")

    def test_a_file_it_read_is_a_file_step_with_its_path(self, tmp_path: Path):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "stock.csv").write_text("part,site,qty\nP100,S1,250\nP100,S2,90\n")
        step = self._read(header="data/stock.csv — lines 1-3 of 3")

        step_out = _only_step(step, answer="S1 holds 250 units.", files=AgentFiles.on_this_server(tmp_path))

        assert (step_out.kind, step_out.label, step_out.path) == (SourceKind.file, "stock.csv", "data/stock.csv")

    @pytest.mark.parametrize("header", [
        "data/stock.csv — lines 1-3 of 3",
        "data/stock.csv — lines 1-3 of 3 (partial; continue with start_line=4)",
    ])
    def test_its_line_range_header_is_not_the_files_text(self, header: str):
        sources = find_figure_sources(answer="The file has 3 rows; S1 holds 250.", question="", steps=[self._read(header=header)])

        assert _origins(sources) == [("3", FigureOrigin.none), ("250", FigureOrigin.step)]

    def test_the_line_range_it_asked_for_does_not_hide_a_value_the_file_holds(self):
        step = self._read(header="data/stock.csv — lines 1-250 of 900", start_line=1, end_line=250)

        assert _origins(find_figure_sources(answer="S1 holds 250.", question="", steps=[step])) == [
            ("250", FigureOrigin.step),
        ]

    def test_a_long_line_marker_is_not_the_files_text(self):
        step = self._read(header="data/stock.csv — lines 1-1 of 1", body="P100 … [line truncated: 4096 chars] … 250\n")

        sources = find_figure_sources(answer="A line of 4096 characters; S1 holds 250.", question="", steps=[step])

        assert _origins(sources) == [("4096", FigureOrigin.none), ("250", FigureOrigin.step)]

    @pytest.mark.parametrize(("stop", "shortened"), [("done", 0), ("page", 0), ("chars", 0), ("page", 2)])
    def test_antons_header_and_long_line_mark_are_not_the_files_text(self, stop: str, shortened: int):
        """The header and mark as anton's read_text_file writes them."""
        line = "P100," + "x" * 5_000 + ",250"
        content = text_file.TextRange(
            path="data/stock.csv", total_lines=900, first=1, last=2, body=f"part,qty\n{text_file._shorten(line)}\n",
            stop=stop, shortened=shortened,
        ).render()
        step = _tool("toolu_t", "read_text_file", {"path": "data/stock.csv"}, content)

        sources = find_figure_sources(
            answer=f"900 lines, {len(line)} characters; S1 holds 250.", question="", steps=[step],
        )

        assert _origins(sources) == [
            ("900", FigureOrigin.none), (str(len(line)), FigureOrigin.none), ("250", FigureOrigin.step),
        ]

    @pytest.mark.parametrize("path", [
        ".anton/memory/lessons.md",
        "/Users/ada/sales/.anton/memory/rules.md",
        ".anton/memory/topics/overtime.md",
        ".anton/artifacts/../memory/lessons.md",
        "./.anton//memory/lessons.md",
        ".Anton/Memory/lessons.md",
        ".anton\\memory\\lessons.md",
        " .anton/memory/lessons.md",
        "\t.anton/memory/lessons.md ",
    ], ids=[
        "relative", "absolute", "topic", "climbs-back", "dot-and-doubled-slash", "other-case", "backslashes",
        "leading-space", "tab-and-trailing-space",
    ])
    def test_a_read_of_a_memory_file_is_not_a_source(self, path: str):
        """A memory file holds what the agent memorized, its own text, as a
        memorize call's result does."""
        read = _tool(
            "toolu_mem", "read_text_file", {"path": path},
            _whole_file_read(path=path, body="Overtime costs 30 dollars an hour.\n"),
        )

        sources = find_figure_sources(answer="Overtime costs $30 an hour.", question="", steps=[read])

        assert _origins(sources) == [("$30", FigureOrigin.none)]
        assert sources.steps == []

    @pytest.mark.parametrize("code", [
        "print(open('.anton/memory/lessons.md').read())",
        "print(open('/workspace/.anton/memory/rules.md').read())",
        "from pathlib import Path\nprint(Path('.Anton/Memory/topics/overtime.md').read_text())",
    ], ids=["relative", "absolute", "other-case-topic"])
    def test_a_cell_that_only_reads_a_memory_file_is_not_a_source(self, code: str):
        """A cell that reads a memory file prints the agent's own text, as a
        read_text_file of it does."""
        cell = _cell("toolu_mem", code=code, stdout="Overtime costs 30 dollars an hour.\n")

        sources = find_figure_sources(answer="Overtime costs $30 an hour.", question="", steps=[cell])

        assert _origins(sources) == [("$30", FigureOrigin.none)]
        assert sources.steps == []

    def test_a_cell_that_reads_a_memory_file_and_another_file_names_only_the_other(self):
        code = (
            "import pandas as pd\nnotes = open('.anton/memory/lessons.md').read()\n"
            "print(pd.read_csv('stock.csv').qty.sum())"
        )
        cell = _cell("toolu_both", code=code, stdout="340\n")

        sources = find_figure_sources(answer="Stock totals 340.", question="", steps=[cell])

        assert _origins(sources) == [("340", FigureOrigin.step)]
        assert [(step.kind, step.label) for step in sources.steps] == [(SourceKind.file, "stock.csv")]

    def test_a_cell_that_reads_a_memory_file_and_prints_an_earlier_cells_data_stays_a_step(self):
        load = _cell("toolu_load", code="import pandas as pd\ndf = pd.read_csv('stock.csv')", stdout="")
        both = _cell(
            "toolu_both", code="notes = open('.anton/memory/lessons.md').read()\nprint(df.qty.sum())", stdout="340\n",
        )

        sources = find_figure_sources(answer="Stock totals 340.", question="", steps=[load, both])

        assert _origins(sources) == [("340", FigureOrigin.step)]
        assert [(step.tool_use_id, step.kind) for step in sources.steps] == [
            ("toolu_load", SourceKind.file), ("toolu_both", SourceKind.step),
        ]

    def test_a_cell_that_reads_a_memory_file_and_makes_a_request_stays_a_step(self):
        """A request whose URL the cell builds at run time, here from the
        memory file, may fetch live data."""
        code = (
            "import json, requests\nurl = json.load(open('.anton/memory/endpoints.json'))['sales']\n"
            "print(requests.get(url).json()['total'])"
        )
        cell = _cell("toolu_req", code=code, stdout="4812\n")

        sources = find_figure_sources(answer="Sales total 4,812.", question="", steps=[cell])

        assert _origins(sources) == [("4,812", FigureOrigin.step)]
        assert [(step.tool_use_id, step.kind, step.requests) for step in sources.steps] == [
            ("toolu_req", SourceKind.step, ["GET {url}"]),
        ]

    @pytest.mark.parametrize("read", [
        "total = sum(pd.read_csv(os.path.join('data', f)).qty.sum() for f in os.listdir('data'))",
        "total = sum(pd.read_csv(Path('data') / f.name).qty.sum() for f in Path('data').iterdir())",
        "total = sum(pd.read_csv(os.path.join('data', row['file'])).qty.sum() for row in manifest)",
        "total = pd.read_csv(f'data\\\\{name}').qty.sum()",
        "total = pd.read_csv('data/{}'.format(name)).qty.sum()",
        "total = sum(pd.read_csv(f).qty.sum() for f in glob.glob('data/*'))",
    ], ids=["joined-name", "path-attribute", "joined-subscript", "backslash", "format-positional", "glob"])
    def test_a_cell_that_reads_a_memory_file_and_a_file_by_a_run_time_path_stays_a_step(self, read: str):
        """A file read whose path the cell builds at run time may read the
        customer's data, as a request whose URL it builds may fetch it. The
        cell reader names that path data/{f}, which ends in no document's
        extension, so without this rule the cell reads as one whose only
        source is the memory file."""
        code = (
            "import glob, os, pandas as pd\nfrom pathlib import Path\nnotes = open('.anton/memory/lessons.md').read()\n"
            f"{read}\nprint(total)"
        )
        cell = _cell("toolu_csv", code=code, stdout="4812\n")

        sources = find_figure_sources(answer="Stock totals 4,812 units.", question="", steps=[cell])

        assert _origins(sources) == [("4,812", FigureOrigin.step)]
        assert [(step.tool_use_id, step.kind) for step in sources.steps] == [("toolu_csv", SourceKind.step)]

    @pytest.mark.parametrize("code", [
        "notes = open('.anton/memory/lessons.md').read()\nextra = open(f'.anton/memory/{topic}.md').read()\n"
        "print(notes, extra)",
        "notes = open('.anton/memory/lessons.md').read()\nwith open(f'out/{name}.txt', 'w') as fh:\n"
        "    fh.write(notes)\nprint(notes)",
        "notes = open('.anton/memory/lessons.md').read()\nprint(f'{done}/{total} lessons')\nprint(notes)",
    ], ids=["a-memory-file-by-a-run-time-path", "a-file-it-writes", "a-printed-caption"])
    def test_a_run_time_string_that_names_no_file_read_outside_memory_leaves_the_cell_out(self, code: str):
        """A run-time path inside the memory folder, a file the cell writes
        and a caption it prints read none of the customer's data, so the
        cell still prints only what the agent memorized."""
        cell = _cell("toolu_mem", code=code, stdout="Overtime costs 30 dollars an hour.\n")

        sources = find_figure_sources(answer="Overtime costs $30 an hour.", question="", steps=[cell])

        assert _origins(sources) == [("$30", FigureOrigin.none)]
        assert sources.steps == []

    @pytest.mark.parametrize("cells", [
        [("toolu_load", "notes = open('.anton/memory/lessons.md').read()"), ("toolu_print", "print(notes)")],
        [
            ("toolu_load", "import json\nrates = json.load(open('.anton/memory/rates.json'))"),
            ("toolu_pick", "overtime = rates['overtime']"),
            ("toolu_print", "print('Overtime costs', overtime, 'dollars an hour.')"),
        ],
    ], ids=["printed-by-the-next-cell", "through-a-cell-between"])
    def test_a_cell_that_prints_only_memory_text_an_earlier_cell_read_is_not_a_source(
        self, cells: list[tuple[str, str]],
    ):
        """An earlier cell left out for reading memory files passes on what
        it read, so a later cell whose only source is that cell prints what
        the agent memorized too."""
        steps = [
            _cell(tool_use_id, code=code, stdout="Overtime costs 30 dollars an hour.\n" if "print" in code else "")
            for tool_use_id, code in cells
        ]

        sources = find_figure_sources(answer="Overtime costs $30 an hour.", question="", steps=steps)

        assert _origins(sources) == [("$30", FigureOrigin.none)]
        assert sources.steps == []

    def test_a_cell_that_reads_a_memory_file_and_queries_stays_a_query(self):
        code = (
            "notes = open('.anton/memory/lessons.md').read()\n"
            "print(query_minds_data('SELECT SUM(qty) FROM stock').iloc[0, 0])"
        )
        cell = _cell("toolu_q", code=code, stdout="340\n", description="Sum stock")

        sources = find_figure_sources(answer="Stock totals 340.", question="", steps=[cell])

        assert _origins(sources) == [("340", FigureOrigin.step)]
        assert [(step.kind, step.label) for step in sources.steps] == [(SourceKind.query, "Sum stock")]

    @pytest.mark.parametrize("path", [
        "docs/lessons.md", "memory/lessons.md", ".anton/artifacts/memory/lessons.md", ".anton/memories/lessons.md",
        ".anton/memory/../artifacts/lessons.md",
    ])
    def test_a_read_of_any_other_file_holding_the_figure_is_a_source(self, path: str):
        read = _tool(
            "toolu_t", "read_text_file", {"path": path},
            _whole_file_read(path=path, body="Overtime costs 30 dollars an hour.\n"),
        )

        sources = find_figure_sources(answer="Overtime costs $30 an hour.", question="", steps=[read])

        assert _origins(sources) == [("$30", FigureOrigin.step)]
        assert [(step.kind, step.label) for step in sources.steps] == [(SourceKind.file, "lessons.md")]


# HubSpot's search and query tools as anton registers them for a connection.
SEARCH_CRM_OBJECTS = namespaced_tool_name("hubspot", "search_crm_objects")
QUERY_CRM_DATA = namespaced_tool_name("hubspot", "query_crm_data")
DEALS_ANSWER = "Your open deals total $61,150: Acme at $48,250 and Globex at $12,900."


class TestConnectorTools:
    """A connector tool is an MCP tool anton registers as {engine}__{tool},
    or {engine}__{connection}__{tool} when the turn has two connections of
    one engine."""

    def test_a_connector_tool_that_returns_the_customers_records_is_a_query_source(self):
        search = _tool("toolu_hs", SEARCH_CRM_OBJECTS, SEARCH_DEALS, HUBSPOT_DEALS)

        sources = find_figure_sources(answer=DEALS_ANSWER, question="", steps=[search])

        # The model added the two amounts up itself, so the total has no source.
        assert _origins(sources) == [
            ("$61,150", FigureOrigin.none), ("$48,250", FigureOrigin.step), ("$12,900", FigureOrigin.step),
        ]
        assert sources.steps == [
            FigureStep(tool_use_id="toolu_hs", kind=SourceKind.query, label="hubspot: search_crm_objects"),
        ]

    @pytest.mark.parametrize(("name", "label"), [
        (namespaced_tool_name("hubspot", "search_crm_objects", connection_name="crm2"), "hubspot (crm2): search_crm_objects"),
        (
            namespaced_tool_name("hubspot", "search_crm_objects", connection_name="acme__eu"),
            "hubspot (acme__eu): search_crm_objects",
        ),
    ])
    def test_a_connector_tool_named_with_its_connection_names_the_connection(self, name: str, label: str):
        search = _tool("toolu_hs", name, SEARCH_DEALS, HUBSPOT_DEALS)

        step = _only_step(search, answer="Acme's deal is $48,250.")

        assert (step.kind, step.label) == (SourceKind.query, label)

    def test_a_secret_in_a_connections_name_is_redacted_in_the_label(self):
        search = _tool(
            "toolu_hs", namespaced_tool_name("hubspot", "search_crm_objects", connection_name="token=tok_XyZsecret"),
            SEARCH_DEALS, HUBSPOT_DEALS,
        )

        event = _event(search, answer="Acme's deal is $48,250.")

        [step] = FigureSources.model_validate_json(event).steps
        assert step.label.startswith("hubspot (token=[redacted]")
        assert "tok_XyZsecret" not in event

    @pytest.mark.parametrize(("tool", "content"), [
        ("tool_guidance", "search_crm_objects returns up to 200 records per call; use at most 5 filterGroups."),
        ("get_properties", json.dumps({"results": [{"name": "amount", "displayOrder": 200, "groupName": "dealinfo"}]})),
    ])
    def test_a_connector_tool_that_returns_docs_or_schema_is_not_a_source(self, tool: str, content: str):
        guidance = _tool("toolu_hs", namespaced_tool_name("hubspot", tool), {}, content)

        sources = find_figure_sources(answer="HubSpot returns at most 200 deals per search.", question="", steps=[guidance])

        assert _origins(sources) == [("200", FigureOrigin.none)]
        assert sources.steps == []

    def test_a_write_tools_output_is_not_a_source(self):
        """A write tool's name can't tell a view from a write, so none counts.
        This create echoes what the model wrote, and the defaults the server
        filled in are not the customer's records."""
        create = _tool(
            "toolu_hs", namespaced_tool_name("hubspot", "manage_crm_objects"),
            {"action": "CREATE", "objectType": "deals", "properties": {"dealname": "Globex", "amount": "75000"}},
            json.dumps({"id": "18234567892", "properties": {"amount": "75000", "hs_probability": "0.2"}}),
        )

        sources = find_figure_sources(
            answer="I created the Globex deal at $75,000 with a 0.2 probability.", question="", steps=[create],
        )

        assert _origins(sources) == [("$75,000", FigureOrigin.none), ("0.2", FigureOrigin.none)]
        assert sources.steps == []

    def test_a_tool_of_an_engine_anton_has_not_classified_is_not_a_source(self):
        issues = _tool("toolu_lin", "linear__list_issues", {"state": "open"}, json.dumps({"open_issues": 12}))

        sources = find_figure_sources(answer="There are 12 open issues.", question="", steps=[issues])

        assert _origins(sources) == [("12", FigureOrigin.none)]
        assert sources.steps == []

    def test_a_number_the_calls_input_states_is_only_echoed_by_its_records(self):
        search = _tool(
            "toolu_hs", SEARCH_CRM_OBJECTS,
            {"objectType": "deals", "filterGroups": [{"filters": [
                {"propertyName": "amount", "operator": "EQ", "value": "48250"},
            ]}]},
            HUBSPOT_DEALS,
        )

        sources = find_figure_sources(answer="Acme's deal is $48,250 and Globex's $12,900.", question="", steps=[search])

        assert _origins(sources) == [("$48,250", FigureOrigin.none), ("$12,900", FigureOrigin.step)]

    @pytest.mark.parametrize(("created", "modified"), [
        ("2026-03-04T09:12:44.512Z", "2026-10-12T22:05:31.087Z"),
        ("2026-03-04 09:12:44", "2026-10-12T22:05:31+05:00"),
    ])
    def test_the_timestamps_on_a_connectors_records_source_no_figure(self, created: str, modified: str):
        """HubSpot stamps every record with when it was created and last
        changed. Read as numbers, the minutes and zone offsets of those times
        would source almost any small figure."""
        record = {
            "id": "18234567890",
            "properties": {
                "dealname": "Acme renewal", "amount": "48250.00", "createdate": created, "hs_lastmodifieddate": modified,
            },
            "createdAt": created,
            "updatedAt": modified,
        }
        records = json.dumps({"total": 1, "results": [record]})
        search = _tool("toolu_hs", SEARCH_CRM_OBJECTS, SEARCH_DEALS, records)

        sources = find_figure_sources(
            answer="Acme's deal is $48,250; win rate is 12% across 5 reps.", question="", steps=[search],
        )

        assert _origins(sources) == [
            ("$48,250", FigureOrigin.step), ("12%", FigureOrigin.none), ("5", FigureOrigin.none),
        ]

    def test_a_date_time_the_calls_filter_states_echoes_no_figure(self):
        """Read as numbers, a filter's date-time would make the records'
        values equal to its hour, minutes or seconds read as echoes of the
        input: here the minutes of 23:59:59."""
        search = _tool(
            "toolu_hs", SEARCH_CRM_OBJECTS,
            {"objectType": "deals", "filterGroups": [{"filters": [
                {"propertyName": "closedate", "operator": "LTE", "value": "2026-03-31T23:59:59Z"},
            ]}]},
            json.dumps({"total": 59, "results": []}),
        )

        sources = find_figure_sources(answer="59 deals closed in Q1.", question="", steps=[search])

        assert _origins(sources) == [("59", FigureOrigin.step)]

    @pytest.mark.parametrize(("arguments", "queries"), [
        (
            {"sql": "SELECT dealname, amount FROM deals WHERE dealstage = 'contractsent' AND token = 'XyZs3cr3t'"},
            ["SELECT dealname, amount FROM deals WHERE dealstage = 'contractsent' AND token = '[redacted]'"],
        ),
        ({"objectType": "deals"}, []),
    ], ids=["with-its-statement", "without-one"])
    def test_a_connector_query_lists_the_statement_it_ran_redacted(self, arguments: dict, queries: list[str]):
        """HubSpot's query_crm_data takes a SQL-shaped statement as its sql
        argument, which the step lists as a cell's SQL is listed."""
        query = _tool("toolu_hs", QUERY_CRM_DATA, arguments, HUBSPOT_DEALS)

        event = _event(query, answer="Acme's deal is $48,250.")

        [step] = FigureSources.model_validate_json(event).steps
        assert (step.kind, step.label, step.queries) == (SourceKind.query, "hubspot: query_crm_data", queries)
        assert "XyZs3cr3t" not in event

    def test_an_earlier_turns_connector_step_holds_no_figure(self):
        earlier = _earlier(_tool("toolu_old", SEARCH_CRM_OBJECTS, SEARCH_DEALS, HUBSPOT_DEALS))

        sources = find_figure_sources(answer="Acme's deal is $48,250.", question="", steps=[earlier])

        assert _origins(sources) == [("$48,250", FigureOrigin.none)]
        assert sources.steps == []

    def test_an_anton_that_cannot_classify_connector_tools_counts_none(self, tmp_path: Path):
        """An anton older than mindsdb/anton#585 has an anton.core.mcp with no
        returns_records and no names module. The module still imports there,
        and only the tools anton runs itself are evidence."""
        program = textwrap.dedent(f"""
            import sys
            import anton.core.mcp.access
            del anton.core.mcp.access.returns_records
            sys.modules["anton.core.mcp.names"] = None
            from cowork.harnesses.anton_harness.figure_sources import is_evidence_tool
            print([is_evidence_tool(name=name) for name in ({SEARCH_CRM_OBJECTS!r}, "web_fetch")])
        """)

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=60, check=False,
            env={**os.environ, "COWORK_HOME": str(tmp_path), "DATABASE_URI": "sqlite://"},
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines()[-1] == "[False, True]"

    def test_importing_the_module_does_not_import_the_mcp_sdk(self, tmp_path: Path):
        """The module splits a connector tool's name with anton's
        anton.core.mcp.names, which does not import the MCP SDK, rather than
        anton.core.mcp.registry, which does. It runs in a fresh interpreter,
        because this test process imported the SDK already."""
        program = textwrap.dedent("""
            import sys
            import cowork.harnesses.anton_harness.figure_sources
            print(sorted(m for m in sys.modules if m.split(".")[0] in ("mcp", "mcp_types")))
        """)

        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=60, check=False,
            env={**os.environ, "COWORK_HOME": str(tmp_path), "DATABASE_URI": "sqlite://"},
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines()[-1] == "[]"


class TestQueriesAcrossCells:
    def test_a_query_an_earlier_cell_spelled_is_the_later_cells(self):
        spell = _cell("toolu_1", code="sql = \"SELECT qty FROM stock WHERE site = 'S1'\"", stdout="")
        run = _cell("toolu_2", code="print(query_minds_data(3, sql))", stdout="250\n")

        step = find_figure_sources(answer="S1 holds 250.", question="", steps=[spell, run]).steps[0]

        assert (step.tool_use_id, step.queries) == ("toolu_2", ["SELECT qty FROM stock WHERE site = 'S1'"])

    @pytest.mark.parametrize(("action", "content"), [
        ("reset", "Scratchpad 'main' reset. All state cleared."),
        ("remove", "Scratchpad 'main' removed."),
    ])
    def test_a_reset_that_ran_forgets_the_strings_its_cells_bound(self, action: str, content: str):
        spell = _cell("toolu_1", code="sql = 'SELECT qty FROM stock'", stdout="")
        reset = _scratchpad("toolu_reset", action=action, content=content)
        run = _cell("toolu_2", code="cur.execute(sql)\nprint(cur.fetchone()[0])", stdout="250\n")

        step = find_figure_sources(answer="S1 holds 250.", question="", steps=[spell, reset, run]).steps[0]

        assert step.queries == []

    def test_a_reset_anton_discarded_keeps_the_strings_its_cells_bound(self):
        spell = _cell("toolu_1", code="sql = 'SELECT qty FROM stock'", stdout="")
        discarded = _discarded("toolu_reset", action="reset")
        run = _cell("toolu_2", code="cur.execute(sql)\nprint(cur.fetchone()[0])", stdout="250\n")

        step = find_figure_sources(answer="S1 holds 250.", question="", steps=[spell, discarded, run]).steps[0]

        assert step.queries == ["SELECT qty FROM stock"]

    def test_a_cell_anton_discarded_neither_binds_names_nor_is_a_step(self):
        spell = _cell("toolu_1", code="sql = 'SELECT qty FROM stock'", stdout="")
        discarded = _discarded("toolu_x", action="exec", code="sql = 'SELECT 250 FROM other'")
        run = _cell("toolu_2", code="cur.execute(sql)\nprint(cur.fetchone()[0])", stdout="250\n")

        sources = find_figure_sources(answer="S1 holds 250.", question="", steps=[spell, discarded, run])

        assert [(s.tool_use_id, s.queries) for s in sources.steps] == [("toolu_2", ["SELECT qty FROM stock"])]

    def test_another_scratchpads_strings_are_not_this_ones(self):
        spell = _cell("toolu_1", code="sql = 'SELECT qty FROM stock'", stdout="", pad="other")
        run = _cell("toolu_2", code="cur.execute(sql)\nprint(cur.fetchone()[0])", stdout="250\n")

        assert find_figure_sources(answer="S1 holds 250.", question="", steps=[spell, run]).steps[0].queries == []


class TestClientsAcrossCells:
    def test_a_client_keeps_its_base_in_the_later_cells_serialized_request(self):
        setup = _cell("setup", code="client = httpx.Client(base_url='https://erp.example.com')", stdout="")
        run = _cell("run", code="print(client.get('/stock?token=synthetic-secret').text)", stdout="250\n")

        event = find_figure_sources(answer="There are 250 units.", question="", steps=[setup, run]).model_dump_json()

        [step] = FigureSources.model_validate_json(event).steps
        assert (step.kind, step.requests) == (SourceKind.web, ["GET https://erp.example.com/stock"])
        assert "synthetic-secret" not in event

    @pytest.mark.parametrize("action", ["reset", "remove"])
    def test_a_namespace_reset_forgets_the_client_base(self, action: str):
        setup = _cell("setup", code="client = httpx.Client(base_url='https://erp.example.com')", stdout="")
        reset = _scratchpad("reset", action=action, content="Scratchpad state cleared.")
        run = _cell("run", code="print(client.get('/stock').text)", stdout="250\n")

        [step] = find_figure_sources(answer="There are 250 units.", question="", steps=[setup, reset, run]).steps

        assert step.requests == []

    def test_another_scratchpads_client_is_not_in_scope(self):
        setup = _cell("setup", code="client = httpx.Client(base_url='https://erp.example.com')", stdout="", pad="other")
        run = _cell("run", code="print(client.get('/stock').text)", stdout="250\n")

        [step] = find_figure_sources(answer="There are 250 units.", question="", steps=[setup, run]).steps

        assert step.requests == []

    def test_a_discarded_reset_preserves_the_client_base(self):
        setup = _cell("setup", code="client = httpx.Client(base_url='https://erp.example.com')", stdout="")
        run = _cell("run", code="print(client.get('/stock').text)", stdout="250\n")

        [step] = find_figure_sources(
            answer="There are 250 units.", question="", steps=[setup, _discarded("reset", action="reset"), run],
        ).steps

        assert step.requests == ["GET https://erp.example.com/stock"]


class TestFunctionsAcrossCells:
    def test_a_rebound_global_query_never_survives_in_the_serialized_event(self):
        define = _cell("define", code="SQL = 'SELECT qty FROM old_stock'\ndef load():\n    return cur.execute(SQL)", stdout="")
        rebind = _cell("rebind", code="SQL = 'SELECT qty FROM new_stock'", stdout="")
        call = _cell("call", code="print(load().fetchone()[0])", stdout="250\n")

        event = find_figure_sources(answer="There are 250 units.", question="", steps=[define, rebind, call]).model_dump_json()

        assert "old_stock" not in event
        assert all(not step.queries for step in FigureSources.model_validate_json(event).steps)

    def test_a_helper_an_earlier_cell_defined_makes_its_request_in_the_cell_that_calls_it(self):
        define = _cell(
            "toolu_1",
            code=(
                "import requests\nBASE = 'https://erp.example.com/api'\n"
                "def get(path):\n    return requests.get(BASE + path).json()"
            ),
            stdout="",
        )
        call = _cell("toolu_2", code="print(get('/stock?site=S1')['qty'])", stdout="250\n")

        sources = find_figure_sources(answer="S1 holds 250.", question="", steps=[define, call])

        step = next(step for step in sources.steps if step.tool_use_id == "toolu_2")
        assert (step.kind, step.requests) == (SourceKind.web, ["GET https://erp.example.com/api{path}"])
        # The helper's cell follows the cell that called it, as one the data came through.
        assert [s.tool_use_id for s in _figure(sources, "250").sources] == ["toolu_2", "toolu_1"]

    def test_a_helper_that_runs_a_query_makes_the_calling_cell_a_query_with_its_sql(self):
        define = _cell(
            "toolu_1", code="def stock(site):\n    return query_minds_data('SELECT qty FROM stock')", stdout="",
        )
        call = _cell("toolu_2", code="print(stock('S1').qty.sum())", stdout="250\n")

        sources = find_figure_sources(answer="S1 holds 250.", question="", steps=[define, call])

        step = next(step for step in sources.steps if step.tool_use_id == "toolu_2")
        assert (step.kind, step.queries) == (SourceKind.query, ["SELECT qty FROM stock"])

    def test_a_helper_that_reads_a_file_names_the_cell_that_spelled_the_file(self, tmp_path: Path):
        (tmp_path / "stock.csv").write_text("qty\n250\n90\n")
        define = _cell("toolu_1", code="def load():\n    return pd.read_csv('stock.csv')", stdout="")
        call = _cell("toolu_2", code="print(load().qty.sum())", stdout="340\n")

        sources = find_figure_sources(
            answer="340 units.", question="", steps=[define, call], files=AgentFiles.on_this_server(tmp_path),
        )

        assert [s.tool_use_id for s in _figure(sources, "340").sources] == ["toolu_2", "toolu_1"]
        assert (sources.steps[0].kind, sources.steps[0].path) == (SourceKind.file, "stock.csv")

    def test_a_reset_forgets_the_helpers_too(self):
        define = _cell("toolu_1", code="def stock():\n    return query_minds_data('SELECT qty FROM stock')", stdout="")
        reset = _scratchpad("toolu_reset", action="reset", content="Scratchpad 'main' reset. All state cleared.")
        call = _cell("toolu_2", code="print(stock().qty.sum())", stdout="250\n")

        [step] = find_figure_sources(answer="S1 holds 250.", question="", steps=[define, reset, call]).steps

        assert (step.kind, step.queries) == (SourceKind.step, [])


class TestTheStepsAFigureWasCalculatedFrom:
    def test_a_figure_a_later_cell_computed_names_the_cell_that_read_its_data(self, tmp_path: Path):
        (tmp_path / "stock.csv").write_text("part,site,qty\nP100,S1,250\nP100,S2,90\n")
        load = _cell("toolu_a", code="import pandas as pd\ndf = pd.read_csv('stock.csv')", stdout="")
        total = _cell("toolu_b", code="print(df.qty.sum())", stdout="340\n")

        sources = find_figure_sources(
            answer="There are 340 units across S1 and S2.", question="", steps=[load, total],
            files=AgentFiles.on_this_server(tmp_path),
        )

        assert _figure(sources, "340").sources == [
            FigureSource(tool_use_id="toolu_b"), FigureSource(tool_use_id="toolu_a"),
        ]
        assert sources.steps == [
            FigureStep(tool_use_id="toolu_a", kind=SourceKind.file, label="stock.csv", path="stock.csv"),
            FigureStep(tool_use_id="toolu_b", kind=SourceKind.step, label=""),
        ]

    def test_a_sum_of_a_query_cells_frame_names_the_query_with_its_sql(self):
        query = _cell("toolu_a", code="df = query_minds_data('SELECT site, qty FROM stock')", stdout="")
        total = _cell("toolu_b", code="print(df.qty.sum())", stdout="340\n")

        sources = find_figure_sources(answer="340 units in all.", question="", steps=[query, total])

        assert [s.tool_use_id for s in _figure(sources, "340").sources] == ["toolu_b", "toolu_a"]
        assert (sources.steps[0].kind, sources.steps[0].queries) == (SourceKind.query, ["SELECT site, qty FROM stock"])

    def test_a_name_computed_from_read_data_carries_its_cell_on(self):
        load = _cell("toolu_a", code="df = pd.read_csv('stock.csv')", stdout="")
        derive = _cell("toolu_b", code="total = int(df.qty.sum())", stdout="")
        show = _cell("toolu_c", code="print(total)", stdout="340\n")

        sources = find_figure_sources(answer="340 units in all.", question="", steps=[load, derive, show])

        assert [s.tool_use_id for s in _figure(sources, "340").sources] == ["toolu_c", "toolu_a"]

    def test_a_name_rebound_without_read_data_names_no_earlier_cell(self):
        load = _cell("toolu_a", code="df = pd.read_csv('stock.csv')", stdout="")
        rebind = _cell("toolu_b", code="df = make_frame(rows)", stdout="")
        total = _cell("toolu_c", code="print(df.qty.sum())", stdout="340\n")

        sources = find_figure_sources(answer="340 units in all.", question="", steps=[load, rebind, total])

        assert [s.tool_use_id for s in _figure(sources, "340").sources] == ["toolu_c"]

    def test_names_another_scratchpad_bound_are_not_this_ones(self):
        load = _cell("toolu_a", code="df = pd.read_csv('stock.csv')", stdout="", pad="other")
        total = _cell("toolu_b", code="print(df.qty.sum())", stdout="340\n")

        sources = find_figure_sources(answer="340 units in all.", question="", steps=[load, total])

        assert [s.tool_use_id for s in _figure(sources, "340").sources] == ["toolu_b"]

    def test_a_reset_forgets_where_names_came_from(self):
        load = _cell("toolu_a", code="df = pd.read_csv('stock.csv')", stdout="")
        reset = _scratchpad("toolu_reset", action="reset", content="Scratchpad 'main' reset. All state cleared.")
        total = _cell("toolu_b", code="print(df.qty.sum())", stdout="340\n")

        sources = find_figure_sources(answer="340 units in all.", question="", steps=[load, reset, total])

        assert [s.tool_use_id for s in _figure(sources, "340").sources] == ["toolu_b"]

    def test_the_read_cells_follow_every_holder(self):
        load_a = _cell("toolu_a", code="a = pd.read_csv('a.csv')", stdout="")
        load_b = _cell("toolu_b", code="b = pd.read_csv('b.csv')", stdout="")
        total = _cell("toolu_c", code="print(a.qty.sum() + b.qty.sum())", stdout="340\n")

        sources = find_figure_sources(answer="340 units in all.", question="", steps=[load_a, load_b, total])

        assert [s.tool_use_id for s in _figure(sources, "340").sources] == ["toolu_c", "toolu_a", "toolu_b"]

    def test_the_cells_a_figure_was_calculated_from_never_push_out_a_holder(self):
        load = _cell("toolu_load", code="df = pd.read_csv('stock.csv')", stdout="")
        holders = [_cell(f"toolu_{i}", code="print(df.qty.sum())", stdout="340\n") for i in range(MAX_FIGURE_SOURCES)]

        sources = find_figure_sources(answer="340 units in all.", question="", steps=[load, *holders])

        assert [s.tool_use_id for s in _figure(sources, "340").sources] == [
            f"toolu_{i}" for i in range(MAX_FIGURE_SOURCES)
        ]
        assert "toolu_load" not in [step.tool_use_id for step in sources.steps]

    def test_a_cell_that_read_the_data_and_holds_the_figure_is_named_once(self):
        load = _cell("toolu_a", code="df = pd.read_csv('stock.csv')\nprint(df)", stdout="   qty\n0  340\n")
        total = _cell("toolu_b", code="print(df.qty.sum())", stdout="340\n")

        sources = find_figure_sources(answer="340 units in all.", question="", steps=[load, total])

        assert [s.tool_use_id for s in _figure(sources, "340").sources] == ["toolu_a", "toolu_b"]


class TestEarlierTurns:
    def test_an_earlier_turns_step_never_holds_a_figure(self):
        earlier = _earlier(_cell("toolu_old", code="print(open('stock.csv').read())", stdout="S1,250\nS2,90\n"))
        now = _cell("toolu_new", code="print(df.qty.max())", stdout="250\n")

        sources = find_figure_sources(answer="S1 holds 250 and S2 holds 90.", question="", steps=[earlier, now])

        assert [s.tool_use_id for s in _figure(sources, "250").sources] == ["toolu_new"]
        assert _figure(sources, "90").origin is FigureOrigin.none
        assert [step.tool_use_id for step in sources.steps] == ["toolu_new"]

    def test_an_earlier_turns_web_page_never_holds_a_figure(self):
        earlier = _earlier(_tool("toolu_old", "web_fetch", {"url": "https://example.com/rates"}, "Overtime: $42 an hour"))

        sources = find_figure_sources(answer="Overtime costs $42 an hour.", question="", steps=[earlier])

        assert _origins(sources) == [("$42", FigureOrigin.none)]
        assert sources.steps == []

    def test_a_figure_an_earlier_turn_printed_by_coincidence_has_no_source(self):
        """An earlier turn printed a stock table whose qty column happens to
        hold 30, 360 and 12. This turn runs no step and works out a cost."""
        quantities = [250, 90, 30, 410, 75, 12, 360, 88, 140, 55, 64, 300, 18, 220, 95, 47, 130, 66, 210, 150]
        table = "part,site,qty\n" + "\n".join(f"P{100 + i},S{1 + i % 3},{q}" for i, q in enumerate(quantities))
        earlier = _earlier(_cell("toolu_t1", code="print(open('stock.csv').read())", stdout=table))

        sources = find_figure_sources(
            answer="Overtime typically costs $30 an hour, so 12 hours cost $360.",
            question="What would 12 hours of overtime cost?", steps=[earlier],
        )

        assert _origins(sources) == [
            ("$30", FigureOrigin.none), ("12", FigureOrigin.question), ("$360", FigureOrigin.none),
        ]
        assert sources.steps == []

    @pytest.mark.parametrize("earlier_first", [True, False])
    def test_names_an_earlier_turn_bound_reach_this_turns_cells(self, earlier_first: bool):
        spell = _earlier(_cell("toolu_old", code='base_sql = "SELECT qty FROM stock"', stdout=""))
        run = _cell("toolu_new", code="""print(query_minds_data(base_sql + " WHERE site = 'S1'"))""", stdout="250\n")

        steps = [spell, run] if earlier_first else [run, spell]
        [step] = find_figure_sources(answer="S1 holds 250.", question="", steps=steps).steps

        assert (step.tool_use_id, step.queries) == ("toolu_new", ["SELECT qty FROM stock WHERE site = 'S1'"])

    def test_a_sum_of_a_frame_an_earlier_turn_read_names_that_turns_cell(self):
        load = _earlier(_cell("toolu_old", code="df = pd.read_csv('stock.csv')", stdout=""))
        total = _cell("toolu_new", code="print(df.qty.sum())", stdout="340\n")

        sources = find_figure_sources(answer="340 units in all.", question="", steps=[load, total])

        assert [s.tool_use_id for s in _figure(sources, "340").sources] == ["toolu_new", "toolu_old"]

    @pytest.mark.parametrize(("cells_after", "named"), [
        (MAX_EARLIER_CELLS - 1, ["toolu_new", "toolu_old"]),
        (MAX_EARLIER_CELLS, ["toolu_new"]),
    ])
    def test_only_the_latest_earlier_cells_are_read(self, cells_after: int, named: list[str]):
        load = _earlier(_cell("toolu_old", code="df = pd.read_csv('stock.csv')", stdout=""))
        after = [_earlier(_cell(f"toolu_{i}", code=f"x_{i} = {i}", stdout="")) for i in range(cells_after)]
        total = _cell("toolu_new", code="print(df.qty.sum())", stdout="340\n")

        sources = find_figure_sources(answer="340 units in all.", question="", steps=[load, *after, total])

        assert _figure(sources, "340").origin is FigureOrigin.step
        assert [s.tool_use_id for s in _figure(sources, "340").sources] == named


def _earlier_answer(*figures: tuple[str, FigureOrigin]) -> EarlierAnswer:
    """An earlier answer whose saved event lists these figures, a step
    figure with the step that sourced it."""
    return EarlierAnswer(figures=[
        Figure(
            start=0, end=len(text), text=text, origin=origin,
            sources=[FigureSource(tool_use_id="toolu_t1")] if origin is FigureOrigin.step else [],
        )
        for text, origin in figures
    ])


def _repeated(
    *, answer: str, earlier: list[EarlierAnswer], question: str = "", steps: list[TurnStep] | None = None,
) -> FigureSources:
    return mark_repeated_figures(
        sources=find_figure_sources(answer=answer, question=question, steps=steps or []), earlier_answers=earlier,
    )


class TestFiguresRepeatedFromEarlierAnswers:
    def test_a_figure_an_earlier_answer_sourced_is_marked_as_repeated(self):
        """Turn 1 read sales of 4,600 for 2023. Turn 2 reads 5,200 for 2024
        and repeats the 4,600 without reading it again."""
        turn_1 = find_figure_sources(
            answer="Sales were 4,600 in 2023.", question="What were sales in 2023?",
            steps=[_cell("toolu_t1", code="print(sales(2023))", stdout="4600\n")],
        )
        assert _origins(turn_1) == [("4,600", FigureOrigin.step), ("2023", FigureOrigin.question)]

        sources = _repeated(
            answer="5,200 in 2024, up from 4,600.", question="and 2024?",
            steps=[_cell("toolu_t2", code="print(sales(2024))", stdout="5200\n")],
            earlier=[EarlierAnswer(figures=turn_1.figures)],
        )

        assert _origins(sources) == [
            ("5,200", FigureOrigin.step), ("2024", FigureOrigin.question), ("4,600", FigureOrigin.earlier_answer),
        ]
        assert _figure(sources, "4,600").sources == []
        assert [step.tool_use_id for step in sources.steps] == ["toolu_t2"]

    @pytest.mark.parametrize(("earlier", "now"), [
        ("4,600", "$4,600"),
        ("$4,600", "4,600"),
        ("4,600", "4.6k"),
        ("4,600", "46%"),
        ("46", "46%"),
        ("$4,600", "$4.6K"),
        ("12h", "12d"),
        ("3x", "3"),
        ("40kg", "40lbs"),
        ("C$30", "$30"),
        ("€4,600", "$4,600"),
        ("-5%", "5%"),
        ("12.5%", "0.125"),
        ("12h", "12hrs"),
        ("40lb", "40lbs"),
        ("4,612", "about 4,600"),
    ])
    def test_a_repeated_number_in_another_form_is_not_marked(self, earlier: str, now: str):
        sources = _repeated(answer=f"It was {now}.", earlier=[_earlier_answer((earlier, FigureOrigin.step))])

        assert [origin for _, origin in _origins(sources)] == [FigureOrigin.none]

    @pytest.mark.parametrize(("earlier", "now"), [
        ("4,600", "4600"),
        ("4,600", "4,600.00"),
        ("4.6e3", "4,600"),
        ("12.5%", "12.5 percent"),
        ("3x", "3×"),
        ("4.6 thousand", "4.6K"),
        ("-5%", "−5%"),
    ])
    def test_the_same_figure_written_alike_is_marked(self, earlier: str, now: str):
        sources = _repeated(answer=f"It was {now}.", earlier=[_earlier_answer((earlier, FigureOrigin.step))])

        assert _origins(sources) == [(now, FigureOrigin.earlier_answer)]

    def test_a_figure_an_earlier_answer_left_unsourced_is_not_marked(self):
        sources = _repeated(answer="It was 4,600.", earlier=[_earlier_answer(("4,600", FigureOrigin.none))])

        assert _origins(sources) == [("4,600", FigureOrigin.none)]

    @pytest.mark.parametrize("origin", [FigureOrigin.question, FigureOrigin.earlier_answer])
    def test_an_earlier_figure_from_the_question_or_a_repeat_counts(self, origin: FigureOrigin):
        sources = _repeated(answer="It was 4,600.", earlier=[_earlier_answer(("4,600", origin))])

        assert _origins(sources) == [("4,600", FigureOrigin.earlier_answer)]

    def test_a_figure_this_turn_sources_keeps_its_own_origin(self):
        sources = _repeated(
            answer="4,600 units across 9 sites.",
            steps=[_cell("toolu_now", code="print(total)", stdout="4600\n")],
            earlier=[_earlier_answer(("4,600", FigureOrigin.step), ("9", FigureOrigin.step))],
        )

        assert _origins(sources) == [("4,600", FigureOrigin.step), ("9", FigureOrigin.earlier_answer)]
        assert _figure(sources, "4,600").sources == [FigureSource(tool_use_id="toolu_now")]

    def test_a_figure_the_question_states_stays_the_questions(self):
        sources = _repeated(
            answer="Yes, 4,600 across 9 sites.", question="Is it 4,600?",
            earlier=[_earlier_answer(("4,600", FigureOrigin.step), ("9", FigureOrigin.step))],
        )

        assert _origins(sources) == [("4,600", FigureOrigin.question), ("9", FigureOrigin.earlier_answer)]

    def test_an_earlier_figure_that_is_not_one_number_is_skipped(self):
        """Each earlier text holds no number, two numbers, or one number with
        other text around it."""
        sources = _repeated(
            answer="It was 4,600.",
            earlier=[_earlier_answer(
                ("1.2.3", FigureOrigin.step), ("4,600 and 5", FigureOrigin.step),
                ("4,600 units", FigureOrigin.step), ("about 4,600", FigureOrigin.step),
            )],
        )

        assert _origins(sources) == [("4,600", FigureOrigin.none)]

    def test_an_answer_with_no_figure_left_unsourced_is_returned_as_found(self):
        found = find_figure_sources(
            answer="S1 holds 250.", question="", steps=[_cell("toolu_now", code="print(qty)", stdout="250\n")],
        )

        assert mark_repeated_figures(
            sources=found, earlier_answers=[_earlier_answer(("250", FigureOrigin.step))],
        ) is found


class TestStepsWithNoId:
    """The client opens a step by its tool_use_id, so a step without one is
    never named."""

    def test_a_figure_only_a_step_with_no_id_holds_is_sourced_but_names_no_step(self):
        unnamed = replace(READ_STOCK, tool_use_id="")

        sources = find_figure_sources(answer="S1 holds 250 units.", question="", steps=[unnamed])

        assert _figure(sources, "250").origin is FigureOrigin.step
        assert _figure(sources, "250").sources == []
        assert sources.steps == []

    def test_a_step_with_no_id_takes_none_of_the_places_a_figure_names(self):
        steps = [
            _cell("", code="print(v)", stdout="250\n"),
            *(_cell(f"toolu_{i}", code="print(v)", stdout="250\n") for i in range(MAX_FIGURE_SOURCES)),
        ]

        sources = find_figure_sources(answer="S1 holds 250 units.", question="", steps=steps)

        expected = [f"toolu_{i}" for i in range(MAX_FIGURE_SOURCES)]
        assert [s.tool_use_id for s in _figure(sources, "250").sources] == expected
        assert [step.tool_use_id for step in sources.steps] == expected

    def test_an_earlier_turns_cell_with_no_id_is_not_named_as_the_one_that_read_the_data(self):
        load = _earlier(_cell("", code="df = pd.read_csv('stock.csv')", stdout=""))
        total = _cell("toolu_b", code="print(df.qty.sum())", stdout="340\n")

        sources = find_figure_sources(answer="340 units in all.", question="", steps=[load, total])

        assert _figure(sources, "340").sources == [FigureSource(tool_use_id="toolu_b")]
        assert [step.tool_use_id for step in sources.steps] == ["toolu_b"]

    def test_the_cells_a_step_with_no_id_read_from_are_not_named_either(self):
        load = _cell("toolu_load", code="df = pd.read_csv('stock.csv')", stdout="")
        total = _cell("", code="print(df.qty.sum())", stdout="340\n")

        sources = find_figure_sources(answer="340 units in all.", question="", steps=[load, total])

        assert _figure(sources, "340").origin is FigureOrigin.step
        assert _figure(sources, "340").sources == []
        assert sources.steps == []


class TestFilePaths:
    def test_a_file_the_project_holds_gives_the_step_its_path(self, tmp_path: Path):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "stock.csv").write_text("part,qty\nP100,250\n")
        step = _cell("toolu_read", code="print(pd.read_csv('data/stock.csv'))", stdout="P100 250\n")

        step_out = _only_step(step, answer="P100: 250 units.", files=AgentFiles.on_this_server(tmp_path))

        assert (step_out.kind, step_out.label, step_out.path) == (SourceKind.file, "stock.csv", "data/stock.csv")

    def test_a_file_missing_when_the_turn_ends_has_no_path(self, tmp_path: Path):
        step = _cell("toolu_read", code="print(pd.read_csv('data/stock.csv'))", stdout="P100 250\n")

        step_out = _only_step(step, answer="P100: 250 units.", files=AgentFiles.on_this_server(tmp_path))

        assert (step_out.label, step_out.path) == ("stock.csv", None)

    def test_a_turn_without_its_project_names_no_path(self, tmp_path: Path):
        (tmp_path / "stock.csv").write_text("P100,250\n")
        step = _cell("toolu_read", code=f"print(open({str(tmp_path / 'stock.csv')!r}).read())", stdout="P100,250\n")

        assert _only_step(step, answer="P100: 250 units.").path is None


    @pytest.mark.parametrize("save", [
        'df.to_csv("rates.csv")',
        'with open("rates.csv", "w") as f:\n    f.write(r.text)',
    ])
    def test_a_file_a_web_cell_saves_is_not_its_path(self, tmp_path: Path, save: str):
        (tmp_path / "rates.csv").write_text("rate\n1.37\n")
        code = (
            'r = requests.get("https://api.example.com/v1/rates?key=abc")\n'
            "df = pd.DataFrame(r.json())\n"
            f"{save}\n"
            "print(df.rate.mean())"
        )
        step = _cell("toolu_api", code=code, stdout="1.37\n")

        assert _only_step(step, answer="The mean rate is 1.37.", files=AgentFiles.on_this_server(tmp_path)) == FigureStep(
            tool_use_id="toolu_api", kind=SourceKind.web, label="api.example.com/v1/rates",
            url="https://api.example.com/v1/rates", path=None, requests=["GET https://api.example.com/v1/rates"],
        )

    def test_a_cell_that_only_writes_a_file_is_a_step_without_a_path(self, tmp_path: Path):
        (tmp_path / "out.csv").write_text("n\n42\n")
        step = _cell("toolu_w", code="df = pd.DataFrame(rows)\ndf.to_csv('out.csv')\nprint(len(df))", stdout="42\n")

        step_out = _only_step(step, answer="There are 42 rows.", files=AgentFiles.on_this_server(tmp_path))

        assert (step_out.kind, step_out.label, step_out.path) == (SourceKind.step, "", None)

    @pytest.mark.parametrize("save", [
        "wb = openpyxl.Workbook()\nwb.active.append([total])\nwb.save('summary.xlsx')",
        "df.to_excel('summary.xlsx')",
        "with pd.ExcelWriter('summary.xlsx') as writer:\n    df.to_excel(writer)",
        "np.savetxt('summary.csv', [total])",
        "df.to_excel(os.path.join(out_dir, 'summary.xlsx'))",
    ])
    def test_a_cell_that_saves_a_workbook_of_its_result_stays_a_step(self, tmp_path: Path, save: str):
        (tmp_path / "summary.xlsx").write_text("340")
        (tmp_path / "summary.csv").write_text("340")
        code = f"total = int(df.qty.sum())\n{save}\nprint(total)"
        step = _cell("toolu_w", code=code, stdout="340\n")

        step_out = _only_step(step, answer="340 units in all.", files=AgentFiles.on_this_server(tmp_path))

        assert (step_out.kind, step_out.label, step_out.path) == (SourceKind.step, "", None)

    def test_a_calculation_that_saves_its_result_is_not_from_that_file(self, tmp_path: Path):
        (tmp_path / "notes.md").write_text("9002")
        step = _cell(
            "toolu_c", code='total = a * b; open("notes.md", "w").write(str(total)); print(total)', stdout="9002\n",
        )

        step_out = _only_step(step, answer="The total is 9002.", files=AgentFiles.on_this_server(tmp_path))

        assert (step_out.kind, step_out.label, step_out.path) == (SourceKind.step, "", None)

    def test_a_document_name_the_cell_only_prints_is_not_a_document_it_read(self, tmp_path: Path):
        (tmp_path / "rates.csv").write_text("rate\n1.37\n")
        code = (
            'r = requests.get("https://api.example.com/v1/rates")\n'
            "print('Saved to rates.csv')\n"
            "print(r.json()['rate'])"
        )
        step = _cell("toolu_api", code=code, stdout="Saved to rates.csv\n1.37\n")

        step_out = _only_step(step, answer="The rate is 1.37.", files=AgentFiles.on_this_server(tmp_path))

        assert (step_out.kind, step_out.label, step_out.path) == (SourceKind.web, "api.example.com/v1/rates", None)

    def test_a_document_the_cell_reads_and_prints_is_still_read(self, tmp_path: Path):
        (tmp_path / "stock.csv").write_text("qty\n250\n")
        code = "df = pd.read_csv('stock.csv')\nprint('stock.csv')\nprint(df.qty.sum())"
        step = _cell("toolu_r", code=code, stdout="stock.csv\n250\n")

        step_out = _only_step(step, answer="S1 holds 250.", files=AgentFiles.on_this_server(tmp_path))

        assert (step_out.kind, step_out.label, step_out.path) == (SourceKind.file, "stock.csv", "stock.csv")

    def test_the_event_names_the_project_the_turn_ran_in(self, tmp_path: Path):
        project_id = uuid4()

        sources = find_figure_sources(
            answer="S1 holds 250.", question="", steps=[READ_STOCK],
            files=AgentFiles.on_this_server(tmp_path, project_id=project_id),
        )

        assert sources.project_id == str(project_id)
        assert find_figure_sources(answer="S1 holds 250.", question="", steps=[READ_STOCK]).project_id is None

    def test_a_file_attached_to_the_conversation_is_named_by_its_id_and_its_name(self, tmp_path: Path):
        file_id = str(uuid4())
        stored = tmp_path / "files" / file_id / "stock.csv"
        stored.parent.mkdir(parents=True)
        stored.write_text("part,qty\nP100,250\n")
        project = tmp_path / "project"
        project.mkdir()
        step = _cell("toolu_att", code=f"print(open({str(stored)!r}).read())", stdout="part,qty\nP100,250\n")
        files = AgentFiles.on_this_server(project, attachments=[Attachment(file_id=file_id, stored_path=str(stored))])

        assert _only_step(step, answer="P100: 250 units.", files=files) == FigureStep(
            tool_use_id="toolu_att", kind=SourceKind.file, label="stock.csv", attachment_id=file_id,
        )

    def test_a_query_whose_result_is_saved_has_no_path(self, tmp_path: Path):
        (tmp_path / "summary.csv").write_text("qty\n250\n")
        step = _cell(
            "toolu_q",
            code='df = query_minds_data("SELECT qty FROM stock"); df.to_csv("summary.csv"); print(df.qty.sum())',
            stdout="250\n",
        )

        step_out = _only_step(step, answer="S1 holds 250.", files=AgentFiles.on_this_server(tmp_path))

        assert (step_out.kind, step_out.path) == (SourceKind.query, None)

    def test_a_file_read_then_another_written_keeps_the_one_read(self, tmp_path: Path):
        (tmp_path / "stock.csv").write_text("qty\n250\n")
        (tmp_path / "out.csv").write_text("qty\n250\n")
        step = _cell(
            "toolu_r",
            code='df = pd.read_csv("stock.csv"); df.to_csv("out.csv"); print(df.qty.sum())',
            stdout="250\n",
        )

        step_out = _only_step(step, answer="S1 holds 250.", files=AgentFiles.on_this_server(tmp_path))

        assert (step_out.kind, step_out.label, step_out.path) == (SourceKind.file, "stock.csv", "stock.csv")


class TestSeveralDocuments:
    @pytest.fixture
    def looked_up(self, monkeypatch) -> list[str]:
        """The literals AgentFiles.named_file is asked about."""
        literals: list[str] = []
        named_file = AgentFiles.named_file

        def spy(self: AgentFiles, literal: str) -> NamedFile | None:
            literals.append(literal)
            return named_file(self, literal)

        monkeypatch.setattr(AgentFiles, "named_file", spy)
        return literals

    @pytest.mark.parametrize(("first", "second"), [("lines.csv", "rates.pdf"), ("rates.pdf", "lines.csv")])
    def test_a_cell_that_reads_two_documents_names_both_with_no_path_or_page(
        self, tmp_path: Path, looked_up: list[str], first: str, second: str,
    ):
        for name in (first, second):
            (tmp_path / name).write_text("x")
        code = f"a = read_document({first!r})\nb = read_document({second!r})\nprint(a, b)"
        step = _cell("toolu_two", code=code, stdout="--- page 1 ---\nOvertime on Line 7 costs $42 an hour.\n")

        sources = find_figure_sources(
            answer="Overtime costs $42 an hour.", question="", steps=[step], files=AgentFiles.on_this_server(tmp_path),
        )

        assert sources.steps == [
            FigureStep(tool_use_id="toolu_two", kind=SourceKind.file, label=f"{first}, {second}"),
        ]
        assert _figure(sources, "$42").sources == [FigureSource(tool_use_id="toolu_two", page=None)]
        assert looked_up == []

    def test_one_document_is_looked_up_once(self, tmp_path: Path, looked_up: list[str]):
        (tmp_path / "stock.csv").write_text("qty\n250\n")
        step = _cell("toolu_r", code="print(pd.read_csv('stock.csv').qty.sum())", stdout="250\n")

        step_out = _only_step(step, answer="S1 holds 250.", files=AgentFiles.on_this_server(tmp_path))

        assert (step_out.path, looked_up) == ("stock.csv", ["stock.csv"])

    def test_past_the_first_few_documents_the_label_counts_the_rest(self, tmp_path: Path, looked_up: list[str]):
        names = [f"d/f{i}.csv" for i in range(MAX_LABELED_DOCUMENTS + 2)]
        step = _cell(
            "toolu_many", code=f"for name in {names!r}:\n    print(pd.read_csv(name).qty.sum())", stdout="250\n",
        )

        step_out = _only_step(step, answer="S1 holds 250.", files=AgentFiles.on_this_server(tmp_path))

        shown = ", ".join(f"f{i}.csv" for i in range(MAX_LABELED_DOCUMENTS))
        assert (step_out.label, step_out.path) == (f"{shown} and 2 more", None)
        assert looked_up == []


class TestRedaction:
    def test_a_user_and_password_after_a_placeholder_scheme_are_redacted(self):
        code = (
            'r = requests.get(f"{proto}://admin:s3cret@erp.internal/api/stock")\n'
            "a = con.execute(f\"SELECT qty FROM postgres_scan('{proto}://erp:s3cret@db.internal/erp', 'public', 'stock')\")\n"
            "print(r.json()['qty'])"
        )

        step = _only_step(_cell("toolu_p", code=code, stdout="250\n"), answer="S1 holds 250.")

        assert step.requests == ["GET {proto}://erp.internal/api/stock"]
        assert step.queries == ["SELECT qty FROM postgres_scan('{proto}://***@db.internal/erp', 'public', 'stock')"]
        assert "s3cret" not in step.model_dump_json()

    def test_credentials_in_queries_requests_and_urls_are_redacted(self):
        code = """
con.execute("ATTACH 'dbname=erp user=erp password=hunter2' AS pg (TYPE postgres)")
a = con.execute("SELECT qty FROM postgres_scan('postgresql://erp:s3cr3t@db.example.com/erp', 'public', 'stock')")
b = con.execute("WITH s AS (SELECT * FROM mysql_scan('host=db user=erp password=pa55word', 'stock')) SELECT qty FROM s")
c = con.execute("SELECT qty FROM read_csv('https://data.example.com/stock.csv?api_key=abc123&region=eu')")
r = requests.get("https://erp:t0ken@api.example.com/v1/stock?api_key=abc123")
print(a.fetchone()[0])
"""
        step = _only_step(_cell("toolu_x", code=code, stdout="250\n"), answer="S1 holds 250.")

        assert step.queries == [
            "SELECT qty FROM postgres_scan('postgresql://***@db.example.com/erp', 'public', 'stock')",
            "WITH s AS (SELECT * FROM mysql_scan('host=db user=erp password=[redacted]', 'stock')) SELECT qty FROM s",
            "SELECT qty FROM read_csv('https://data.example.com/stock.csv?...')",
        ]
        assert step.requests == ["GET https://api.example.com/v1/stock"]
        event = step.model_dump_json()
        for secret in ("hunter2", "s3cr3t", "pa55word", "abc123", "t0ken"):
            assert secret not in event

    def test_a_known_key_shape_in_a_query_is_scrubbed(self):
        key = "sk-" + "a" * 30
        step = _cell("toolu_k", code=f"cur.execute(\"SELECT qty FROM api_scan('{key}')\")\nprint(cur.fetchone()[0])", stdout="250\n")

        assert _only_step(step, answer="S1 holds 250.").queries == ["SELECT qty FROM api_scan('[REDACTED_API_KEY]')"]

    def test_a_long_query_is_cut_to_the_limit(self):
        long_query = "SELECT " + ", ".join(f"col_{i}" for i in range(1_000)) + " FROM wide"
        step = _cell("toolu_w", code=f"cur.execute({long_query!r})\nprint(cur.fetchone()[0])", stdout="250\n")

        [query] = _only_step(step, answer="S1 holds 250.").queries

        assert len(query) == MAX_QUERY_CHARS
        assert query.endswith("…")

    def test_a_long_request_is_cut_to_the_limit(self):
        path = "/".join(f"segment{i}" for i in range(200))
        step = _cell("toolu_r", code=f"r = requests.get('https://example.com/{path}')\nprint(r.json()['qty'])", stdout="250\n")

        [request] = _only_step(step, answer="S1 holds 250.").requests

        assert len(request) == MAX_REQUEST_CHARS
        assert request.startswith("GET https://example.com/segment0/")
        assert request.endswith("…")


def _event(step: TurnStep, *, answer: str = "There are 42 rows.") -> str:
    """The whole figure_sources event a step's figure gives, as sent."""
    return find_figure_sources(answer=answer, question="", steps=[step]).model_dump_json()


class TestSecretsInUrls:
    @pytest.mark.parametrize(("url", "secret", "masked"), [
        (
            "https://hooks.slack.com/services/T0001/B0002/XyZsecretWebhook",
            "XyZsecretWebhook", "https://hooks.slack.com/***",
        ),
        (
            "https://discord.com/api/webhooks/123456/XyZsecretWebhook",
            "XyZsecretWebhook", "https://discord.com/api/webhooks/***",
        ),
        (
            "https://contoso.webhook.office.com/webhookb2/abc@def/IncomingWebhook/XyZsecretWebhook/ghi",
            "XyZsecretWebhook", "https://contoso.webhook.office.com/***",
        ),
        (
            "https://api.telegram.org/bot123456:XyZsecretBotToken/sendMessage",
            "XyZsecretBotToken", "https://api.telegram.org/bot***",
        ),
        (
            "https://hooks.slack-gov.com/services/T0001/B0002/XyZsecretWebhook",
            "XyZsecretWebhook", "https://hooks.slack-gov.com/***",
        ),
        (
            "https://hooks.zapier.com/hooks/catch/123456/XyZsecretHook/",
            "XyZsecretHook", "https://hooks.zapier.com/***",
        ),
        (
            "https://maker.ifttt.com/trigger/stock/with/key/XyZsecretIftttKey",
            "XyZsecretIftttKey", "https://maker.ifttt.com/***",
        ),
    ])
    def test_a_url_whose_path_is_a_credential_is_masked_and_not_linked(self, url: str, secret: str, masked: str):
        step = _cell("toolu_hook", code=f"r = requests.post({url!r}, json=rows)\nprint(len(rows))", stdout="42\n")

        event = _event(step)

        [step_out] = FigureSources.model_validate_json(event).steps
        assert (step_out.kind, step_out.url, step_out.requests) == (SourceKind.web, None, [f"POST {masked}"])
        assert secret not in event
        assert "123456:" not in event

    def test_a_webhook_bound_to_a_name_is_masked_too(self):
        code = "HOOK = 'https://hooks.slack.com/services/T0001/B0002/XyZsecretWebhook'\nrequests.post(HOOK, json=rows)\nprint(len(rows))"

        event = _event(_cell("toolu_hook", code=code, stdout="42\n"))

        assert "XyZsecretWebhook" not in event
        assert "T0001" not in event

    def test_a_token_joined_into_a_url_path_reads_as_its_name(self):
        code = (
            "TOKEN = 'tok-XyZsecret123'\n"
            "r = requests.get('https://erp.example.com/api/' + TOKEN + '/orders')\n"
            "print(len(r.json()))"
        )

        event = _event(_cell("toolu_t", code=code, stdout="42\n"))

        [step_out] = FigureSources.model_validate_json(event).steps
        assert step_out.requests == ["GET https://erp.example.com/api/{TOKEN}/orders"]
        # A URL with a name in place of a part of it is not a page to open.
        assert step_out.url is None
        assert "XyZsecret123" not in event

    @pytest.mark.parametrize("password", ["Xk9#2mQ", "Xk9?2mQ", "Xk9/2mQ"])
    def test_a_url_password_with_a_delimiter_in_it_is_redacted(self, password: str):
        fetch = _cell(
            "toolu_web", code=f"r = requests.get('https://erp:{password}@example.com/rates')\nprint(len(r.json()))",
            stdout="42\n",
        )
        sql = f"SELECT qty FROM postgres_scan('postgresql://erp:{password}@db.internal/erp', 'public', 'stock')"
        query = _cell("toolu_sql", code=f"a = con.execute({sql!r})\nprint(a.fetchone()[0])", stdout="250\n")

        event = find_figure_sources(
            answer="42 rates; S1 holds 250.", question="", steps=[fetch, query],
        ).model_dump_json()

        web_step, query_step = FigureSources.model_validate_json(event).steps
        assert web_step.requests == ["GET https://example.com/rates"]
        assert (web_step.label, web_step.url) == ("example.com/rates", "https://example.com/rates")
        assert query_step.queries == [
            "SELECT qty FROM postgres_scan('postgresql://***@db.internal/erp', 'public', 'stock')",
        ]
        assert "Xk9" not in event
        assert "erp:" not in event

    def test_a_path_with_an_at_sign_is_still_a_link(self):
        step = _tool("toolu_f", "web_fetch", {"url": "https://www.youtube.com/@mindsdb/videos"}, "42 videos")

        step_out = _only_step(step, answer="There are 42 videos.")

        assert (step_out.url, step_out.label) == ("https://www.youtube.com/@mindsdb/videos", "www.youtube.com/@mindsdb/videos")


class TestSecretsInLabels:
    @pytest.mark.parametrize("value", [
        "private_key='-----BEGIN PRIVATE KEY-----\nMIIXyZsecret\n-----END PRIVATE KEY-----'",
        "password='firstSecret\nsecondSecret'",
        'password="firstSecret\r\nsecondSecret"',
        "password='firstSecret\\\nsecondSecret'",
        "password='firstSecret\nsecondSecret",
    ])
    @pytest.mark.parametrize("where", ["label", "query"])
    def test_a_multiline_named_secret_is_removed_from_the_whole_event(self, value: str, where: str):
        sql = f"SELECT qty FROM read_source({value})" if where == "query" else "SELECT qty FROM stock"
        step = _cell(
            "toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n",
            description=value if where == "label" else "Read stock",
        )

        event = _event(step)

        assert "MIIXyZsecret" not in event
        assert "firstSecret" not in event
        assert "secondSecret" not in event
        assert FigureSources.model_validate_json(event).steps

    def test_a_description_that_holds_a_secret_is_redacted_in_the_label(self):
        step = _cell(
            "toolu_q", code="print(query_minds_data('SELECT qty FROM stock'))", stdout="42\n",
            description="Query stock with password=hunter2 and token: tok_XyZ",
        )

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].label == (
            "Query stock with password=[redacted] and token: [redacted]"
        )
        assert "hunter2" not in event
        assert "tok_XyZ" not in event

    @pytest.mark.parametrize(("url", "label"), [
        ("https://bob:" + "p" * 1_100 + "XyZs3cr3t@example.com/page", "example.com/page"),
        ("https://bob:p''" + "p" * 1_100 + "XyZs3cr3t@example.com/page", "example.com/page"),
        ("https://" + "u" * 300 + "XyZs3cr3t:pw@example.com/page", "example.com/page"),
        ("https://" + "u" * 300 + "XyZs3cr3t@example.com/page", "example.com/page"),
        ("https://bob:" + "p" * 1_100 + "/XyZs3cr3t/XyZb@example.com/page", "example.com/page"),
        ("https://XyZbob:" + "p" * 1_100 + "?XyZs3cr3t@example.com/page", "https://***"),
        ("https://" + "u" * 300 + "XyZuser:p/XyZs3cr3t@example.com/page", "example.com/page"),
    ], ids=[
        "long-password", "long-password-with-a-doubled-quote", "long-user", "long-user-alone",
        "long-password-with-a-slash", "long-password-with-a-question-mark", "long-user-with-a-slash-in-the-password",
    ])
    def test_a_long_user_or_password_is_left_out_of_a_web_steps_label(self, url: str, label: str):
        """The label is the host and path, read without the user and password
        of any length. A /, ? or # inside a long password ends urlsplit's host
        there, so the label is built from what follows the last @, or is the
        masked URL when the query string's cut took the @."""
        step = _tool("toolu_f", "web_fetch", {"url": url}, "It sold 4,812 units.")

        event = _event(step, answer="It sold 4,812 units.")

        [step_out] = FigureSources.model_validate_json(event).steps
        assert (step_out.label, step_out.url) == (label, None)
        assert "XyZ" not in event

    @pytest.mark.parametrize(("url", "label"), [
        ("https://example.com:8443/rates?k=1", "example.com:8443/rates"),
        ("https://bob:pw@[::1]:8443/rates", "[::1]:8443/rates"),
    ], ids=["port", "ipv6-host"])
    def test_a_web_steps_label_keeps_the_hosts_port(self, url: str, label: str):
        step = _tool("toolu_f", "web_fetch", {"url": url}, "Rate: 42")

        [step_out] = FigureSources.model_validate_json(_event(step, answer="The rate is 42.")).steps

        assert step_out.label == label

    def test_a_secret_in_a_web_steps_label_and_url_is_never_sent(self):
        step = _tool(
            "toolu_f", "web_fetch", {"url": "https://example.com/api/token=XyZsecret/rates"}, "Rate: 42",
        )

        event = _event(step, answer="The rate is 42.")

        [step_out] = FigureSources.model_validate_json(event).steps
        assert step_out.url is None
        assert "XyZsecret" not in event


class TestSecretsInQueries:
    @pytest.mark.parametrize("sql", [
        "SELECT 1--0; CREATE USER user_a IDENTIFIED BY 'XyZsecret'",
        "SELECT 1 # '\n; CREATE USER user_a IDENTIFIED BY 'XyZsecret'",
        "SELECT 1 /*!; CREATE USER user_a IDENTIFIED BY 'XyZsecret' */",
        "SELECT qty FROM stock AS $tag$; CREATE USER user_a IDENTIFIED BY 'XyZsecret'; SELECT qty FROM stock AS $tag$",
    ])
    def test_dialect_ambiguous_batches_never_expose_trailing_ddl(self, sql: str):
        step = _cell("sql", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert "XyZsecret" not in event
        assert "CREATE USER" not in event
        assert all(not step.queries for step in FigureSources.model_validate_json(event).steps)

    @pytest.mark.parametrize("quote", ["'", '"'])
    @pytest.mark.parametrize("prefix_length", [2, 1_100])
    @pytest.mark.parametrize("repeats", [1, 150])
    @pytest.mark.parametrize("where", ["label", "query"])
    def test_uri_with_the_opposite_quote_is_redacted_without_a_registered_secret(
        self, quote: str, prefix_length: int, repeats: int, where: str,
    ):
        opposite = '"' if quote == "'" else "'"
        uri = "postgresql://admin:" + "p" * prefix_length + opposite + "XyZsecret" * repeats + "@db.example/stock"
        sql = f"SELECT qty FROM stock WHERE source_uri = {quote}{uri}{quote}"
        step = _cell(
            "toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n",
            description=uri if where == "label" else "Read stock",
        )

        event = _event(step)

        assert "XyZsecret" not in event
        assert "admin:" not in event
        assert FigureSources.model_validate_json(event).steps

    @pytest.mark.parametrize("prefix_length", [2, 1_100])
    @pytest.mark.parametrize("quote", ["'", '"'])
    @pytest.mark.parametrize("escaping", ["doubled", "backslash"])
    @pytest.mark.parametrize("where", ["label", "query"])
    def test_long_sql_quoted_uri_userinfo_is_redacted_before_the_event_is_capped(
        self, prefix_length: int, quote: str, escaping: str, where: str, monkeypatch: pytest.MonkeyPatch,
    ):
        password = "p" * prefix_length + quote + "XyZsecret" * 150
        monkeypatch.setenv("DS_FIGURE_TEST_PASSWORD", password)
        escaped_password = password.replace(quote, quote * 2 if escaping == "doubled" else "\\" + quote)
        uri = f"postgresql://admin:{escaped_password}@db.example.com/stock"
        sql = f"SELECT qty FROM postgres_scan({quote}{uri}{quote}, 'public', 'stock')"
        step = _cell(
            "toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n",
            description=uri if where == "label" else "Read stock",
        )

        event = _event(step)

        assert "XyZsecret" not in event
        assert "admin:" not in event
        assert FigureSources.model_validate_json(event).steps

    @pytest.mark.parametrize(("literal", "secret"), [
        ("Server=db;Uid=erp;Pwd=XyZpass9;", "XyZpass9"),
        ("s3_secret_access_key=XyZsecretKey9", "XyZsecretKey9"),
        ("{'Authorization': 'Bearer XyZbearer9'}", "XyZbearer9"),
        ("key_id=AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
        ("session=ASIAIOSFODNN7EXAMPLE", "ASIAIOSFODNN7EXAMPLE"),
        ("gh=ghp_" + "a1B2" * 9, "a1B2a1B2a1B2"),
        ("slack=xoxb-1234567890-0987654321-XyZslack", "XyZslack"),
        ("jwt=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.XyZsignature", "XyZsignature"),
        ("-----BEGIN RSA PRIVATE KEY-----\nMIIXyZpem\n-----END RSA PRIVATE KEY-----", "MIIXyZpem"),
        (
            "https://bucket.s3.amazonaws.com/stock.csv?X-Amz-Credential=AKIA%2F&X-Amz-Signature=XyZsig9",
            "XyZsig9",
        ),
        ("https://acct.blob.core.windows.net/c/stock.csv?sv=2024&sig=XyZsas9", "XyZsas9"),
        ("Authorization: Basic dXNlcjpzM2NyZXQ=", "dXNlcjpzM2NyZXQ="),
        ("Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b", "9944b09199c62bcf"),
        ("password => XyZhunter22", "XyZhunter22"),
        ("scott/XyZtiger5@db.example.com:1521/orcl", "XyZtiger5"),
        ("auth=sk_live_51HXyZabcdefghijklmnop", "XyZabcdefghijklmnop"),
        ("tok=hf_AbCdEfGhIjKlMnOpQrStUvWxYz12", "AbCdEfGhIjKlMnOpQrStUvWxYz12"),
        ("at=ya29.a0AfH6SMBxXyZsecretToken123456789", "XyZsecretToken"),
        ("k=xapp-1-A0123-4567-abcdef0123456789", "abcdef0123456789"),
        ("host=db user=erp pass=XyZsup3rS3cret db=prod", "XyZsup3rS3cret"),
        (
            "DefaultEndpointsProtocol=https;AccountName=acct;AccountKey=XyZm9vYmFyU2VjcmV0==;EndpointSuffix=core.windows.net",
            "XyZm9vYmFyU2VjcmV0",
        ),
        ("pat=github_pat_11ABCDEFG0123456789_abcdefghijXyZ", "abcdefghijXyZ"),
        ("gh=gho_" + "XyZ1" * 9, "XyZ1XyZ1XyZ1"),
        ("rk=rk_test_51HXyZabcdefghij", "XyZabcdefghij"),
        ("slack=xoxp-1234567890-XyZslackuser", "XyZslackuser"),
        ("Bearer XyZabcdefghijk", "XyZabcdefghijk"),
        ("Driver={ODBC Driver 18};Server=db;Uid=erp;Pwd={ab;XyZpass9};", "XyZpass9"),
    ])
    def test_a_credential_shape_in_sql_is_never_sent(self, literal: str, secret: str):
        sql = f"SELECT qty FROM read_source('{literal}')"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        [step_out] = FigureSources.model_validate_json(event).steps
        assert step_out.queries[0].startswith("SELECT qty FROM read_source('")
        assert step_out.queries[0].endswith("')")
        assert secret not in event

    @pytest.mark.parametrize("form", ["{key}='XyZv9alue'", '"{key}": "XyZv9alue"', "{key}=XyZv9alue"])
    @pytest.mark.parametrize("key", [
        "password", "passwd", "passphrase", "pwd", "secret", "token", "api_key", "apikey", "X-Api-Key", "access_key",
        "account_key", "private_key", "authorization", "credential", "cookie", "pass", "db_pass", "signature",
        "X-Amz-Signature", "sig", "x_sig",
    ])
    def test_each_listed_key_word_is_redacted(self, key: str, form: str):
        sql = f"SELECT qty FROM read_source({form.format(key=key)})"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        assert "XyZv9alue" not in _event(step)

    @pytest.mark.parametrize("key", ["sigma", "signal", "design", "sig_level", "pass_rate", "passenger"])
    def test_a_key_holding_pass_or_sig_but_not_as_its_last_part_is_not_a_secret_key(self, key: str):
        """pass and sig name a secret only as a key's last part, so these
        columns keep their values."""
        sql = f"SELECT qty FROM stock WHERE {key}=0.5 AND {key}: 0.5"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        assert FigureSources.model_validate_json(_event(step)).steps[0].queries == [sql]

    @pytest.mark.parametrize("length", [1_025, 5_000])
    def test_a_url_password_longer_than_the_userinfo_window_is_masked(self, length: int):
        sql = f"SELECT qty FROM postgres_scan('postgres://u:{'p' * length}@db.example/x', 'public', 'stock')"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert "ppppp" not in event
        assert "u:" not in event

    def test_a_percent_encoded_token_as_a_url_password_is_masked(self):
        """An RDS IAM auth token, percent-encoded as a database URL needs it."""
        token = quote(
            "db.example.rds.amazonaws.com:5432/?Action=connect&DBUser=u&X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Date=20261015T000000Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host"
            "&X-Amz-Security-Token=" + "IQoJb3JpZ2luX2VjE" * 60 + "&X-Amz-Signature=XyZsignature0123456789",
            safe="",
        )
        sql = f"SELECT qty FROM postgres_scan('postgresql://u:{token}@db.example.rds.amazonaws.com:5432/db', 'public', 'stock')"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        for secret in ("X-Amz-Security-Token", "IQoJb3JpZ2luX2VjE", "X-Amz-Signature", "XyZsignature"):
            assert secret not in event

    @pytest.mark.parametrize("call", [
        "print(query_minds_data({sql!r}).qty.sum())", "cur.execute({sql!r})\nprint(cur.fetchone()[0])",
    ], ids=["query_minds_data", "cursor"])
    @pytest.mark.parametrize(("sql", "sent"), [
        ("SELECT qty FROM stock; CREATE USER app IDENTIFIED BY 'XyZhunter2'", ["SELECT qty FROM stock;"]),
        ("SELECT qty FROM stock; ALTER USER root WITH PASSWORD 'XyZhunter2'", ["SELECT qty FROM stock;"]),
        ("SELECT qty FROM stock; SET PASSWORD FOR app = 'XyZhunter2'", ["SELECT qty FROM stock;"]),
        # Postgres and SQLite read 'C:\' as a closed literal, MySQL as an
        # open one, so where the query ends depends on the database.
        ("SELECT qty FROM stock WHERE p = 'C:\\'; ALTER USER bob WITH PASSWORD 'XyZhunter2'", []),
    ], ids=["create-user", "alter-user", "set-password", "after-a-backslash"])
    def test_a_statement_after_a_query_is_not_sent(self, sql: str, sent: list[str], call: str):
        step = _cell("toolu_k", code=call.format(sql=sql), stdout="42\n")

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].queries == sent
        assert "XyZhunter2" not in event

    @pytest.mark.parametrize(("sql", "listed"), [
        (
            "SELECT q'[a']' FROM dual; CREATE USER bob IDENTIFIED BY XyZs3cr3t; SELECT 'b' FROM dual",
            "SELECT q'[a']' FROM dual; CREATE USER bob IDENTIFIED BY [redacted]; SELECT 'b' FROM dual",
        ),
        (
            "SELECT a FROM t WHERE b = `; CREATE USER bob PASSWORD 'XyZs3cr3t'; --`",
            "SELECT a FROM t WHERE b = `; CREATE USER bob PASSWORD '[redacted]'; --`",
        ),
        (
            "SELECT 'ALTER USER bob IDENTIFIED BY XyZs3cr3t' AS ddl FROM dual",
            "SELECT 'ALTER USER bob IDENTIFIED BY [redacted]' AS ddl FROM dual",
        ),
        (
            'SELECT \'CREATE USER "bob" IDENTIFIED WITH mysql_native_password BY "XyZs3cr3t"\' AS ddl FROM t',
            'SELECT \'CREATE USER "bob" IDENTIFIED WITH mysql_native_password BY "[redacted]"\' AS ddl FROM t',
        ),
        (
            'SELECT \'CREATE ROLE bob LOGIN ENCRYPTED PASSWORD "XyZs3cr3t"\' AS ddl FROM t',
            'SELECT \'CREATE ROLE bob LOGIN ENCRYPTED PASSWORD "[redacted]"\' AS ddl FROM t',
        ),
        (
            'SELECT \'CREATE SECRET (TYPE S3, KEY_ID "AKIAEXAMPLE", SECRET "XyZs3cr3t")\' AS ddl FROM t',
            'SELECT \'CREATE SECRET (TYPE S3, KEY_ID "AKIAEXAMPLE", SECRET "[redacted]")\' AS ddl FROM t',
        ),
        (
            'SELECT \'SET PASSWORD FOR "app"@"%" = "XyZs3cr3t"\' AS ddl FROM t',
            'SELECT \'SET PASSWORD FOR "app"@"%" = "[redacted]"\' AS ddl FROM t',
        ),
        (
            'SELECT \'CREATE USER app IDENTIFIED BY PASSWORD "*XyZs3cr3t"\' AS ddl FROM t',
            'SELECT \'CREATE USER app IDENTIFIED BY PASSWORD "[redacted]"\' AS ddl FROM t',
        ),
        (
            'SELECT "CREATE USER \'bob\'@\'%\' IDENTIFIED WITH \'mysql_native_password\' BY \'XyZs3cr3t\'" AS ddl FROM t',
            'SELECT "CREATE USER \'bob\'@\'%\' IDENTIFIED WITH \'mysql_native_password\' BY \'[redacted]\'" AS ddl FROM t',
        ),
        (
            'SELECT \'CREATE USER "bob" IDENTIFIED WITH "mysql_native_password" BY "XyZs3cr3t"\' AS ddl FROM t',
            'SELECT \'CREATE USER "bob" IDENTIFIED WITH "mysql_native_password" BY "[redacted]"\' AS ddl FROM t',
        ),
        (
            'SELECT \'CREATE USER "bob" IDENTIFIED WITH `caching_sha2_password` BY "XyZs3cr3t"\' AS ddl FROM t',
            'SELECT \'CREATE USER "bob" IDENTIFIED WITH `caching_sha2_password` BY "[redacted]"\' AS ddl FROM t',
        ),
        (
            'SELECT \'CREATE USER "bob" IDENTIFIED WITH caching_sha2_password AS "XyZs3cr3t"\' AS ddl FROM t',
            'SELECT \'CREATE USER "bob" IDENTIFIED WITH caching_sha2_password AS "[redacted]"\' AS ddl FROM t',
        ),
        (
            'SELECT \'ALTER USER bob IDENTIFIED BY VALUES "S:XyZs3cr3t"\' AS ddl FROM dual',
            'SELECT \'ALTER USER bob IDENTIFIED BY VALUES "[redacted]"\' AS ddl FROM dual',
        ),
        (
            'SELECT \'ALTER USER "bob" IDENTIFIED BY "XyZnew" REPLACE "XyZs3cr3t"\' AS ddl FROM t',
            'SELECT \'ALTER USER "bob" IDENTIFIED BY "[redacted]" REPLACE "[redacted]"\' AS ddl FROM t',
        ),
        (
            "SELECT q'[a']' FROM dual; ALTER USER bob IDENTIFIED BY XyZnew REPLACE XyZs3cr3t; SELECT 'b' FROM dual",
            "SELECT q'[a']' FROM dual; ALTER USER bob IDENTIFIED BY [redacted] REPLACE [redacted]; SELECT 'b' FROM dual",
        ),
        (
            'SELECT \'ALTER USER "bob" IDENTIFIED BY RANDOM PASSWORD REPLACE "XyZs3cr3t"\' AS ddl FROM t',
            'SELECT \'ALTER USER "bob" IDENTIFIED BY RANDOM PASSWORD REPLACE "[redacted]"\' AS ddl FROM t',
        ),
        (
            "SELECT a FROM t WHERE b = `; ALTER ROLE bob PASSWORD E'XyZs3cr3t'; --`",
            "SELECT a FROM t WHERE b = `; ALTER ROLE bob PASSWORD E'[redacted]'; --`",
        ),
        (
            "SELECT a FROM t WHERE b = `; CREATE USER bob IDENTIFIED BY N'XyZs3cr3t'; --`",
            "SELECT a FROM t WHERE b = `; CREATE USER bob IDENTIFIED BY N'[redacted]'; --`",
        ),
        (
            "SELECT a FROM t WHERE b = `; CREATE USER bob IDENTIFIED BY _utf8mb4'XyZs3cr3t'; --`",
            "SELECT a FROM t WHERE b = `; CREATE USER bob IDENTIFIED BY _utf8mb4'[redacted]'; --`",
        ),
        (
            'SELECT \'CREATE SECRET (TYPE S3, KEY_ID "AKIAEXAMPLE", SECRET "XyZa", SESSION_TOKEN "XyZs3cr3t")\' AS ddl FROM t',
            'SELECT \'CREATE SECRET (TYPE S3, KEY_ID "AKIAEXAMPLE", SECRET "[redacted]", SESSION_TOKEN "[redacted]")\' AS ddl FROM t',
        ),
        (
            'SELECT \'CREATE SECRET (TYPE AZURE, TENANT_ID "t", CLIENT_ID "c", CLIENT_SECRET "XyZs3cr3t")\' AS ddl FROM t',
            'SELECT \'CREATE SECRET (TYPE AZURE, TENANT_ID "t", CLIENT_ID "c", CLIENT_SECRET "[redacted]")\' AS ddl FROM t',
        ),
        (
            'SELECT \'CREATE SECRET (TYPE AZURE, ACCOUNT_NAME "a", ACCOUNT_KEY "XyZs3cr3t")\' AS ddl FROM t',
            'SELECT \'CREATE SECRET (TYPE AZURE, ACCOUNT_NAME "a", ACCOUNT_KEY "[redacted]")\' AS ddl FROM t',
        ),
        (
            'SELECT \'CREATE SECRET hf (TYPE HUGGINGFACE, TOKEN "XyZs3cr3t")\' AS ddl FROM t',
            'SELECT \'CREATE SECRET hf (TYPE HUGGINGFACE, TOKEN "[redacted]")\' AS ddl FROM t',
        ),
        (
            'SELECT \'CREATE SERVER s FOREIGN DATA WRAPPER w OPTIONS (sfPassword "XyZs3cr3t")\' AS ddl FROM t',
            'SELECT \'CREATE SERVER s FOREIGN DATA WRAPPER w OPTIONS (sfPassword "[redacted]")\' AS ddl FROM t',
        ),
    ], ids=[
        "oracle-q-quote-split-miss", "backtick-split-miss", "identified-by-in-a-literal", "identified-with-a-plugin",
        "password-keyword", "secret-keyword", "set-password-for", "identified-by-password-hash",
        "identified-with-a-quoted-plugin", "identified-with-a-double-quoted-plugin",
        "identified-with-a-backticked-plugin", "identified-with-a-plugin-as-a-hash", "identified-by-values",
        "replace-quoted", "replace-bare", "random-password-replace", "postgres-escape-string", "national-string",
        "charset-introducer", "duckdb-s3-session-token", "duckdb-azure-client-secret", "duckdb-azure-account-key",
        "duckdb-huggingface-token", "foreign-server-option",
    ])
    def test_a_password_a_statement_sets_by_keyword_is_masked(self, sql: str, listed: str):
        """A split the readings miss lists the statement after a query, and a
        literal may hold DDL. IDENTIFIED BY, a secret key's name before a
        quoted value, and REPLACE after a new password set a password without
        a key=value pair."""
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].queries == [listed]
        assert "XyZs3cr3t" not in event

    @pytest.mark.parametrize("sql", [
        "SELECT 'Your password ' || name FROM users",
        "SELECT CONCAT('Temporary password ', name, '!') FROM users",
        "SELECT a FROM t WHERE label = 'top secret ' AND owner = 'bob'",
        "SELECT a FROM t WHERE brand = 'Victoria''s Secret ' AND qty > 0 AND region = 'EU'",
        "SELECT 'Users identified by ' || name FROM users",
        "SELECT 'CREATE USER bob IDENTIFIED BY RANDOM PASSWORD' AS ddl FROM t",
    ], ids=[
        "password-then-concatenation", "password-then-argument", "secret-then-and", "doubled-quote-then-secret",
        "identified-by-then-concatenation", "random-password",
    ])
    def test_a_literal_that_ends_in_a_keyword_keeps_the_sql_after_it(self, sql: str):
        """The quote that closes a literal ending in PASSWORD, SECRET or
        IDENTIFIED BY is followed by a space, a comma or a ), so it is not
        read as a password's opening quote, which would mask the SQL up to
        the next quote. RANDOM PASSWORD sets no password the text holds."""
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        assert FigureSources.model_validate_json(_event(step)).steps[0].queries == [sql]

    @pytest.mark.parametrize(("text", "masked"), [
        (
            "CREATE USER u IDENTIFIED BY secret_token=ab'XyZs3cr3t",
            "CREATE USER u IDENTIFIED BY [redacted]",
        ),
        (
            "SELECT q'[a']' FROM dual; ALTER USER bob IDENTIFIED BY pwd=XyZ(abc)XyZs3cr3t; SELECT 'b' FROM dual",
            "SELECT q'[a']' FROM dual; ALTER USER bob IDENTIFIED BY [redacted]; SELECT 'b' FROM dual",
        ),
        (
            "CREATE USER u IDENTIFIED BY token}Bearer XyZs3cr3t0123",
            "CREATE USER u IDENTIFIED BY [redacted] [redacted]",
        ),
    ], ids=["key-value-with-a-quote", "key-value-with-parentheses", "bearer-token"])
    @pytest.mark.parametrize("where", ["label", "query"])
    def test_identified_by_leaves_a_key_or_bearer_value_to_its_own_rule(self, text: str, masked: str, where: str):
        """IDENTIFIED BY's unquoted value stops at a quote or a parenthesis,
        so it runs after the key=value and bearer rules have masked the
        whole value whose key or Bearer it would take."""
        sql = text if text.startswith("SELECT") else f"SELECT qty FROM stock -- {text}"
        step = _cell(
            "toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n",
            description=f"Run {text}" if where == "label" else "Read stock",
        )

        event = _event(step)

        [step_out] = FigureSources.model_validate_json(event).steps
        if where == "label":
            assert step_out.label == f"Run {masked}"
        else:
            assert step_out.queries == [masked if text.startswith("SELECT") else f"SELECT qty FROM stock -- {masked}"]
        assert "XyZ" not in event

    @pytest.mark.parametrize("description", [
        "Reset the password policy", "Count secret santa entries", 'Check the "password": "x" key',
    ])
    def test_a_word_after_password_or_secret_that_is_not_quoted_stays(self, description: str):
        """PASSWORD and SECRET mask only a quoted value after a space, so prose
        and a JSON key keep their words."""
        step = _cell(
            "toolu_k", code="print(query_minds_data('SELECT qty FROM stock'))", stdout="42\n", description=description,
        )

        [step_out] = FigureSources.model_validate_json(_event(step)).steps

        assert step_out.label == description.replace('"x"', '"[redacted]"')

    @pytest.mark.parametrize(("literal", "redacted"), [
        ("password => 'x9secret'", "password => '[redacted]'"),
        ('password=>"x9secret"', 'password=>"[redacted]"'),
    ])
    def test_a_value_after_a_fat_arrow_is_redacted_once(self, literal: str, redacted: str):
        sql = f"SELECT qty FROM t WHERE {literal}"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        assert FigureSources.model_validate_json(_event(step)).steps[0].queries == [
            f"SELECT qty FROM t WHERE {redacted}",
        ]

    @pytest.mark.parametrize(("text", "masked"), [
        ("ALTER LOGIN sa WITH PASSWORD = N'correct horse'", "ALTER LOGIN sa WITH PASSWORD = [redacted]'"),
        ("CREATE LOGIN bob WITH PASSWORD=N'Tr0ub,4dor'", "CREATE LOGIN bob WITH PASSWORD=[redacted]'"),
        ("CALL connect(password => E'a b')", "CALL connect(password => [redacted]')"),
        ("SET PASSWORD = _utf8mb4'a b'", "SET PASSWORD = [redacted]'"),
        # After a quoted key the unquoted-value rule does not read again, so the introducer and both quotes stay.
        ("{\"password\": N'a b'}", "{\"password\": N'[redacted]'}"),
        ("CALL connect('secret' => E'a b')", "CALL connect('secret' => E'[redacted]')"),
    ], ids=[
        "national-string", "national-string-with-a-comma", "escape-string-after-a-fat-arrow", "charset-introducer",
        "national-string-after-a-quoted-key", "escape-string-after-a-quoted-key",
    ])
    @pytest.mark.parametrize("where", ["label", "query"])
    def test_a_quoted_value_after_a_string_introducer_is_redacted_whole(self, text: str, masked: str, where: str):
        """A string's introducer may stand between the = and the value's
        quote, as SQL Server's scripts write PASSWORD=N'...'. Read as an
        unquoted value, the password would end at its first space or comma
        and leave the rest."""
        sql = f"SELECT qty FROM stock -- {text}"
        step = _cell(
            "toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n",
            description=f"Run {text}" if where == "label" else "Read stock",
        )

        event = _event(step)

        [step_out] = FigureSources.model_validate_json(event).steps
        if where == "label":
            assert step_out.label == f"Run {masked}"
        else:
            assert step_out.queries == [f"SELECT qty FROM stock -- {masked}"]
        for secret in ("horse", "4dor", "a b"):
            assert secret not in event

    @pytest.mark.parametrize(("sql", "masked"), [
        (
            "SELECT * FROM read_source('host=db;pwd=_XyZsecret') WHERE qty > 5 AND region = 'west'",
            "SELECT * FROM read_source('host=db;pwd=[redacted]') WHERE qty > 5 AND region = 'west'",
        ),
        (
            "SELECT * FROM read_source('host=db password=_XyZTr0ub4dor') WHERE qty > 5",
            "SELECT * FROM read_source('host=db password=[redacted]') WHERE qty > 5",
        ),
    ], ids=["after-a-semicolon", "after-a-space"])
    def test_an_unquoted_secret_starting_with_an_underscore_keeps_the_query_after_its_literal(
        self, sql: str, masked: str,
    ):
        """A value that starts with _ right before the quote that closes its
        literal is not a _charset introducer. Read as one, that quote would
        open a quoted value running to the next quote and mask the rest of
        the query."""
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].queries == [masked]
        assert "XyZ" not in event

    @pytest.mark.parametrize(
        "filler",
        [
            "a" * 100_000, "password" * 7_500, "x://a:" * 25_000, "x://a:''" * 10_240, "eyJ-" * 65_536,
            # Many :// or many EZConnect starts, from each of which a URL or
            # EZConnect rule reads up to its window or to the end of the run,
            # or an EZConnect password with many words that almost name a
            # secret key: a rule that read the same text again from each
            # start would take time that grows with the square of the length.
            ("x://''x:" * 125 + " ") * 63, ("x://" + "a" * 300 + "/") * 200, 'a/"' + ' b/""' * 13_000,
            "x:///" * 13_000, ' "' + 'a""' * 21_000, "a/" + ("#" + "x" * 60 + "passwor") * 1_500 + "@a",
            # A braced value left open, a proxy login's bracket before each /,
            # closed or with its quote left open, and URL users holding lone
            # and escaped quotes.
            "pwd={" + "x;" * 50_000, ("a[" + "b" * 127 + "]/") * 700, ('a["' + "b" * 127 + "/") * 700,
            "x://o'b\\\"c" * 12_500,
            # Keywords that start a password a statement sets, each without
            # its value or with a plugin's quote left open, and a REPLACE
            # after each masked password.
            ("password for " + "x" * 250 + " ") * 380, "identified with p by\t" * 4_800,
            ('identified with "' + "p" * 60 + " ") * 850, "identified by x replace " * 2_700,
            # URL users holding an @ in every other character up to the user
            # window, before a short or a long password; introducers after a
            # secret key with no quote after them; and EZConnect logins
            # before a placeholder's first character.
            ("://" + "a@" * 128 + ":") * 1_000, ("://" + "a@" * 128 + ":" + "p" * 1_100 + " ") * 160,
            "password=N" * 25_000, ("password=_" + "x" * 40) * 5_000, "a/b@{" * 50_000,
        ],
        ids=[
            "word", "secret-key", "userinfo", "quoted-userinfo", "jwt",
            "doubled-quote-user", "long-user", "ezconnect-quoted", "query-string", "ezconnect-quoted-user",
            "ezconnect-near-miss-keys", "open-brace-value", "ezconnect-proxy", "ezconnect-proxy-open-quote",
            "quoted-user", "keyword-password-for", "keyword-identified", "keyword-open-quoted-plugin",
            "keyword-replace", "user-with-at-signs", "long-password-after-a-user-with-at-signs",
            "introducer-without-a-quote", "charset-introducer-without-a-quote", "ezconnect-placeholder-host",
        ],
    )
    def test_a_long_query_is_redacted_quickly(self, filler: str):
        sql = f"SELECT qty FROM read_source('{filler}')"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")
        start = time.perf_counter()

        _event(step)

        assert time.perf_counter() - start < TIME_BOUND_SECONDS

    def test_a_quoted_passphrase_is_redacted_whole_and_keeps_its_quotes(self):
        sql = "SELECT qty FROM pg_scan(host='db', password='correct horse battery staple', table='stock')"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].queries == [
            "SELECT qty FROM pg_scan(host='db', password='[redacted]', table='stock')",
        ]
        for word in ("correct", "horse", "battery", "staple"):
            assert word not in event

    def test_a_quoted_passphrase_key_is_a_secret_key(self):
        sql = "SELECT qty FROM t WHERE passphrase = 'correct horse battery'"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].queries == [
            "SELECT qty FROM t WHERE passphrase = '[redacted]'",
        ]
        assert "horse" not in event

    @pytest.mark.parametrize("quote", ["'", '"'])
    def test_a_quoted_value_with_a_doubled_quote_in_it_is_redacted_whole(self, quote: str):
        """SQL writes a quote inside a quoted value as two quotes."""
        value = f"{quote}it{quote}{quote}s XyZs3cret{quote}"
        sql = f"SELECT qty FROM pg_scan(host='db', password={value}, table='stock')"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].queries == [
            f"SELECT qty FROM pg_scan(host='db', password={quote}[redacted]{quote}, table='stock')",
        ]
        assert "XyZs3cret" not in event

    def test_a_uri_password_with_a_doubled_quote_in_it_is_redacted(self):
        sql = "SELECT qty FROM postgres_scan('postgresql://admin:Pa''XyZss9@db.example.com/prod', 'public', 'stock')"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].queries == [
            "SELECT qty FROM postgres_scan('postgresql://***@db.example.com/prod', 'public', 'stock')",
        ]
        assert "XyZss9" not in event
        assert "admin" not in event

    @pytest.mark.parametrize("password", [
        "a" * 10 + "@XyZ" + "b" * 1_100,
        "a" * 10 + "@XyZ''" + "b" * 1_100,
    ], ids=["at-sign", "at-sign-and-doubled-quote"])
    def test_a_long_uri_password_with_an_at_sign_in_it_is_masked_whole(self, password: str):
        """A shorter rule would stop at the @ inside the password and leave
        the rest of it."""
        sql = f"SELECT qty FROM postgres_scan('postgresql://u:{password}@db.example/x', 'public', 'stock')"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].queries == [
            "SELECT qty FROM postgres_scan('postgresql://*** 'public', 'stock')",
        ]
        assert "XyZ" not in event
        assert "u:" not in event

    @pytest.mark.parametrize(("url", "masked"), [
        ("postgresql://o''brien:XyZpw@db.example/x", "postgresql://***@db.example/x')"),
        ("postgresql://" + "u" * 300 + ":XyZpw@db.example/x", "postgresql://***"),
        ("https://ab''XyZtok@github.example/x.csv", "https://***@github.example/x.csv')"),
        ('https://ab""XyZtok@github.example/x.csv', "https://***@github.example/x.csv')"),
    ], ids=["doubled-quote-in-user", "long-user", "user-only-doubled-quote", "user-only-doubled-double-quote"])
    def test_a_uri_user_with_a_doubled_quote_or_over_256_characters_is_masked(self, url: str, masked: str):
        sql = f"SELECT qty FROM read_source('{url}')"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].queries == [f"SELECT qty FROM read_source('{masked}"]
        assert "XyZ" not in event

    @pytest.mark.parametrize("password", [
        "XyZ://" + "b" * 300,
        "XyZsec://" + "b" * 300 + ":tail",
    ], ids=["long-run-then-at-sign", "long-run-then-colon"])
    def test_a_short_uri_password_holding_a_scheme_is_masked_with_its_user(self, password: str):
        """The run of over 256 characters after the :// inside the password
        is not read as a long user name, which would mask from that :// on
        and leave the user and the start of the password visible."""
        sql = f"SELECT qty FROM read_source('postgresql://bob:{password}@db.example/x')"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].queries == [
            "SELECT qty FROM read_source('postgresql://***@db.example/x')",
        ]
        assert "XyZ" not in event
        assert "bob:" not in event

    @pytest.mark.parametrize(("login", "masked"), [
        ('"XyZuser"/password=Tr0ub@XyZdor', "***/password=[redacted]"),
        ("XyZuser/password=Tr0ub" + "x" * 1_100 + "@XyZdor", "***/password=[redacted]"),
        ("XyZuser/password=Tr0ub@XyZdor", "***/password=[redacted]"),
        ("XyZuser/DB_Token:Tr0ub@XyZdor", "***/password=[redacted]"),
        ('"XyZuser"/#password=Tr0ub@XyZdor', "***/password=[redacted]"),
        ("XyZuser/''password=Tr0ub@XyZdor", "***/password=[redacted]"),
        ("XyZuser/XyZpw&password=Tr0ub@XyZdor", "***/password=[redacted]"),
        ("XyZuser/password=Tr0ub&XyZpw@XyZdor", "***/password=[redacted]"),
        ("XyZuser/XyZs3cr3t_token=@db.example:1521/orcl", "***/password=[redacted]"),
        ("XyZuser/XyZPwd:XyZs3cr3t@db.example:1521/orcl", "***/password=[redacted]"),
    ], ids=[
        "double-quoted-user", "long-value", "short-value", "colon-and-capitals", "key-after-a-mark",
        "key-after-a-doubled-quote", "password-before-the-key", "password-after-the-keys-value",
        "password-is-a-key-name", "password-starts-with-a-key-name",
    ])
    def test_an_ezconnect_password_holding_a_secret_key_is_masked_with_the_keys_whole_value(
        self, login: str, masked: str,
    ):
        """A word, a / and then text holding a secret key's name and = read
        both as an EZConnect login and as a key and its value, whose value
        runs past the @. The user, the whole password and the whole value
        are masked. The key's name may be the password itself, or its start,
        so a fixed key name stands in for it."""
        sql = f"SELECT qty FROM oracle_scan('{login}')"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].queries == [
            f"SELECT qty FROM oracle_scan('{masked}')",
        ]
        assert "XyZ" not in event

    @pytest.mark.parametrize("login", [
        'scott/"XyZtiger"@', "scott/" + "t" * 1_100 + "XyZ@", "scott/ti''XyZger@",
        '"Scott"/XyZtiger@', '"O""Brien"/"XyZtiger"@', "proxy[XyZscott]/XyZtiger@", 'proxy["XyZ scott"]/XyZtiger@',
    ], ids=[
        "double-quoted", "long", "doubled-quote", "double-quoted-user", "double-quoted-user-and-password",
        "proxy-user", "double-quoted-proxy-user",
    ])
    def test_an_ezconnect_login_quoted_long_or_with_a_doubled_quote_is_masked(self, login: str):
        sql = f"SELECT qty FROM oracle_scan('{login}db.example:1521/orcl')"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].queries == [
            "SELECT qty FROM oracle_scan('***@db.example:1521/orcl')",
        ]
        assert "XyZ" not in event

    @pytest.mark.parametrize(("host", "masked"), [
        (
            "(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=db.example)(PORT=1521))(CONNECT_DATA=(SERVICE_NAME=orcl)))",
            "***@(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=db.example)(PORT=1521))(CONNECT_DATA=(SERVICE_NAME=orcl)))",
        ),
        ("[::1]:1521/orcl", "***@[::1]:1521/orcl"),
    ], ids=["connect-descriptor", "ipv6-host"])
    @pytest.mark.parametrize("where", ["label", "query"])
    def test_an_ezconnect_login_before_a_connect_descriptor_or_an_ipv6_host_is_masked(
        self, host: str, masked: str, where: str,
    ):
        login = f"scott/XyZtiger@{host}"
        sql = f"SELECT qty FROM oracle_scan('{login}')"
        step = _cell(
            "toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n",
            description=f"Connect {login}" if where == "label" else "Read stock",
        )

        event = _event(step)

        [step_out] = FigureSources.model_validate_json(event).steps
        if where == "label":
            assert step_out.label == f"Connect {masked}"
        else:
            assert step_out.queries == [f"SELECT qty FROM oracle_scan('{masked}')"]
        assert "XyZ" not in event
        assert "scott" not in event

    @pytest.mark.parametrize(("call", "host"), [
        ("cur.execute(f\"SELECT qty FROM oracle_scan('scott/XyZtiger@{host}:1521/orcl')\")", "{host}"),
        (
            "cur.execute(\"SELECT qty FROM oracle_scan('scott/XyZtiger@%(host)s:1521/orcl')\", {'host': host})",
            "%(host)s",
        ),
        ("cur.execute(\"SELECT qty FROM oracle_scan('scott/XyZtiger@$HOST:1521/orcl')\")", "$HOST"),
    ], ids=["fstring-placeholder", "query-parameter", "variable"])
    def test_an_ezconnect_login_before_a_placeholder_host_is_masked(self, call: str, host: str):
        """The code may fill the host in when the query runs, as the URL rules
        already mask a login before {host}."""
        step = _cell("toolu_k", code=f"{call}\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].queries == [
            f"SELECT qty FROM oracle_scan('***@{host}:1521/orcl')",
        ]
        assert "XyZ" not in event
        assert "scott" not in event

    @pytest.mark.parametrize("sql", [
        "SELECT qty FROM read_source(E'postgres://o\\'XyZbrien:XyZpw@db.example/x')",
        "SELECT qty FROM read_source('postgres://o\"XyZbrien:XyZpw@db.example/x')",
        "SELECT qty FROM read_source(E'https://ab\\'XyZtok@github.example/x.csv')",
        "SELECT qty FROM read_source('https://ab\"XyZtok@github.example/x.csv')",
    ], ids=["backslash-quote-in-user", "lone-double-quote-in-user", "user-only-backslash-quote", "user-only-lone-quote"])
    def test_a_uri_user_with_an_escaped_or_lone_quote_is_masked(self, sql: str):
        """Without the SQL dialect a quote in a URL's user name may be part of
        it, as it may be in the password."""
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        [query] = FigureSources.model_validate_json(event).steps[0].queries
        assert "://***@" in query
        assert "XyZ" not in event

    @pytest.mark.parametrize(("url", "masked", "masked_sql"), [
        (
            "snowflake://XyZjane.doe@corp.com:XyZHunter2Pw@acct/db", "snowflake://***@acct/db",
            "SELECT qty FROM read_source('snowflake://***@acct/db')",
        ),
        (
            "postgresql://XyZadmin@srv:XyZHunter2Pw@db.example.com/erp", "postgresql://***@db.example.com/erp",
            "SELECT qty FROM read_source('postgresql://***@db.example.com/erp')",
        ),
        (
            "postgresql://XyZadmin%40srv:XyZHunter2Pw@db.example.com/erp", "postgresql://***@db.example.com/erp",
            "SELECT qty FROM read_source('postgresql://***@db.example.com/erp')",
        ),
        # A password over 1,024 characters masks the whole non-whitespace span, the
        # host and the literal's closing quote included.
        (
            "snowflake://XyZjane.doe@corp.com:XyZ" + "p" * 1_100 + "@acct/db", "snowflake://***",
            "SELECT qty FROM read_source('snowflake://***",
        ),
        (
            "postgresql://XyZadmin@srv:XyZ" + "p" * 1_100 + "@db.example.com/erp", "postgresql://***",
            "SELECT qty FROM read_source('postgresql://***",
        ),
    ], ids=[
        "email-user", "user-at-server", "percent-encoded-at-sign",
        "email-user-long-password", "user-at-server-long-password",
    ])
    @pytest.mark.parametrize("where", ["label", "query"])
    def test_a_uri_user_holding_an_at_sign_is_masked_with_its_password(
        self, url: str, masked: str, masked_sql: str, where: str,
    ):
        """urllib reads a URL's user and password up to the last @ before its
        path, and SQLAlchemy reads the user up to its first : and the
        password up to the next @. Both read an email login or a user@server
        login that stands unencoded before the password as the user."""
        sql = f"SELECT qty FROM read_source('{url}')"
        step = _cell(
            "toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n",
            description=f"Read {url}" if where == "label" else "Read stock",
        )

        event = _event(step)

        [step_out] = FigureSources.model_validate_json(event).steps
        if where == "label":
            assert step_out.label == f"Read {masked}"
        else:
            assert step_out.queries == [masked_sql]
        assert "XyZ" not in event
        assert "ppppp" not in event

    @pytest.mark.parametrize(("url", "masked", "password_parts"), [
        ("mysql://app@db.internal:3306/sales?password=P@ssw0rd", "mysql://***@db.internal:3306/sales?...", ["ssw0rd"]),
        (
            "postgresql://app@db.example.com:5432/erp?sslmode=require&password=Tr0ub@dor9",
            "postgresql://***@db.example.com:5432/erp?...", ["Tr0ub", "dor9"],
        ),
    ], ids=["mysql", "postgresql"])
    @pytest.mark.parametrize("where", ["label", "query"])
    def test_a_query_strings_password_holding_an_at_sign_after_a_user_and_a_port_is_masked(
        self, url: str, masked: str, password_parts: list[str], where: str,
    ):
        """urllib reads these URLs' user as app and leaves password=... in the
        query string. Read as a user that holds an @ and a password up to the
        query string's @, db.internal:3306/sales?password=P would be masked
        as the user and password, and the password's part after its @ would
        show."""
        sql = f"SELECT qty FROM read_source('{url}')"
        step = _cell(
            "toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n",
            description=f"Read {url}" if where == "label" else "Read stock",
        )

        event = _event(step)

        [step_out] = FigureSources.model_validate_json(event).steps
        if where == "label":
            assert step_out.label == f"Read {masked}"
        else:
            assert step_out.queries == [f"SELECT qty FROM read_source('{masked}')"]
        for part in password_parts:
            assert part not in event

    @pytest.mark.parametrize(("sql", "masked"), [
        (
            "SELECT qty FROM read_source('postgresql://app@db.example.com:5432/erp','password=Tr0ub@dor9')",
            "SELECT qty FROM read_source('postgresql://***@db.example.com:5432/erp','password=[redacted]')",
        ),
        (
            "SELECT qty FROM read_source('{\"dsn\":\"mysql://app@db.internal:3306/sales\",\"password\":\"Tr0ub@dor9\"}')",
            "SELECT qty FROM read_source('{\"dsn\":\"mysql://***@db.internal:3306/sales\",\"password\":\"[redacted]\"}')",
        ),
        (
            "SELECT qty FROM read_source('jdbc:sqlserver://app@db:1433;databaseName=erp;password=Tr0ub@dor9')",
            "SELECT qty FROM read_source('jdbc:sqlserver://***@db:1433;databaseName=erp;password=[redacted]')",
        ),
    ], ids=["next-literal", "compact-json", "semicolon-parameter"])
    @pytest.mark.parametrize("where", ["label", "query"])
    def test_a_secret_after_a_user_host_and_port_in_the_same_run_is_masked_whole(
        self, sql: str, masked: str, where: str,
    ):
        """A URL with a user, a host and a port and no password, followed with
        no space by a secret key's value that holds an @. Read as a user that
        holds an @ and a password up to that @, the URL and the key would be
        masked as one login and the value's part after its @ would show."""
        step = _cell(
            "toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n",
            description=f"Run {sql}" if where == "label" else "Read stock",
        )

        event = _event(step)

        [step_out] = FigureSources.model_validate_json(event).steps
        if where == "label":
            assert step_out.label == f"Run {masked}"
        else:
            assert step_out.queries == [masked]
        assert "dor9" not in event

    def test_a_password_holding_an_at_sign_and_a_hash_after_a_user_with_no_at_sign_is_masked_whole(self):
        """A user with no @ is read first, so its password runs to the last @
        even past a # in it."""
        url = "postgresql://app:Tr0ub@dor#9@db.example.com/erp"
        step = _cell(
            "toolu_k", code="cur.execute('SELECT qty FROM stock')\nprint(cur.fetchone()[0])", stdout="42\n",
            description=f"Read {url}",
        )

        event = _event(step)

        [step_out] = FigureSources.model_validate_json(event).steps
        assert step_out.label == "Read postgresql://***@db.example.com/erp"
        assert "dor" not in event

    @pytest.mark.parametrize(("source", "masked"), [
        (
            "api_read('{\"url\":\"https://api.example\",\"password\":\"Tr0ub@XyZdor\"}')",
            "api_read('{\"url\":\"https://api.example\",\"password\":\"[redacted]\"}')",
        ),
        ("read_source('http://h','token=Tr0ub@XyZdor')", "read_source('http://h','token=[redacted]')"),
        (
            "read_source('http://h','" + "x" * 300 + "@a',password='XyZone XyZtwo')",
            "read_source('http://h','" + "x" * 300 + "@a',password='[redacted]')",
        ),
        ("read_source(['http://h'],password='Tr0ub@XyZdor')", "read_source(['http://h'],password='[redacted]')"),
        (
            "read_source({'endpoint':'http://minio','password':'Tr0ub@XyZdor'})",
            "read_source({'endpoint':'http://minio','password':'[redacted]'})",
        ),
    ], ids=["json-member", "next-argument", "long-next-argument", "after-a-bracket", "python-dict"])
    @pytest.mark.parametrize("where", ["label", "query"])
    def test_a_quote_that_closes_a_url_keeps_the_next_secret_keys_name(self, source: str, masked: str, where: str):
        """The quote that closes a literal holding a URL ends the URL's user
        name, so the user name does not run on into the next literal and take
        the name of the key whose value holds an @."""
        sql = f"SELECT qty FROM {source}"
        step = _cell(
            "toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n",
            description=f"Read {source}" if where == "label" else "Read stock",
        )

        event = _event(step)

        [step_out] = FigureSources.model_validate_json(event).steps
        if where == "label":
            assert step_out.label == f"Read {masked}"
        else:
            assert step_out.queries == [f"SELECT qty FROM {masked}"]
        assert "XyZ" not in event

    @pytest.mark.parametrize("source", [
        "https://acct.blob.core.windows.net/c/Q3 sales.csv?sv=2024-05-04&sp=r&sig=XyZs3cr3t",
        "E'https://acct.blob.core.windows.net/c/O\\'Brien.csv?sv=2024-05-04&sp=r&sig=XyZs3cr3t",
        'https://acct.blob.core.windows.net/c/a"b.csv?sv=2024-05-04&sp=r&sig=XyZs3cr3t',
        "BlobEndpoint=https://acct.blob.core.windows.net/;SharedAccessSignature=sv=2024-05-04&sp=r&sig=XyZs3cr3t",
        "https://b.s3.amazonaws.com/Q3 sales.csv?X-Amz-Credential=AKIA%2F&X-Amz-Signature=XyZs3cr3t",
        "https://storage.googleapis.com/b/Q3 sales.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Signature=XyZs3cr3t",
    ], ids=[
        "space-in-path", "backslash-quote-in-path", "lone-double-quote-in-path", "azure-connection-string",
        "s3-presigned-space-in-path", "gcs-signed-space-in-path",
    ])
    def test_a_url_signature_the_query_string_rule_misses_is_masked_by_its_key(self, source: str):
        """A space or a quote in a URL's path ends the query string rule's
        path before the ?, and a connection string's signature has no URL
        query at all; the signature's own key still names it."""
        literal = source if source.startswith("E'") else f"'{source}"
        sql = f"SELECT qty FROM read_csv({literal}')"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].queries
        assert "XyZs3cr3t" not in event

    @pytest.mark.parametrize(("url", "masked"), [
        ("https://h.example/" + "a" * 2_100 + "?sig=XyZsig", "https://h.example/" + "a" * 2_100 + "?..."),
        ("https://h.example/x.csv?sig=ab''XyZsig", "https://h.example/x.csv?..."),
        ("https://h.example/o''x.csv?sig=XyZsig", "https://h.example/o''x.csv?..."),
    ], ids=["long-path", "doubled-quote-in-query", "doubled-quote-in-path"])
    def test_a_query_string_after_a_long_or_quoted_path_is_masked(self, url: str, masked: str):
        sql = f"SELECT qty FROM read_csv('{url}')"
        step = _cell("toolu_k", code=f"cur.execute({sql!r})\nprint(cur.fetchone()[0])", stdout="42\n")

        event = _event(step)

        assert FigureSources.model_validate_json(event).steps[0].queries == [f"SELECT qty FROM read_csv('{masked}')"]
        assert "XyZsig" not in event


class TestFiguresWithoutASource:
    def test_a_figure_no_step_returned_has_no_source_and_names_no_step(self):
        sources = find_figure_sources(
            answer="An hour of overtime costs $30.", question="What does an hour of overtime cost?",
            steps=[READ_STOCK],
        )

        assert _figure(sources, "$30").origin is FigureOrigin.none
        assert _figure(sources, "$30").sources == []
        assert sources.steps == []

    def test_a_cell_that_prints_a_literal_it_was_given_is_not_a_source(self):
        echo = _cell("toolu_echo", code="cost = 30\nprint(cost)", stdout="30\n")

        sources = find_figure_sources(answer="Overtime costs $30 an hour.", question="", steps=[echo])

        assert _figure(sources, "$30").origin is FigureOrigin.none

    @pytest.mark.parametrize(("hours", "figure"), [
        ("about **22 hours**", "22"), ("a 22-hour wait", "22"), ("about 22h", "22h"), ("about 22 hrs", "22"),
        ("about 22 working hours", "22"), ("about 22-25 hours", "22"), ("22 to 25 hours", "22"),
    ], ids=["word", "hyphen", "glued", "short-form", "qualified", "range-hyphen", "range-to"])
    def test_a_figure_in_hours_is_not_held_by_a_days_column_that_prints_its_number(self, hours: str, figure: str):
        """The model worked out 22 hours from 0.92 days; the 22 the sample
        rows print is another item's lead_time_days."""
        sample = _cell(
            "toolu_sample",
            code=(
                "cur.execute('SELECT * FROM inventory LIMIT 3')\n"
                "print([d.name for d in cur.description])\n"
                "for row in cur.fetchall():\n    print(row)"
            ),
            stdout="['sku', 'count_in_stock', 'lead_time_days']\n('SKU-0027', 46, 25)\n('SKU-0028', 313, 22)\n",
            description="Inspect the inventory table's sample records",
        )
        stockout = _cell(
            "toolu_stockout",
            code=(
                "cur.execute('SELECT sku, projected_out_of_stock_days FROM inventory ORDER BY 2')\n"
                "print(cur.fetchone())"
            ),
            stdout="('SKU-0127', 0.9157509)\n",
        )

        sources = find_figure_sources(
            answer=f"SKU-0127 runs out in **0.92 days**, {hours}.", question="", steps=[sample, stockout],
        )

        assert _figure(sources, figure).origin is FigureOrigin.none
        assert _figure(sources, figure).sources == []
        assert _figure(sources, "0.92").sources == [FigureSource(tool_use_id="toolu_stockout")]
        assert [step.tool_use_id for step in sources.steps] == ["toolu_stockout"]

    def test_a_memorize_call_is_not_a_source(self):
        memorize = _tool("toolu_mem", "memorize", {"text": "Overtime costs 30 dollars"}, "Saved: Overtime costs 30 dollars")

        sources = find_figure_sources(answer="Overtime costs $30.", question="", steps=[memorize])

        assert _figure(sources, "$30").origin is FigureOrigin.none

    def test_a_non_exec_scratchpad_action_is_not_a_source(self):
        dump = TurnStep(
            tool_use_id="toolu_dump", tool_name="scratchpad", action="dump", tool_input=ToolInput.from_text(text="{}"),
            content="250",
        )

        sources = find_figure_sources(answer="S1 holds 250 units.", question="", steps=[dump])

        assert _figure(sources, "250").origin is FigureOrigin.none

    @pytest.mark.parametrize("tool_input", [
        pytest.param(ToolInput.from_text(text=json.dumps({"query": "overtime rates 2024"})), id="streamed"),
        pytest.param(ToolInput.from_call(arguments={"query": "overtime rates 2024"}), id="history-row"),
    ])
    def test_a_number_a_web_calls_query_states_is_only_echoed_by_its_results(self, tool_input: ToolInput):
        search = TurnStep(
            tool_use_id="toolu_search", tool_name="web_search", action="", tool_input=tool_input,
            content="2024 overtime: $42 per hour",
        )

        sources = find_figure_sources(answer="In 2024 overtime was $42.", question="", steps=[search])

        assert _figure(sources, "2024").origin is FigureOrigin.none
        assert _figure(sources, "$42").sources == [FigureSource(tool_use_id="toolu_search", page=None)]

    def test_a_figure_the_question_states_is_the_questions(self):
        sources = find_figure_sources(
            answer="I found no stock data for the next 3 weeks.", question="What ships in the next 3 weeks?", steps=[],
        )

        assert _figure(sources, "3").origin is FigureOrigin.question


class TestWhatIsNotAFigure:
    def test_identifiers_dates_list_markers_links_and_code_are_skipped(self):
        answer = (
            "1. Part A2210 (32GB) at W3001, question Q-03, on 2026-10-14 at 09:30, 14 October.\n"
            "2. See [the doc](https://example.com/page/42) and `qty = 99`.\n"
            "```\nprint(77)\n```\n"
            "Version 1.2.3 of P100 at S1."
        )

        assert find_figure_sources(answer=answer, question="", steps=[READ_STOCK]) == FigureSources()

    @pytest.mark.parametrize(
        "filler", ["](" * 49_152, "[^1]" * 24_576, "10/14/2026 " * 12_000], ids=["link", "citation", "date"],
    )
    def test_a_long_answer_full_of_skipped_spans_is_read_quickly(self, filler: str):
        text = filler + " S1 holds 250 units."
        start = time.perf_counter()

        sources = find_figure_sources(answer=text, question=text, steps=[READ_STOCK])

        assert time.perf_counter() - start < TIME_BOUND_SECONDS
        assert [f.text for f in sources.figures] == ["250"]

    def test_an_answer_with_no_numbers_has_no_figures(self):
        assert find_figure_sources(
            answer="The data doesn't hold that.", question="", steps=[READ_STOCK],
        ) == FigureSources()


class TestOffsets:
    def test_offsets_count_utf16_units_so_javascript_slices_the_figure(self):
        answer = "📦 S1 holds 250 units."

        [figure] = find_figure_sources(answer=answer, question="", steps=[READ_STOCK]).figures

        utf16 = answer.encode("utf-16-le")
        assert utf16[figure.start * 2:figure.end * 2].decode("utf-16-le") == "250"
        assert figure.start == answer.index("250") + 1


def _origins(sources: FigureSources) -> list[tuple[str, FigureOrigin]]:
    return [(f.text, f.origin) for f in sources.figures]


def _printed(stdout: str, *, answer: str, code: str = "print(total)") -> list[tuple[str, FigureOrigin]]:
    return _origins(find_figure_sources(answer=answer, question="", steps=[_cell("toolu_c", code=code, stdout=stdout)]))


class TestRanges:
    def test_both_ends_of_a_hyphen_range_are_figures(self):
        assert _origins(find_figure_sources(answer="My best estimate is 150-200 units.", question="", steps=[])) == [
            ("150", FigureOrigin.none), ("200", FigureOrigin.none),
        ]

    def test_an_invented_upper_end_is_not_sourced_with_the_lower(self):
        sources = find_figure_sources(answer="Stock ranges 90-400 units.", question="", steps=[READ_STOCK])

        assert _origins(sources) == [("90", FigureOrigin.step), ("400", FigureOrigin.none)]

    @pytest.mark.parametrize(("answer", "texts"), [
        ("Overtime costs $30-$45 an hour.", ["$30", "$45"]),
        ("Overtime costs $10-$20 an hour.", ["$10", "$20"]),
        ("Demand is 5k-10k units.", ["5k", "10k"]),
        ("Lead time is 5-10 days.", ["5", "10"]),
    ])
    def test_currency_and_scaled_ranges_keep_both_ends(self, answer: str, texts: list[str]):
        assert [f.text for f in find_figure_sources(answer=answer, question="", steps=[]).figures] == texts

    @pytest.mark.parametrize("answer", [
        "See question Q-03.", "Order PO-2024-0012 shipped.", "It shipped 10-14-2025.", "Use model A2210-30.",
        "Part 4711-002 is short.", "Call 555-0100.", "SKU 100-200-300 ships.",
    ])
    def test_a_hyphen_inside_an_identifier_or_a_date_still_hides_its_numbers(self, answer: str):
        assert find_figure_sources(answer=answer, question="", steps=[]) == FigureSources()

    def test_a_percent_range_is_two_positive_figures(self):
        sources = _printed("low 0.05\nhigh 0.10\n", answer="Margins run 5%-10%.", code="print(df.margin.describe())")

        assert sources == [("5%", FigureOrigin.step), ("10%", FigureOrigin.step)]

    @pytest.mark.parametrize(("answer", "texts"), [
        ("Balances are -5 and -10.", ["-5", "-10"]),
        ("The drop was -5%, then -10%.", ["-5%", "-10%"]),
        ("Use x-160 here.", []),
    ])
    def test_a_minus_sign_after_a_space_or_a_word_is_not_a_range_hyphen(self, answer: str, texts: list[str]):
        assert [f.text for f in find_figure_sources(answer=answer, question="", steps=[]).figures] == texts

    def test_both_ends_of_a_range_in_the_question_are_the_questions(self):
        sources = find_figure_sources(answer="Yes, 200 units fit.", question="Can we order 150-200 units?", steps=[])

        assert _origins(sources) == [("200", FigureOrigin.question)]


class TestQuestionForms:
    def test_a_figure_in_another_form_than_the_questions_number_is_not_the_questions(self):
        sources = find_figure_sources(
            answer="Overtime costs $30 an hour.", question="What would overtime cost over the next 30 days?", steps=[],
        )

        assert _origins(sources) == [("$30", FigureOrigin.none)]

    def test_a_signed_figure_is_not_the_questions_unsigned_number(self):
        sources = find_figure_sources(answer="Margin moved -5% this year.", question="Did margin fall 5%?", steps=[])

        assert _origins(sources) == [("-5%", FigureOrigin.none)]

    def test_the_question_rule_reads_no_unit(self):
        """A unit written right after the digits is part of the form an
        earlier answer must match, not the question's: a question match only
        spares a figure the no-source mark."""
        sources = find_figure_sources(
            answer="Yes, 12 hours is the cap.", question="Can a crew work 12h straight?", steps=[],
        )

        assert _origins(sources) == [("12", FigureOrigin.question)]

    def test_a_year_in_the_question_does_not_cover_a_rounded_figure(self):
        sources = find_figure_sources(answer="About 2k units ship.", question="How many units ship in 2026?", steps=[])

        assert _origins(sources) == [("2k", FigureOrigin.none)]

    def test_a_date_in_the_question_states_no_figure(self):
        sources = find_figure_sources(answer="14 orders shipped.", question="What shipped on October 14?", steps=[])

        assert _origins(sources) == [("14", FigureOrigin.none)]

    @pytest.mark.parametrize(("answer", "question", "text"), [
        ("Yes, $30 an hour is the rate.", "Is $30 an hour the overtime rate?", "$30"),
        ("12% of orders is the target.", "Is the target 12% of orders?", "12%"),
        ("1,200 units is the cap.", "Is the cap 1200 units?", "1,200"),
    ])
    def test_the_same_form_stays_the_questions(self, answer: str, question: str, text: str):
        sources = find_figure_sources(answer=answer, question=question, steps=[])

        assert _origins(sources) == [(text, FigureOrigin.question)]


class TestDates:
    @pytest.mark.parametrize(("answer", "text"), [
        ("Maybe 40 units arrive.", "40"),
        ("Inventory decreased 15 units.", "15"),
        ("We placed 12 separate orders.", "12"),
        ("Margin 12 points.", "12"),
        ("The remaining 12 may arrive late.", "12"),
    ])
    def test_a_word_that_starts_like_a_month_does_not_hide_a_figure(self, answer: str, text: str):
        assert [f.text for f in find_figure_sources(answer=answer, question="", steps=[]).figures] == [text]

    @pytest.mark.parametrize("answer", [
        "It shipped 10/14/2026.",
        "It shipped 2026/10/14.",
        "It shipped 10-14-2026.",
        "It ships in October 2026.",
        "It ships in Oct. 2026.",
        "It ships in Q3 2026.",
        "It ships in FY 2026.",
        "It ships in 2026 Q3.",
        "It ships at 10:30am.",
        "It ships at 3 PM.",
        "It ships at 10 a.m. sharp.",
        "It shipped 12 May.",
    ])
    def test_more_date_forms_are_not_figures(self, answer: str):
        assert find_figure_sources(answer=answer, question="", steps=[]) == FigureSources()

    def test_a_two_part_slash_is_two_figures(self):
        assert [f.text for f in find_figure_sources(answer="12/15 sites reported.", question="", steps=[]).figures] == [
            "12", "15",
        ]


class TestNumberForms:
    @pytest.mark.parametrize(("answer", "stdout", "text"), [
        ("Orders cost $1.2m in total.", "1234567.89", "$1.2m"),
        ("The contract is worth $5MM.", "5000000", "$5MM"),
        ("Orders cost $1.2 Million in total.", "1234567.89", "$1.2 Million"),
        ("12 % of orders shipped late.", "0.12", "12 %"),
        ("12 percent of orders shipped late.", "0.125", "12 percent"),
        ("The new line runs 1.5x faster.", "1.5", "1.5x"),
        ("The pallet weighs 250kg.", "250", "250kg"),
        ("Overtime costs C$30 an hour.", "30", "C$30"),
        ("S1 holds _250_ units.", "250", "250"),
        ("S1 holds 250 B-grade units.", "250", "250"),
    ])
    def test_a_number_form_reads_at_its_value(self, answer: str, stdout: str, text: str):
        assert _printed(f"{stdout}\n", answer=answer) == [(text, FigureOrigin.step)]

    @pytest.mark.parametrize("answer", ["The site is 250m away.", "Part A2210 needs 32GB.", "See row_250_b."])
    def test_a_unit_or_name_glued_to_a_number_is_still_not_a_figure(self, answer: str):
        assert find_figure_sources(answer=answer, question="", steps=[]) == FigureSources()


class TestRoundedWholeNumbers:
    @pytest.mark.parametrize(("answer", "stdout", "origin"), [
        ("Orders cost about 1,200 dollars.", "1234", FigureOrigin.step),
        ("Orders cost roughly $1,200,000.", "1234567.89", FigureOrigin.step),
        ("S1 holds ~250 units.", "247", FigureOrigin.step),
        ("Orders cost 1,200 dollars.", "1234", FigureOrigin.none),
        ("S1 holds 250 units.", "247", FigureOrigin.none),
        ("Orders cost about 1,000 dollars.", "640", FigureOrigin.none),
        ("Orders cost about 1,000 dollars.", "1040", FigureOrigin.none),
    ])
    def test_a_hedged_round_number_matches_to_its_second_significant_digit(
        self, answer: str, stdout: str, origin: FigureOrigin,
    ):
        [(_, found)] = _printed(f"{stdout}\n", answer=answer)

        assert found is origin


class TestThousandsSeparators:
    @pytest.mark.parametrize(("stdout", "answer", "text", "origin"), [
        ("Total units: 1,250\n", "S1 holds 250 units.", "250", FigureOrigin.none),
        ("Total units 1,250\n", "S1 holds 250 units.", "250", FigureOrigin.none),
        ("Revenue: $1,250,000\n", "S1 holds 250 units.", "250", FigureOrigin.none),
        ("Cost: 3,042.50\n", "Overtime costs $42.50.", "$42.50", FigureOrigin.none),
        ("qty\n1,250\n", "S1 holds 250 units.", "250", FigureOrigin.none),
        ("Totals, by site:\n1,250\n", "S1 holds 250 units.", "250", FigureOrigin.none),
        ("Total units: 1,250\n", "There are 1,250 units.", "1,250", FigureOrigin.step),
        ("[100,250,300]\n", "S1 holds 250 units.", "250", FigureOrigin.step),
        ("site_id,qty\n1,250\n2,90\n", "Site 1 holds 250 units.", "250", FigureOrigin.step),
        ("part,site,qty\nP100,S1,250\n", "S1 holds 250 units.", "250", FigureOrigin.step),
    ])
    def test_a_separator_splits_a_number_only_between_fields(
        self, stdout: str, answer: str, text: str, origin: FigureOrigin,
    ):
        step = _cell("toolu_c", code="print(x)", stdout=stdout)

        assert _figure(find_figure_sources(answer=answer, question="", steps=[step]), text).origin is origin


class TestEchoedLiterals:
    @pytest.mark.parametrize(("code", "stdout", "answer", "text"), [
        ("rate = 29.99\nprint(rate)", "29.99", "Overtime costs $30 an hour.", "$30"),
        ("growth = 0.05\nprint(growth)", "0.05", "Demand grows 5% a year.", "5%"),
        ("target = 1_200_000\nprint(target)", "1200000", "The target is $1.2M.", "$1.2M"),
    ])
    def test_an_output_equal_to_a_literal_is_not_a_source_even_rounded(
        self, code: str, stdout: str, answer: str, text: str,
    ):
        assert _printed(f"{stdout}\n", answer=answer, code=code) == [(text, FigureOrigin.none)]

    def test_a_literal_near_a_real_value_does_not_hide_it(self):
        code = "threshold = 249.9\nprint(df[df.qty > threshold].qty.max())"

        assert _printed("250\n", answer="S1 holds 250 units.", code=code) == [("250", FigureOrigin.step)]


def _printed_frame(labels: list[int]) -> str:
    """A stock frame as print(df) shows it, with these row labels."""
    width = max(len(str(label)) for label in labels)
    lines = [" " * width + "  part site  qty"]
    for label in labels:
        part, site, qty = ("P100", "S1", 250) if label % 2 == 0 else ("P200", "S2", 90)
        lines.append(f"{label:<{width}}  {part}   {site}  {qty:>3}")
    return "\n".join(lines) + "\n"


_SAMPLE_FULL = (
    "[sample:DataFrame]\n"
    "Shape: 40 rows x 3 cols\n"
    "Columns: ['part', 'site', 'qty']\n"
    "Dtypes:\npart    object\nsite    object\nqty      int64\n\n"
    "Describe:\n"
    "        part site         qty\n"
    "count     40   40   40.000000\n"
    "unique     2    2         NaN\n"
    "top     P100   S1         NaN\n"
    "freq      20   20         NaN\n"
    "mean     NaN  NaN  170.000000\n"
    "std      NaN  NaN   81.019149\n"
    "min      NaN  NaN   90.000000\n"
    "25%      NaN  NaN   90.000000\n"
    "50%      NaN  NaN  170.000000\n"
    "75%      NaN  NaN  250.000000\n"
    "max      NaN  NaN  250.000000\n\n"
    "First 40 rows:\n" + _printed_frame(list(range(40)))
)


class TestFrameRowLabels:
    def test_a_helpers_literal_caption_survives_rebinding_the_data_it_prints(self):
        define = _cell("define", code="def show():\n    print('qty')\n    print(stock)", stdout="")
        call = _cell(
            "call", code="stock = pd.Series([250, 90, 250], name='qty')\nshow()",
            stdout="qty\n0    250\n1     90\n2    250\nName: qty, dtype: int64\n",
        )

        event = find_figure_sources(answer="The rate is $2.", question="", steps=[define, call])

        assert _origins(event) == [("$2", FigureOrigin.none)]

    @pytest.mark.parametrize("code", [
        "caption = 'qty'; print(caption); print(stock)",
        "def show():\n    print('qty')\n    print(stock)\nshow()",
    ])
    def test_a_variable_or_helper_caption_does_not_name_an_ordinary_series_index(self, code: str):
        stdout = "qty\n0    250\n1     90\n2    250\nName: qty, dtype: int64\n"

        assert _printed(stdout, answer="Stock is 250 units; the rate is $2.", code=code) == [
            ("250", FigureOrigin.step), ("$2", FigureOrigin.none),
        ]

    @pytest.mark.parametrize("caption", ["Stock quantities:", "qty", "Stock quantities"])
    def test_a_separately_printed_caption_does_not_name_an_ordinary_series_index(self, caption: str):
        stdout = caption + "\n0    250\n1     90\n2    250\nName: qty, dtype: int64\n"

        assert _printed(stdout, answer="Stock is 250 units; the rate is $2.", code=f"print({caption!r}); print(stock)") == [
            ("250", FigureOrigin.step), ("$2", FigureOrigin.none),
        ]

    @pytest.mark.parametrize(("stdout", "invented"), [
        (_printed_frame(list(range(40))), "$30"),
        (_printed_frame(list(range(5))) + "..  ...  ...  ...\n" + _printed_frame(list(range(95, 100))).split("\n", 1)[1]
         + "\n[100 rows x 3 columns]\n", "$3"),
        (_SAMPLE_FULL, "$30"),
        (_SAMPLE_FULL, "25%"),
        (_printed_frame([18, 37, 35]), "$37"),
        (_printed_frame([8]), "$8"),
        ("0    250\n1     90\n2    250\nName: qty, dtype: int64\n", "$2"),
        # The prints below are pandas 3.0.6's.
        pytest.param(
            "3      7\n2     40\n1     90\n0    250\nName: qty, dtype: int64\n", "$2",
            id="print(df.qty.sort_values()), row numbers out of order",
        ),
        pytest.param(
            "Top sites\n4    300\n0    250\n1     90\nName: qty, dtype: int64\n", "$4",
            id="print('Top sites'); print(df.qty.nlargest(3)), a caption over row numbers",
        ),
        pytest.param(
            "   perf  qty\n0 -1.25  250\n1  3.50   90\n2 -4.75   40\n", "$2",
            id="print(df), one space before a negative first value",
        ),
        pytest.param(
            "     qty\n99  1099\n98  1098\n97  1097\n96  1096\n95  1095\n..   ...\n"
            "4   1004\n3   1003\n2   1002\n1   1001\n0    250\n\n[100 rows x 1 columns]\n", "$3",
            id="print(df.sort_values('qty', ascending=False)) of 100 rows",
        ),
    ])
    def test_a_row_label_or_shape_footer_is_not_a_source(self, stdout: str, invented: str):
        answer = f"S1 holds 250 units; the rate is {invented}."

        assert _printed(stdout, answer=answer) == [("250", FigureOrigin.step), (invented, FigureOrigin.none)]

    @pytest.mark.parametrize("stdout", [
        # The prints are pandas 3.0.6's.
        # print(df.pivot_table(index='year', columns='region', values='sales', aggfunc='sum'))
        "region    east    west\nyear                  \n2023    1200.0  3400.0\n2024    2600.0  2600.0\n"
        "2025     900.0     NaN\n",
        # print(pd.crosstab(df.year, df.region))
        "region  east  west\nyear              \n2023       1     1\n2024       1     1\n2025       1     0\n",
        # print(df.groupby('fiscal year')[['sales']].sum())
        "             sales\nfiscal year       \n2023          4600\n2024          5200\n2025           900\n",
        # print(df.groupby(['year', 'region'])['sales'].sum())
        "year  region\n2023  east      1200\n      west      3400\n2024  east      2600\n      west      2600\n"
        "2025  east       900\nName: sales, dtype: int64\n",
    ], ids=["pivot_table with named columns", "crosstab", "frame index name with a space", "MultiIndex Series"])
    def test_a_named_index_is_read_with_named_columns_a_spaced_name_or_a_multiindex(self, stdout: str):
        assert _printed(stdout, answer="Sales peaked in 2024, after 2023, and fell in 2025.") == [
            ("2024", FigureOrigin.step), ("2023", FigureOrigin.step), ("2025", FigureOrigin.step),
        ]

    @pytest.mark.parametrize("stdout", [
        # The prints are pandas 3.0.6's. pandas leaves a column before a key for its minus sign, so an
        # inner key starts a column right of a negative one above it, or a column left of a positive one.
        # print(df.groupby(['year', 'days_late'])[['orders']].sum())
        "                orders\nyear days_late        \n2023 -2             14\n      0             61\n"
        "      3              9\n2024 -2             11\n      0             58\n      3              7\n"
        "2025  0             66\n      3              5\n",
        # print(df.groupby(['year', 'days_late'])['orders'].sum())
        "year  days_late\n2023  -2           14\n       0           61\n       3            9\n"
        "2024  -2           11\n       0           58\n       3            7\n2025   0           66\n"
        "       3            5\nName: orders, dtype: int64\n",
        # print(df.set_index(['year', 'days_late'])), its rows unsorted.
        "                orders\nyear days_late        \n2023  0             61\n     -2             14\n"
        "2024  0             58\n     -2             11\n2025  3              5\n     -2              9\n",
        # print(df.set_index(['year', 'days_late'])['orders'])
        "year  days_late\n2023   0           61\n      -2           14\n2024   0           58\n"
        "      -2           11\n2025   3            5\n      -2            9\nName: orders, dtype: int64\n",
    ], ids=[
        "frame, a negative key first", "Series, a negative key first",
        "frame, a negative key under a positive one", "Series, a negative key under a positive one",
    ])
    def test_a_multiindex_whose_inner_keys_hold_a_negative_number_is_read(self, stdout: str):
        assert _printed(stdout, answer="In 2025, 9 orders shipped early; 2024 had 58.") == [
            ("2025", FigureOrigin.step), ("9", FigureOrigin.step),
            ("2024", FigureOrigin.step), ("58", FigureOrigin.step),
        ]

    @pytest.mark.parametrize("stdout", [
        # The prints are pandas 3.0.6's, of 100 rows indexed by year.
        # print(df.set_index('year')['v'])
        "year\n1990     0\n1991     1\n1992     2\n1993     3\n1994     4\n        ..\n2085    95\n2086    96\n"
        "2087    97\n2088    98\n2089    99\nName: v, Length: 100, dtype: int64\n",
        # print(df.set_index('year'))
        "       v   w\nyear        \n1990   0   0\n1991   1   1\n1992   2   2\n1993   3   3\n1994   4   4\n"
        "...   ..  ..\n2085  95  95\n2086  96  96\n2087  97  97\n2088  98  98\n2089  99  99\n"
        "\n[100 rows x 2 columns]\n",
    ], ids=["Series", "frame"])
    def test_a_truncated_named_index_is_read_on_both_sides_of_the_cut(self, stdout: str):
        assert _printed(stdout, answer="Year 1991 had 1 and year 2087 had 97.") == [
            ("1991", FigureOrigin.step), ("1", FigureOrigin.step),
            ("2087", FigureOrigin.step), ("97", FigureOrigin.step),
        ]

    @pytest.mark.parametrize(("stdout", "answer", "read"), [
        # The prints are pandas 3.0.6's, of frames wider than its 80-column
        # display, so it cuts columns out and marks them with "..." on every
        # line, the index name line included.
        # print(df.set_index('rank'))
        (
            "        sku product_type        supplier  ...  carrier  inspection defect_rate\n"
            "rank                                      ...                                 \n"
            "1     SKU51     haircare     Wise Supply  ...      DHL     Pending        0.41\n"
            "2     SKU38    cosmetics     Wise Supply  ...    FedEx     Pending        2.32\n"
            "3     SKU31     skincare  Organic Supply  ...    FedEx     Pending        1.12\n"
            "4     SKU90     skincare  Organic Supply  ...    FedEx        Pass        3.05\n"
            "5      SKU2     haircare       SupplyMax  ...    FedEx     Pending        0.27\n"
            "\n[5 rows x 8 columns]\n",
            "Rank 5 had a defect rate of 0.27.", ["5", "0.27"],
        ),
        # print(df.set_index(['store', 'rank']))
        (
            "              sku product_type        supplier  ...  lead_time  carrier inspection\n"
            "store rank                                      ...                               \n"
            "1     1     SKU51     haircare     Wise Supply  ...         18      DHL    Pending\n"
            "      2     SKU38    cosmetics     Wise Supply  ...         10    FedEx    Pending\n"
            "2     1     SKU31     skincare  Organic Supply  ...         24    FedEx    Pending\n"
            "      2     SKU90     skincare  Organic Supply  ...          8    FedEx       Pass\n"
            "3     1      SKU2     haircare       SupplyMax  ...         12    FedEx    Pending\n"
            "\n[5 rows x 7 columns]\n",
            "Store 3 had a lead time of 12 days.", ["3", "12"],
        ),
        # print(df.pivot_table(index='year', columns='sub_category', values='sales', aggfunc='sum'))
        (
            "sub_category  Acoustic Privacy Panel  ...  Smart Ergonomic Chair\n"
            "year                                  ...                       \n"
            "2024                       242540.11  ...              144700.00\n"
            "2025                       225538.21  ...              161280.85\n"
            "\n[2 rows x 4 columns]\n",
            "Chairs sold 161,280.85 in 2025, after 2024.", ["161,280.85", "2025", "2024"],
        ),
    ], ids=["set_index", "MultiIndex", "pivot_table"])
    def test_a_named_index_is_read_where_pandas_cut_columns_out(self, stdout: str, answer: str, read: list[str]):
        assert _printed(stdout, answer=answer) == [(text, FigureOrigin.step) for text in read]

    @pytest.mark.parametrize("stdout", [
        # The prints are pandas 3.0.6's.
        # print(df.pivot_table(index='year', columns='region', values='sales', aggfunc='sum').reset_index())
        "region  year    east    west\n0       2023  1200.0  3400.0\n1       2024  2600.0  2600.0\n"
        "2       2025   900.0     NaN\n",
        # print(df.pivot_table(index='year', columns='region', values=['sales', 'profit'],
        #                      aggfunc='sum').reset_index()): the columns' level names, unpadded,
        # under an indented header.
        "        year profit         sales        \nregion         east  west    east    west\n"
        "0       2023   10.0  20.0  1200.0  3400.0\n1       2024   30.0  40.0  2600.0  2600.0\n"
        "2       2025   50.0   NaN   900.0     NaN\n",
    ], ids=["one columns level", "two columns levels"])
    def test_row_numbers_under_a_named_columns_header_are_not_a_source(self, stdout: str):
        assert _printed(stdout, answer="East sold 2600 in 2024; 2 regions grew.") == [
            ("2600", FigureOrigin.step), ("2024", FigureOrigin.step), ("2", FigureOrigin.none),
        ]

    @pytest.mark.parametrize(("stdout", "answer", "origins"), [
        # The prints are pandas 3.0.6's: print(df.groupby('year')[['sales']].sum()), then a second frame.
        # print(pd.DataFrame({2023: [10, 20], 2024: [30, 40]}))
        (
            "      sales\nyear       \n2023   4600\n2024   5200\n   2023  2024\n0    10    30\n1    20    40\n",
            "Sales were 4600 in 2023; the rate is $1.",
            [("4600", FigureOrigin.step), ("2023", FigureOrigin.step), ("$1", FigureOrigin.none)],
        ),
        # print(pd.DataFrame({'p50': [11, 12], 'p90': [13, 14]})), whose header is as wide as the rows above it.
        (
            "      sales\nyear       \n2023   4600\n2024   5200\n   p50  p90\n0   11   13\n1   12   14\n",
            "Sales were 4600 in 2023; the rate is $1.",
            [("4600", FigureOrigin.step), ("2023", FigureOrigin.step), ("$1", FigureOrigin.none)],
        ),
        # print(df.groupby('store')[['units']].sum()), then print(df[['q3', 'units']]) of 10 rows, whose
        # header is as wide as the rows above it, indented past their one-digit keys and holds a digit.
        (
            "       units\nstore       \n1        162\n2        158\n   q3  units\n0  51     12\n1  69     23\n"
            "2  93     31\n3  88     29\n4  64     41\n5  52     32\n6  90     43\n7  81     48\n8  66     35\n"
            "9  95     26\n",
            "Store 1 sold 162 units; Q3 scores ranged across 7 stores.",
            [("1", FigureOrigin.step), ("162", FigureOrigin.step), ("7", FigureOrigin.none)],
        ),
        # print(df.groupby('store')[['units']].sum()), then print(df[['q3']]) of 10 floats, whose header is
        # as wide as the rows above it and starts a column right of where their value starts.
        (
            "       units\nstore       \n1        162\n2        158\n          q3\n0  51.234568\n1  69.512346\n"
            "2  93.751234\n3  88.012346\n4  64.125123\n5  52.501235\n6  90.250123\n7  81.750123\n8  66.501234\n"
            "9  95.001234\n",
            "Store 1 sold 162 units; Q3 scores came from 7 stores.",
            [("1", FigureOrigin.step), ("162", FigureOrigin.step), ("7", FigureOrigin.none)],
        ),
    ], ids=[
        "a header wider than the rows above", "a header as wide as the rows above", "a header past short keys",
        "a header a column right of a field",
    ])
    def test_a_frame_printed_after_a_named_index_keeps_its_own_row_numbers(
        self, stdout: str, answer: str, origins: list[tuple[str, FigureOrigin]],
    ):
        assert _printed(stdout, answer=answer) == origins

    def test_values_printed_one_space_before_a_negative_value_are_read(self):
        # print() puts one space between its arguments, as a frame row puts
        # before a negative first value, but these lines have no frame header.
        stdout = "2023 -5.2\n2024 -3.1\n2025 -0.4\n"

        assert _printed(
            stdout, answer="Sales fell in 2023 and in 2025.",
            code="for year, change in changes:\n    print(year, change)",
        ) == [("2023", FigureOrigin.step), ("2025", FigureOrigin.step)]

    @pytest.mark.parametrize("stdout", [
        # print(df.qty.value_counts())
        "qty\n250    2\n90     1\nName: count, dtype: int64\n",
        # print(df.set_index('qty'))
        "     part site\nqty          \n250  P100   S1\n90   P100   S2\n",
        # Printed by hand.
        "250  S1\n90   S2\n",
        # A named numeric index is data even when sorted ascending.
        "qty\n90     1\n250    2\nName: count, dtype: int64\n",
        "     part site\nqty          \n90   P100   S2\n250  P100   S1\n",
        # The same print with its trailing spaces stripped: a single name is read unpadded.
        "     part site\nqty\n90   P100   S2\n250  P100   S1\n",
        # pandas 3.0.6: print(df.q1.value_counts().sort_index()), a name with a digit.
        "q1\n40     1\n90     2\n250    3\nName: count, dtype: int64\n",
    ])
    def test_named_index_values_and_noncounting_labels_are_read(self, stdout: str):
        assert _printed(stdout, answer="S1 holds 250 units and S2 holds 90.", code="print(stock)") == [
            ("250", FigureOrigin.step), ("90", FigureOrigin.step),
        ]

    def test_a_single_named_index_value_is_read(self):
        assert _printed(
            "qty\n250    2\nName: count, dtype: int64\n",
            answer="There are 250 units.", code="print(stock)",
        ) == [("250", FigureOrigin.step)]


def _pdf_page(stdout: str) -> int | None:
    sources = find_figure_sources(
        answer="Overtime costs $42 an hour.", question="",
        steps=[_cell("toolu_pdf", code="print(pdf_text('rates.pdf'))", stdout=stdout)],
    )
    [source] = _figure(sources, "$42").sources
    return source.page


class TestPdfPages:
    def test_a_footer_closes_its_page(self):
        assert _pdf_page("Rates.\nPage 1 of 2\nOvertime costs $42 an hour.\nPage 2 of 2\n") == 2

    def test_a_figure_before_any_header_takes_the_footer_after_it(self):
        assert _pdf_page("Overtime costs $42 an hour.\nPage 1 of 2\nOther rates.\nPage 2 of 2\n") == 1

    def test_a_page_named_inside_a_line_is_not_a_marker(self):
        assert _pdf_page("--- page 1 ---\nSee page 12 for terms. Overtime costs $42 an hour.\n") == 1

    def test_an_exact_value_places_the_figure_before_a_rounded_one(self):
        assert _pdf_page("--- page 1 ---\nLast year: $41.60.\n--- page 2 ---\nOvertime costs $42 an hour.\n") == 2

    def test_pages_counted_from_zero_are_read_from_one(self):
        assert _pdf_page("--- page 0 ---\nRates.\n--- page 1 ---\nOvertime costs $42 an hour.\n") == 2
        assert _pdf_page("--- page 0 ---\nOvertime costs $42 an hour.\n") == 1


class TestOutOfRangeNumbers:
    @pytest.mark.parametrize("where", ["output", "answer", "question", "code"])
    def test_a_token_past_any_real_size_is_not_a_number_and_keeps_the_rest(self, where: str):
        token = "4E12345678"
        step = _cell(
            "toolu_read",
            code="print(open('stock.csv').read())" + (f"  # part {token}" if where == "code" else ""),
            stdout="part,site,qty\nP100,S1,250\n" + (f"{token},S9,7\n" if where == "output" else ""),
        )

        sources = find_figure_sources(
            answer="S1 holds 250 units." + (f" Part {token} is out." if where == "answer" else ""),
            question="Stock at S1?" + (f" Part {token}?" if where == "question" else ""),
            steps=[step],
        )

        assert _origins(sources) == [("250", FigureOrigin.step)]


class TestManyHolders:
    def test_a_figure_names_at_most_the_first_few_steps_in_step_order(self):
        steps = [_cell(f"toolu_{i}", code="print(v)", stdout="250\n") for i in range(MAX_FIGURE_SOURCES + 2)]

        sources = find_figure_sources(answer="S1 holds 250 units.", question="", steps=steps)

        expected = [f"toolu_{i}" for i in range(MAX_FIGURE_SOURCES)]
        assert [s.tool_use_id for s in _figure(sources, "250").sources] == expected
        assert [step.tool_use_id for step in sources.steps] == expected

    def test_exact_holders_are_kept_before_rounded_ones(self):
        printed = ["249.8", "250.2", "250", "249.9", "250"]
        steps = [_cell(f"toolu_{i}", code="print(v)", stdout=f"{value}\n") for i, value in enumerate(printed)]

        sources = find_figure_sources(answer="S1 holds 250 units.", question="", steps=steps)

        assert MAX_FIGURE_SOURCES == 3
        assert [s.tool_use_id for s in _figure(sources, "250").sources] == ["toolu_0", "toolu_2", "toolu_4"]

    def test_a_large_turn_names_every_holding_step_quickly(self, monkeypatch):
        printed: list[set[int]] = []
        steps = []
        for i in range(25):
            values = [(i * 37 + k * 11) % 2_000 for k in range(800)]
            printed.append(set(values))
            steps.append(_cell(f"toolu_{i}", code="print(rows)", stdout="\n".join(f"{v}, {v}.25" for v in values)))
        answer = "Values: " + ", ".join(str(v) for v in range(0, 2_000, 5)) + "."
        compared = 0
        same_value = figure_sources_mod._same_value

        def counted(*args, **kwargs):
            nonlocal compared
            compared += 1
            return same_value(*args, **kwargs)

        monkeypatch.setattr(figure_sources_mod, "_same_value", counted)
        start = time.perf_counter()

        sources = find_figure_sources(answer=answer, question="", steps=steps)

        # A scan of every output value per figure and step compares 400 x 25 x
        # 1,600 times; the sorted index compares none.
        assert compared < 400 * 25
        assert time.perf_counter() - start < TIME_BOUND_SECONDS
        assert len(sources.figures) == 400
        for figure in sources.figures:
            holders = [f"toolu_{i}" for i, values in enumerate(printed) if int(figure.text) in values]
            assert [s.tool_use_id for s in figure.sources] == holders[:MAX_FIGURE_SOURCES]


class TestSigns:
    def test_a_negative_figure_is_not_sourced_by_a_positive_value(self):
        assert _printed("160\n", answer="Projected balance at S2: -160 units.") == [("-160", FigureOrigin.none)]

    def test_a_minus_sign_reads_as_a_sign(self):
        assert _printed("160\n", answer="Projected balance at S2: −160 units.") == [
            ("−160", FigureOrigin.none),
        ]

    def test_a_negative_figure_is_sourced_by_a_negative_value(self):
        assert _printed("balance: -160\n", answer="Projected balance at S2: -160 units.") == [
            ("-160", FigureOrigin.step),
        ]

    def test_an_unsigned_figure_is_sourced_by_either_sign(self):
        assert _printed("balance: -160\n", answer="S2 is 160 units short.") == [("160", FigureOrigin.step)]


class TestCitations:
    def test_citation_markers_are_not_figures(self):
        answer = (
            "Steel trades at $640 a ton [1], up from $600 [^2].\n\n"
            "Sources:\n[1] https://example.com/steel\n[^2]: https://example.com/old"
        )

        assert [f.text for f in find_figure_sources(answer=answer, question="", steps=[]).figures] == ["$640", "$600"]


class TestModelWorkedFigures:
    def test_a_sum_the_model_works_out_in_its_text_has_no_source(self):
        sources = find_figure_sources(answer="S1 and S2 hold 340 units in all.", question="", steps=[READ_STOCK])

        assert _origins(sources) == [("340", FigureOrigin.none)]
        assert sources.steps == []


class TestDisplayMath:
    @pytest.mark.parametrize("math", [
        "$$250 + 90 = 340$$",
        "$$\n\\text{total} = 250 + 90\n$$",
        "\\[250 + 90\\]",
        "\\(250 + 90\\)",
    ])
    def test_numbers_in_display_math_are_not_figures(self, math: str):
        sources = find_figure_sources(answer=f"The sum {math} gives 77 sites.", question="", steps=[])

        assert [f.text for f in sources.figures] == ["77"]

    def test_a_single_dollar_sign_is_currency(self):
        sources = find_figure_sources(answer="Overtime costs $30 to $45 an hour.", question="", steps=[])

        assert [f.text for f in sources.figures] == ["$30", "$45"]

    @pytest.mark.parametrize("opener", ["\\[", "\\("])
    def test_math_left_open_runs_to_the_end_and_is_read_quickly(self, opener: str):
        text = opener * 49_152 + " S1 holds 250 units."
        start = time.perf_counter()

        sources = find_figure_sources(answer=text, question=text, steps=[READ_STOCK])

        assert time.perf_counter() - start < TIME_BOUND_SECONDS
        assert sources == FigureSources()


class TestUrlsInProse:
    @pytest.mark.parametrize(("printed", "url"), [
        ("Source: https://example.com/steel.", "https://example.com/steel"),
        ("See https://example.com/steel, which lists it.", "https://example.com/steel"),
        ("Source: **https://example.com/steel**", "https://example.com/steel"),
        ("Source: `https://example.com/steel`", "https://example.com/steel"),
        ("[Steel](https://example.com/steel)", "https://example.com/steel"),
        ("(see https://en.wikipedia.org/wiki/Steel_(alloy))", "https://en.wikipedia.org/wiki/Steel_(alloy)"),
    ])
    def test_a_url_a_web_search_printed_leaves_out_the_prose_around_it(self, printed: str, url: str):
        step = _cell("toolu_web", code="print(web_search('steel price'))", stdout=f"Steel trades at $640 a ton. {printed}\n")

        step_out = _only_step(step, answer="Steel trades at $640 a ton.")

        assert step_out.url == url

    def test_a_web_search_tools_url_leaves_out_the_prose_around_it(self):
        step = _tool("toolu_s", "web_search", {"query": "steel price"}, "Steel trades at $640 a ton (https://example.com/steel).")

        assert _only_step(step, answer="Steel trades at $640 a ton.").url == "https://example.com/steel"


class TestToleratedArguments:
    """anton runs a cell whose arguments it could read in a tolerant form,
    and its result still carries the code that ran."""

    @pytest.mark.parametrize("tool_input", ["", '{"action": "exec", "name": "main", "code": "cost = 30\\nprint(cost)"}<|end|>'])
    def test_a_cell_whose_arguments_do_not_parse_reads_the_code_that_ran(self, tool_input: str):
        cell = _cell("toolu_c", code="cost = 30\nprint(cost)", stdout="30\n")

        sources = find_figure_sources(
            answer="Overtime costs $30.", question="", steps=[replace(cell, tool_input=ToolInput.from_text(text=tool_input))],
        )

        assert _origins(sources) == [("$30", FigureOrigin.none)]

    def test_a_query_cell_whose_arguments_are_missing_keeps_its_sql(self):
        cell = _cell("toolu_q", code="print(query_minds_data('SELECT qty FROM stock'))", stdout="250\n")

        step_out = _only_step(replace(cell, tool_input=ToolInput.from_text(text="")), answer="S1 holds 250.")

        assert (step_out.kind, step_out.queries) == (SourceKind.query, ["SELECT qty FROM stock"])

    def test_arguments_cut_mid_code_keep_their_scratchpad(self):
        spell = _cell("toolu_1", code="sql = 'SELECT qty FROM stock'", stdout="", pad="calc")
        code = "# " + "x" * 200 + "\ncur.execute(sql)\nprint(cur.fetchone()[0])"
        run = _cell("toolu_2", code=code, stdout="250\n", pad="calc")
        cut = replace(run, tool_input=ToolInput.from_text(text=run.tool_input.text[:120]))

        [step] = find_figure_sources(answer="S1 holds 250.", question="", steps=[spell, cut]).steps

        assert step.queries == ["SELECT qty FROM stock"]


# A call's arguments and what the matcher reads from them: the scratchpad's
# name and code, a web call's url and query, and a file read's path.
_READ_ARGUMENTS = [
    pytest.param(
        {"action": "exec", "name": "main", "code": "print(1)"}, ToolArguments(name="main", code="print(1)"),
        id="a-cell",
    ),
    pytest.param({"name": 7, "code": 5}, ToolArguments(name="7", code="5"), id="numbers-read-as-text"),
    pytest.param({"name": None, "code": None}, ToolArguments(), id="nulls-read-as-empty"),
    pytest.param({"name": False, "query": 0}, ToolArguments(), id="falsy-values-read-as-empty"),
    pytest.param({"path": 5}, ToolArguments(), id="a-path-that-is-not-text"),
    pytest.param({"path": ""}, ToolArguments(), id="an-empty-path"),
    pytest.param({"path": "/workspace/a.csv"}, ToolArguments(path="/workspace/a.csv"), id="a-path"),
    pytest.param({"url": ["https://a.example"]}, ToolArguments(url="['https://a.example']"), id="a-url-list"),
    pytest.param({"url": {"href": "x"}}, ToolArguments(url="{'href': 'x'}"), id="a-url-object"),
    pytest.param({"url": None, "query": "rates"}, ToolArguments(query="rates"), id="a-search"),
    pytest.param({"query": 1e16}, ToolArguments(query="1e+16"), id="a-large-float"),
    pytest.param({}, ToolArguments(), id="no-arguments"),
]


class TestToolArguments:
    """A call's arguments read the same whether the matcher gets them as the
    model streamed them or as a history row holds them."""

    @pytest.mark.parametrize(("arguments", "read"), _READ_ARGUMENTS)
    def test_a_call_reads_the_same_from_its_dict_and_its_text(self, arguments: dict, read: ToolArguments):
        assert ToolArguments.from_call(arguments=arguments) == read
        assert ToolArguments.from_text(text=json.dumps(arguments, default=str)) == read

    @pytest.mark.parametrize(("text", "read"), [
        pytest.param("", ToolArguments(), id="empty"),
        pytest.param("{}", ToolArguments(), id="an-empty-object"),
        pytest.param("[1]", ToolArguments(), id="a-list"),
        pytest.param("null", ToolArguments(), id="null"),
        pytest.param(
            '{"action": "exec", "name": "ma\\u00efn", "co', ToolArguments(name="ma\u00efn"),
            id="cut-short-after-an-escaped-name",
        ),
        pytest.param(
            '{"action": "exec", "name": "main", "code": "x = 1"}<|end|>', ToolArguments(name="main", code="x = 1"),
            id="a-trailing-end-token",
        ),
        pytest.param('{"name": "main"} and more', ToolArguments(name="main"), id="trailing-text"),
        pytest.param(
            '{"name": "main", "code": "print(1)"', ToolArguments(name="main"), id="an-unterminated-object",
        ),
        # The escape does not read, so the name reads as no name at all.
        pytest.param(
            '{"action": "exec", "name": "a\\qb", "code": "print(1', ToolArguments(), id="cut-short-with-a-bad-escape",
        ),
        pytest.param('[{"name": "x"}]', ToolArguments(name="x"), id="a-name-inside-a-list"),
    ])
    def test_arguments_that_do_not_read_as_an_object_keep_only_their_name(self, text: str, read: ToolArguments):
        assert ToolArguments.from_text(text=text) == read

    def test_a_path_object_reads_as_the_path_it_names(self):
        """A row's arguments are read as they are, not through JSON text, so
        a path given as a path object still names the file."""
        call = {"path": PurePosixPath("/workspace/a.csv")}

        assert ToolArguments.from_call(arguments=call).path == "/workspace/a.csv"
        assert ToolArguments.from_call(arguments=call) == ToolArguments.from_text(text=json.dumps(call, default=str))

    async def test_arguments_nested_a_few_hundred_deep_read_whole(self):
        nested = '{"name": "main", "code": "x = 1", "deps": ' + "[" * 500 + "]" * 500 + "}"

        assert await asyncio.to_thread(ToolArguments.from_text, text=nested) == ToolArguments(name="main", code="x = 1")

    async def test_arguments_nested_too_deep_to_parse_read_as_cut_short(self):
        """Read in a worker thread, as find_answer_sources reads them. The
        parser gives up with a RecursionError long before 100,000 levels."""
        nested = '{"name": "main", "code": "x = 1", "deps": ' + "[" * 100_000 + "]" * 100_000 + "}"

        assert await asyncio.to_thread(ToolArguments.from_text, text=nested) == ToolArguments(name="main")


class TestClippedResults:
    """The cloud pod cuts a cell result that is still past its wire limit
    after shortening each field, which leaves it no longer JSON."""

    def test_what_the_cell_printed_is_read_and_its_error_is_not(self):
        cell = Cell(
            code="print(total)", stdout="Total units:\n7777\n€1,250\n", stderr="",
            error="Traceback: retry 4242 failed " + "E" * 70_000, description="Sum stock",
        )
        clipped = _clip_result_content(json.dumps(asdict(cell)))
        step = TurnStep(
            tool_use_id="toolu_c", tool_name="scratchpad", action="exec",
            tool_input=ToolInput.from_text(text=json.dumps({"action": "exec", "name": "main", "code": "print(total)"})),
            content=clipped,
        )

        sources = find_figure_sources(answer="7777 units worth €1,250; 4242 retries.", question="", steps=[step])

        assert _origins(sources) == [
            ("7777", FigureOrigin.step), ("€1,250", FigureOrigin.step), ("4242", FigureOrigin.none),
        ]

    def test_printed_text_cut_in_the_middle_keeps_what_came_before_the_cut(self):
        cell = Cell(code="print(total)", stdout="7777\n" + "€" * 16_000, stderr="", error=None, description="")
        clipped = _clip_result_content(json.dumps(asdict(cell)))
        step = TurnStep(
            tool_use_id="toolu_c", tool_name="scratchpad", action="exec",
            tool_input=ToolInput.from_text(text=json.dumps({"action": "exec", "name": "main", "code": "print(total)"})),
            content=clipped,
        )

        assert _origins(find_figure_sources(answer="7777 units.", question="", steps=[step])) == [
            ("7777", FigureOrigin.step),
        ]


class TestCellResults:
    def test_a_result_without_printed_output_is_logged_and_holds_nothing(self, caplog):
        step = replace(
            _cell("toolu_c", code="print(total)", stdout="250\n"),
            content=json.dumps({"code": "print(total)", "output": "250\n"}),
        )

        with caplog.at_level(logging.WARNING, logger=figure_sources_mod.__name__):
            sources = find_figure_sources(answer="S1 holds 250.", question="", steps=[step])

        assert _origins(sources) == [("250", FigureOrigin.none)]
        assert len(caplog.records) == 1
