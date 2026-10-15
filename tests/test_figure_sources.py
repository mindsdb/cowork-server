from __future__ import annotations

import json
from pathlib import Path

import pytest

from cowork.harnesses.anton_harness.agent_files import AgentFiles
from cowork.harnesses.anton_harness.figure_sources import (
    MAX_FILE_LITERALS,
    MAX_QUERY_CHARS,
    Figure,
    FigureOrigin,
    FigureSource,
    FigureSources,
    FigureStep,
    SourceKind,
    TurnStep,
    find_figure_sources,
    question_text,
)


def _cell(tool_use_id: str, *, code: str, stdout: str, description: str = "", pad: str = "main") -> TurnStep:
    return TurnStep(
        tool_use_id=tool_use_id,
        tool_name="scratchpad",
        action="exec",
        tool_input=json.dumps({"action": "exec", "name": pad, "code": code}),
        content=json.dumps({
            "code": code, "stdout": stdout, "stderr": "", "error": None,
            "description": description, "estimated_time": "", "logs": "",
        }),
    )


def _tool(tool_use_id: str, name: str, tool_input: dict, content: str) -> TurnStep:
    return TurnStep(
        tool_use_id=tool_use_id, tool_name=name, action="", tool_input=json.dumps(tool_input), content=content,
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
            tool_use_id="toolu_c", kind=SourceKind.step, label="a calculation",
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
            tool_use_id="toolu_b", kind=SourceKind.query, label="a query", requests=["GET {DS_ERP_URL}/inventory"],
        )

    def test_a_printed_label_that_reads_like_sql_leaves_the_cell_a_file_step(self):
        step = _cell(
            "toolu_f",
            code='df = pd.read_csv("stock.csv"); print("With rows from stock.csv:", int(df.qty.sum()))',
            stdout="With rows from stock.csv: 250",
        )

        step_out = _only_step(step, answer="250")

        assert (step_out.kind, step_out.label, step_out.queries) == (SourceKind.file, "stock.csv", [])

    def test_a_recall_is_memory_and_a_tool_reading_a_path_is_a_file(self):
        recall = _tool("toolu_r", "recall", {"query": "overtime"}, "Overtime costs 42 an hour")
        read = _tool("toolu_p", "read_file", {"path": "docs/rates.md"}, "Line 7: 17 units")

        sources = find_figure_sources(answer="Overtime costs $42; Line 7 makes 17.", question="", steps=[recall, read])

        assert [(s.kind, s.label) for s in sources.steps] == [
            (SourceKind.memory, "memory"), (SourceKind.file, "rates.md"),
        ]


class TestQueriesAcrossCells:
    def test_a_query_an_earlier_cell_spelled_is_the_later_cells(self):
        spell = _cell("toolu_1", code="sql = \"SELECT qty FROM stock WHERE site = 'S1'\"", stdout="")
        run = _cell("toolu_2", code="print(query_minds_data(3, sql))", stdout="250\n")

        step = find_figure_sources(answer="S1 holds 250.", question="", steps=[spell, run]).steps[0]

        assert (step.tool_use_id, step.queries) == ("toolu_2", ["SELECT qty FROM stock WHERE site = 'S1'"])

    def test_a_reset_scratchpad_forgets_the_strings_its_cells_bound(self):
        spell = _cell("toolu_1", code="sql = 'SELECT qty FROM stock'", stdout="")
        reset = TurnStep(
            tool_use_id="toolu_reset", tool_name="scratchpad", action="reset",
            tool_input=json.dumps({"action": "reset", "name": "main"}), content="",
        )
        run = _cell("toolu_2", code="cur.execute(sql)\nprint(cur.fetchone()[0])", stdout="250\n")

        step = find_figure_sources(answer="S1 holds 250.", question="", steps=[spell, reset, run]).steps[0]

        assert step.queries == []


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

        assert (step_out.kind, step_out.label, step_out.path) == (SourceKind.step, "a calculation", None)

    def test_a_calculation_that_saves_its_result_is_not_from_that_file(self, tmp_path: Path):
        (tmp_path / "notes.md").write_text("9002")
        step = _cell(
            "toolu_c", code='total = a * b; open("notes.md", "w").write(str(total)); print(total)', stdout="9002\n",
        )

        step_out = _only_step(step, answer="The total is 9002.", files=AgentFiles.on_this_server(tmp_path))

        assert (step_out.kind, step_out.label, step_out.path) == (SourceKind.step, "a calculation", None)

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

    def test_only_the_first_few_document_strings_are_looked_up(self, tmp_path: Path, monkeypatch):
        (tmp_path / "stock.csv").write_text("qty\n250\n")
        names = [f"d/f{i}.csv" for i in range(MAX_FILE_LITERALS + 2)] + ["stock.csv"]
        step = _cell(
            "toolu_many", code=f"for name in {names!r}:\n    print(pd.read_csv(name).qty.sum())", stdout="250\n",
        )
        looked_up: list[str] = []
        project_path = AgentFiles.project_path

        def spy(self: AgentFiles, literal: str) -> str | None:
            looked_up.append(literal)
            return project_path(self, literal)

        monkeypatch.setattr(AgentFiles, "project_path", spy)

        step_out = _only_step(step, answer="S1 holds 250.", files=AgentFiles.on_this_server(tmp_path))

        assert len(looked_up) <= MAX_FILE_LITERALS
        assert step_out.path is None


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
            "WITH s AS (SELECT * FROM mysql_scan('host=db user=erp password=[redacted], 'stock')) SELECT qty FROM s",
            "SELECT qty FROM read_csv('https://data.example.com/stock.csv?api_key=[redacted]&region=eu')",
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

    def test_a_memorize_call_is_not_a_source(self):
        memorize = _tool("toolu_mem", "memorize", {"text": "Overtime costs 30 dollars"}, "Saved: Overtime costs 30 dollars")

        sources = find_figure_sources(answer="Overtime costs $30.", question="", steps=[memorize])

        assert _figure(sources, "$30").origin is FigureOrigin.none

    def test_a_non_exec_scratchpad_action_is_not_a_source(self):
        dump = TurnStep(tool_use_id="toolu_dump", tool_name="scratchpad", action="dump", tool_input="{}", content="250")

        sources = find_figure_sources(answer="S1 holds 250 units.", question="", steps=[dump])

        assert _figure(sources, "250").origin is FigureOrigin.none

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
