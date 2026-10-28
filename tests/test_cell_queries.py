from __future__ import annotations

import random
import re
import textwrap
import time
from itertools import pairwise

import pytest

from cowork.harnesses.anton_harness import cell_queries as cell_queries_mod
from cowork.harnesses.anton_harness.cell_queries import (
    MAX_CLIENTS,
    MAX_FUNCTION_GLOBALS,
    MAX_RENDERED_CHARS,
    MAX_STRING_CHARS,
    FunctionReading,
    HttpRequest,
    is_sql,
    read_cell,
    strip_url,
)

# The current code reads each timed input in about a second or less on a
# laptop under coverage, so a CI runner several times slower still passes; the
# slow form each test guards against takes 15 s or more on the same laptop.
TIME_BOUND_SECONDS = 5.0


def _queries(code: str, **names: str) -> list[str]:
    return list(read_cell(textwrap.dedent(code), names=names).queries)


def _requests(code: str, **names: str) -> list[str]:
    return [str(request) for request in read_cell(textwrap.dedent(code), names=names).requests]


def _strings(code: str, **names: str) -> tuple[str, ...]:
    return read_cell(textwrap.dedent(code), names=names).strings


class TestQueriesACellRuns:
    def test_the_desktop_data_helper_runs_its_first_argument(self):
        assert _queries("""
            a = query_minds_data("SELECT qty FROM stock", datasource="erp")
            b = query_minds_data(query="SELECT site FROM sites")
        """) == ["SELECT qty FROM stock", "SELECT site FROM sites"]

    def test_the_cloud_data_helper_runs_its_second_argument(self):
        assert _queries("""
            a = query_minds_data(3, "SELECT qty FROM stock")
            b = query_minds_data(connection_id=3, sql="SELECT site FROM sites", parameters=None)
        """) == ["SELECT qty FROM stock", "SELECT site FROM sites"]

    def test_a_cursor_read_sql_and_sql_calls_run_their_first_argument(self):
        assert _queries("""
            cur.execute("SELECT qty FROM stock WHERE site = %s", ("S1",))
            df = pd.read_sql(text("SELECT site FROM sites"), engine)
            duckdb.sql("WITH t AS (SELECT 1 AS n) SELECT n FROM t").show()
        """) == [
            "SELECT qty FROM stock WHERE site = %s",
            "SELECT site FROM sites",
            "WITH t AS (SELECT 1 AS n) SELECT n FROM t",
        ]

    def test_an_fstring_keeps_its_placeholders_as_written(self):
        assert _queries("""
            region = input()
            df = pd.read_sql(f"SELECT qty FROM stock WHERE region = '{region}' AND day > {start:%Y}", conn)
        """) == ["SELECT qty FROM stock WHERE region = '{region}' AND day > {start:%Y}"]

    def test_a_multi_line_query_is_dedented_and_stripped(self):
        assert _queries('''
            rows = query_minds_data(3, """
                SELECT qty
                FROM stock
            """)
        ''') == ["SELECT qty\nFROM stock"]

    def test_a_query_built_by_concatenation_or_format_keeps_its_template(self):
        assert _queries("""
            cur.execute("SELECT qty FROM stock WHERE site = " + site_id)
            cur.execute("SELECT {cols} FROM stock".format(cols="qty"))
            cur.execute(dedent("SELECT site FROM sites"))
        """) == [
            "SELECT qty FROM stock WHERE site = {site_id}",
            "SELECT {cols} FROM stock",
            "SELECT site FROM sites",
        ]

    def test_a_query_a_name_holds_is_run_through_that_name(self):
        assert _queries("""
            sql = "SELECT qty FROM stock"
            sql += " WHERE site = 'S1'"
            cur.execute(sql)
        """) == ["SELECT qty FROM stock WHERE site = 'S1'"]

    def test_a_base_query_that_plus_extends_is_not_run_and_its_extensions_are(self):
        assert _queries("""
            base = "SELECT qty FROM stock"
            s1 = query_minds_data(base + " WHERE site='S1'")
            s2 = query_minds_data(base + " WHERE site='S2'")
            s3 = query_minds_data(base + " WHERE site='S3'")
        """) == [
            "SELECT qty FROM stock WHERE site='S1'",
            "SELECT qty FROM stock WHERE site='S2'",
            "SELECT qty FROM stock WHERE site='S3'",
        ]

    def test_an_fstring_on_a_bound_base_query_reads_as_the_statement_that_ran(self):
        assert _queries("""
            base = "SELECT site, qty FROM stock"
            for s in sites:
                rows = query_minds_data(f"{base} WHERE site = '{s}'")
        """) == ["SELECT site, qty FROM stock WHERE site = '{s}'"]

    def test_a_base_query_also_run_on_its_own_stays_listed(self):
        assert _queries("""
            base = "SELECT qty FROM stock"
            total = query_minds_data(base)
            s1 = query_minds_data(base + " WHERE site = 'S1'")
        """) == ["SELECT qty FROM stock", "SELECT qty FROM stock WHERE site = 'S1'"]

    def test_a_query_another_starts_with_stays_listed(self):
        assert _queries("""
            now = cur.execute("SELECT qty FROM stock")
            then = cur.execute("SELECT qty FROM stock_history")
        """) == ["SELECT qty FROM stock", "SELECT qty FROM stock_history"]

    def test_a_query_inside_a_bound_fstrings_placeholder_runs_where_it_is_bound(self):
        assert _queries("""
            total = f"{query_minds_data('SELECT SUM(qty) FROM stock').iloc[0, 0]}"
        """) == ["SELECT SUM(qty) FROM stock"]

    def test_a_recursive_or_quoted_cte_is_a_query(self):
        assert _queries("""
            a = con.sql("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT n FROM r")
            b = con.sql('WITH "daily stock" AS (SELECT qty FROM stock) SELECT SUM(qty) FROM "daily stock"')
        """) == [
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT n FROM r",
            'WITH "daily stock" AS (SELECT qty FROM stock) SELECT SUM(qty) FROM "daily stock"',
        ]

    def test_a_query_that_ends_in_a_parameter_marker_is_a_query(self):
        assert _queries("""
            cur.execute("SELECT qty FROM stock WHERE site = ?", ("S1",))
            cur.execute("SELECT qty FROM stock WHERE site=?", ("S2",))
        """) == ["SELECT qty FROM stock WHERE site = ?", "SELECT qty FROM stock WHERE site=?"]

    def test_a_query_an_earlier_cell_bound_is_run_by_this_one(self):
        earlier = read_cell("sql = \"SELECT qty FROM stock WHERE site = 'S1'\"", names={})

        later = read_cell("rows = query_minds_data(3, sql)", names=earlier.names)

        assert later.queries == ("SELECT qty FROM stock WHERE site = 'S1'",)
        assert later.calls_query_minds_data

    def test_a_name_rebound_to_something_else_no_longer_holds_its_query(self):
        reading = read_cell("sql = build_query()\nfor site in sites: pass", names={"sql": "SELECT 1 FROM t", "site": "x"})

        assert reading.names == {}

    def test_a_query_built_with_str_join_reads_as_the_statement_that_ran(self):
        assert _queries('''
            base = "SELECT qty FROM stock"
            sql = " ".join([base, "WHERE site = 'S1'"])
            a = query_minds_data(sql)
            b = query_minds_data("\\n".join(("SELECT site", "FROM sites")))
        ''') == ["SELECT qty FROM stock WHERE site = 'S1'", "SELECT site\nFROM sites"]

    def test_a_percent_formatted_query_keeps_its_fields_as_placeholders(self):
        assert _queries("""
            base = "SELECT qty FROM stock"
            a = cur.execute("SELECT qty FROM stock WHERE site = '%s'" % site)
            b = cur.execute("%s WHERE site = '%s' AND pct > %d%%" % (base, site, limit))
        """) == [
            "SELECT qty FROM stock WHERE site = '{site}'",
            "SELECT qty FROM stock WHERE site = '{site}' AND pct > {limit}%",
        ]


class TestBranches:
    def test_sql_bound_in_if_and_else_lists_both_statements(self):
        assert _queries("""
            if site:
                sql = "SELECT qty FROM stock WHERE site = 'S1'"
            else:
                sql = "SELECT qty FROM stock"
            rows = query_minds_data(sql)
        """) == ["SELECT qty FROM stock WHERE site = 'S1'", "SELECT qty FROM stock"]

    def test_a_default_an_if_overrides_stays_listed(self):
        assert _queries("""
            sql = "SELECT qty FROM stock"
            if site:
                sql = "SELECT qty FROM stock WHERE site = 'S1'"
            rows = query_minds_data(sql)
        """) == ["SELECT qty FROM stock", "SELECT qty FROM stock WHERE site = 'S1'"]

    def test_a_conditional_extension_keeps_the_statement_it_extends(self):
        assert _queries("""
            sql = "SELECT qty FROM stock"
            if site:
                sql += " WHERE site = 'S1'"
            rows = query_minds_data(sql)
        """) == ["SELECT qty FROM stock", "SELECT qty FROM stock WHERE site = 'S1'"]

    def test_a_handler_that_rebinds_keeps_what_the_try_bound(self):
        assert _queries("""
            try:
                sql = "SELECT qty FROM stock_live"
            except Exception:
                sql = "SELECT qty FROM stock_cache"
            rows = query_minds_data(sql)
        """) == ["SELECT qty FROM stock_live", "SELECT qty FROM stock_cache"]

    def test_a_read_inside_the_branch_that_rebound_the_name_sees_only_its_statement(self):
        assert _queries("""
            sql = "SELECT qty FROM stock"
            if site:
                sql += " WHERE site = 'S1'"
                rows = query_minds_data(sql)
        """) == ["SELECT qty FROM stock WHERE site = 'S1'"]


class TestScopes:
    def test_a_parameter_does_not_read_as_an_earlier_cells_string(self):
        earlier = read_cell('sql = "SELECT part, site, qty FROM stock"', names={})

        later = read_cell(textwrap.dedent("""
            def q(sql):
                return query_minds_data(sql)
            rows = q("SELECT qty FROM stock WHERE site = 'S1'")
        """), names=earlier.names)

        assert later.queries == ("SELECT qty FROM stock WHERE site = 'S1'",)
        assert later.names == earlier.names

    def test_a_functions_locals_do_not_hide_the_cells_names(self):
        reading = read_cell(textwrap.dedent("""
            def by_site(site):
                sql = f"SELECT qty FROM stock WHERE site = '{site}'"
                return query_minds_data(sql)
            s1 = by_site("S1")
            total = query_minds_data(sql)
        """), names={"sql": "SELECT SUM(qty) FROM stock"})

        assert reading.queries == ("SELECT qty FROM stock WHERE site = '{site}'", "SELECT SUM(qty) FROM stock")
        assert reading.names == {"sql": "SELECT SUM(qty) FROM stock"}

    def test_a_lambda_parameter_does_not_read_as_the_cells_string(self):
        reading = read_cell("run = lambda sql: query_minds_data(sql)", names={"sql": "SELECT SUM(qty) FROM stock"})

        assert reading.queries == ()

    def test_a_comprehension_target_stays_inside_the_comprehension(self):
        reading = read_cell(textwrap.dedent("""
            rows = [query_minds_data(sql) for sql in statements]
            total = query_minds_data(sql)
        """), names={"sql": "SELECT SUM(qty) FROM stock"})

        assert reading.queries == ("SELECT SUM(qty) FROM stock",)
        assert reading.names == {"sql": "SELECT SUM(qty) FROM stock"}


class TestHelpersFromEarlierCells:
    def test_a_helper_an_earlier_cell_defined_brings_its_request_to_the_call(self):
        earlier = read_cell(textwrap.dedent("""
            BASE = "https://erp.example.com/api"
            def get(path):
                return requests.get(BASE + path).json()
        """), names={})

        later = read_cell('rows = get("/stock?site=S1")', names=earlier.names, functions=earlier.functions)

        assert [str(request) for request in later.requests] == ["GET https://erp.example.com/api{path}"]

    def test_a_helper_brings_its_query_and_its_connection_reads_to_the_call(self):
        earlier = read_cell(textwrap.dedent("""
            def stock(site):
                token = os.environ["DS_ERP_TOKEN"]
                return query_minds_data(f"SELECT qty FROM stock WHERE site = '{site}'")
        """), names={})

        later = read_cell('s1 = stock("S1")', names=earlier.names, functions=earlier.functions)

        assert later.queries == ("SELECT qty FROM stock WHERE site = '{site}'",)
        assert later.calls_query_minds_data
        assert later.environment_variables == ("DS_ERP_TOKEN",)
        assert earlier.functions["stock"] == FunctionReading(
            queries=("SELECT qty FROM stock WHERE site = '{site}'",),
            strings=("DS_ERP_TOKEN",),
            environment_variables=("DS_ERP_TOKEN",),
            calls_query_minds_data=True,
            reads_data=True,
            global_names=("os", "query_minds_data"),
        )

    def test_a_name_that_hides_a_helper_does_not_run_it(self):
        functions = read_cell(
            "def get(path):\n    return requests.get('https://erp.example.com' + path)", names={},
        ).functions

        reading = read_cell(textwrap.dedent("""
            def fetch(get):
                return get("/stock")
            get = None
            get("/sites")
        """), names={}, functions=functions)

        assert reading.requests == ()
        assert "get" not in reading.functions
        assert "fetch" in reading.functions

    @pytest.mark.parametrize("expression", ["sql", 'f"{sql}"', 'sql + " WHERE qty > 0"'])
    @pytest.mark.parametrize("same_cell", [False, True], ids=["across-cells", "same-cell"])
    def test_rebinding_a_helpers_global_omits_its_frozen_query(self, expression: str, same_cell: bool):
        definition = f"sql = 'SELECT qty FROM stock'\ndef fetch():\n    return query_minds_data({expression})\n"
        call = "sql = 'SELECT qty FROM deliveries'\nprint(fetch())"
        if same_cell:
            reading = read_cell(definition + call, names={})
        else:
            earlier = read_cell(definition, names={})
            reading = read_cell(call, names=earlier.names, functions=earlier.functions)

        assert reading.queries == ()
        assert reading.functions["fetch"].queries == ()
        assert reading.calls_query_minds_data

    @pytest.mark.parametrize("same_cell", [False, True], ids=["across-cells", "same-cell"])
    def test_rebinding_a_helpers_global_omits_its_frozen_request(self, same_cell: bool):
        definition = textwrap.dedent('''
            BASE = "https://stock.example.com"
            def fetch():
                return requests.get(BASE + "/inventory")
        ''')
        call = 'BASE = "https://deliveries.example.com"\nprint(fetch())'
        if same_cell:
            reading = read_cell(definition + call, names={})
        else:
            earlier = read_cell(definition, names={})
            reading = read_cell(call, names=earlier.names, functions=earlier.functions)

        assert reading.requests == ()
        assert reading.functions["fetch"].requests == ()
        assert "https://stock.example.com/inventory" not in reading.strings

    @pytest.mark.parametrize("rebind", ["del sql", "sql = None", "if changed:\n    sql = 'SELECT qty FROM deliveries'"])
    def test_deletes_unknown_values_and_conditional_rebindings_invalidate_helpers(self, rebind: str):
        earlier = read_cell("sql = 'SELECT qty FROM stock'\ndef fetch():\n    return query_minds_data(sql)", names={})

        changed = read_cell(rebind, names=earlier.names, functions=earlier.functions)
        later = read_cell("fetch()", names=changed.names, functions=changed.functions)

        assert later.queries == ()

    def test_a_stable_global_and_an_unrelated_local_rebinding_keep_the_helpers_details(self):
        earlier = read_cell(textwrap.dedent('''
            sql = "SELECT qty FROM stock"
            def fetch():
                return query_minds_data(sql)
        '''), names={})

        later = read_cell(textwrap.dedent('''
            unrelated = 1
            def other(sql):
                sql = "SELECT qty FROM deliveries"
                return sql
            print(fetch())
        '''), names=earlier.names, functions=earlier.functions)

        assert later.queries == ("SELECT qty FROM stock",)

    def test_a_helpers_own_local_query_does_not_depend_on_a_global_with_the_same_name(self):
        earlier = read_cell(textwrap.dedent('''
            sql = "SELECT qty FROM deliveries"
            def fetch():
                sql = "SELECT qty FROM stock"
                return query_minds_data(sql)
        '''), names={})

        later = read_cell("sql = None\nfetch()", names=earlier.names, functions=earlier.functions)

        assert later.queries == ("SELECT qty FROM stock",)

    @pytest.mark.parametrize("rebind", ["sql = 'SELECT qty FROM deliveries'", "fetch = replacement"])
    def test_a_wrapper_also_depends_on_the_wrapped_helpers_globals(self, rebind: str):
        earlier = read_cell(textwrap.dedent('''
            sql = "SELECT qty FROM stock"
            def fetch():
                return query_minds_data(sql)
            def wrapped():
                sql = "local shadow"
                return fetch()
        '''), names={})

        later = read_cell(rebind + "\nwrapped()", names=earlier.names, functions=earlier.functions)

        assert later.queries == ()

    def test_queries_called_before_a_rebind_are_kept(self):
        reading = read_cell(textwrap.dedent('''
            sql = "SELECT qty FROM stock"
            def fetch():
                return query_minds_data(sql)
            first = fetch()
            sql = "SELECT qty FROM deliveries"
            second = fetch()
        '''), names={})

        assert reading.queries == ("SELECT qty FROM stock",)
        assert reading.functions["fetch"].queries == ()

    def test_an_uncalled_body_emits_no_query_or_request_details(self):
        reading = read_cell(textwrap.dedent('''
            def fetch():
                return query_minds_data("SELECT qty FROM stock"), requests.get("https://stock.example.com")
        '''), names={})

        assert reading.queries == ()
        assert reading.requests == ()
        # Static source literals still describe the defining cell in a
        # later caller's lineage; they do not report a request being made.
        assert reading.strings == ("https://stock.example.com",)
        assert not reading.calls_query_minds_data
        assert reading.functions["fetch"].queries == ("SELECT qty FROM stock",)

    def test_a_called_body_keeps_file_details_but_not_printed_or_written_paths(self):
        code = textwrap.dedent('''
            def fetch():
                print("label.csv")
                df = pd.read_csv("stock.csv")
                df.to_csv("output.csv")
                return df
        ''')
        earlier = read_cell(code, names={})

        same_cell = read_cell(code + "fetch()", names={})
        later_cell = read_cell("fetch()", names=earlier.names, functions=earlier.functions)

        assert earlier.strings == ("stock.csv",)
        for called in (same_cell, later_cell):
            assert called.strings == ("stock.csv", "label.csv")
            assert called.labels == ("label.csv",)
            assert tuple(text for text in called.strings if text not in called.labels) == ("stock.csv",)

    def test_a_helpers_defining_cell_keeps_the_literal_file_for_later_lineage(self):
        definition = read_cell("def load():\n    return pd.read_csv('stock.csv')", names={})

        call = read_cell("print(load().qty.sum())", names=definition.names, functions=definition.functions)

        assert definition.strings == ("stock.csv",)
        assert definition.bindings["load"].from_read
        assert call.strings == ("stock.csv",)
        assert "load" in call.loaded_names

    def test_a_helpers_defining_cell_does_not_retain_a_rendered_global_path(self):
        reading = read_cell("def load():\n    return pd.read_csv(path)", names={"path": "stock.csv"})

        assert reading.strings == ()
        assert reading.functions["load"].strings == ("stock.csv",)

    def test_a_rebound_file_global_does_not_leave_its_old_document_label(self):
        earlier = read_cell("path = 'stock.csv'\ndef load():\n    return pd.read_csv(path)", names={})

        later = read_cell("path = 'deliveries.csv'\nload()", names=earlier.names, functions=earlier.functions)

        assert "stock.csv" not in later.strings

    def test_a_closure_does_not_freeze_an_enclosing_local(self):
        reading = read_cell(textwrap.dedent('''
            def outer():
                sql = "SELECT qty FROM stock"
                def inner():
                    return query_minds_data(sql)
                sql = "SELECT qty FROM deliveries"
                return inner()
            outer()
        '''), names={})

        assert reading.queries == ()

    def test_function_dependencies_are_bounded_without_untracked_frozen_details(self):
        loads = "\n".join(f"    print(global_{index})" for index in range(MAX_FUNCTION_GLOBALS + 1))
        code = "def fetch():\n" + loads + "\n    return query_minds_data('SELECT qty FROM stock')\nfetch()"

        reading = read_cell(code, names={})

        assert len(reading.functions["fetch"].global_names) <= MAX_FUNCTION_GLOBALS
        assert reading.queries == ()
        assert reading.calls_query_minds_data


class TestNamesFromReads:
    @pytest.mark.parametrize("value", [
        'pd.read_csv("stock.csv")',
        'pd.read_excel("stock.xlsx", sheet_name="S1")',
        'pd.read_parquet("stock.parquet")',
        'pd.read_json("stock.json")',
        'pd.read_sql("SELECT qty FROM stock", conn)',
        'open("stock.csv").read()',
        'open("stock.csv", "r")',
        'Path("notes.md").read_text()',
        'query_minds_data("SELECT qty FROM stock")',
        'cur.fetchall()',
        'con.execute("SELECT qty FROM stock").df()',
        'requests.get("https://erp.example.com/api/stock").json()',
        'web_search("steel price")',
    ])
    def test_a_name_bound_from_a_read_is_from_a_read(self, value: str):
        assert read_cell(f"data = {value}", names={}).bindings["data"].from_read

    @pytest.mark.parametrize("value", [
        'pd.DataFrame({"qty": [1]})',
        'open("out.csv", "w")',
        '"stock.csv"',
        "len(rows)",
    ])
    def test_a_name_bound_from_anything_else_is_not(self, value: str):
        assert not read_cell(f"data = {value}", names={}).bindings["data"].from_read

    def test_a_name_computed_from_a_name_the_cell_read_is_from_a_read(self):
        reading = read_cell("df = pd.read_csv('stock.csv')\ntop = df[df.qty > 5]\nn = len(top)", names={})

        assert reading.bindings["top"].from_read
        assert reading.bindings["n"].from_read

    def test_a_list_the_cell_fills_from_reads_is_from_a_read(self):
        reading = read_cell(textwrap.dedent("""
            frames = []
            for path in paths:
                frames.append(pd.read_csv(path))
            totals = {}
            totals["S1"] = query_minds_data("SELECT SUM(qty) FROM stock")
            stock = pd.concat(frames)
        """), names={})

        assert reading.bindings["frames"].from_read
        assert reading.bindings["totals"].from_read
        assert reading.bindings["stock"].from_read

    def test_a_name_rebound_from_something_else_is_no_longer_from_a_read(self):
        reading = read_cell("df = pd.read_csv('stock.csv')\ndf = pd.DataFrame()", names={})

        assert not reading.bindings["df"].from_read

    def test_names_from_earlier_cells_are_what_the_cell_loads_and_its_bindings_inputs(self):
        reading = read_cell(textwrap.dedent("""
            summary = df.groupby("site").qty.sum()
            print(summary.loc["S1"])
        """), names={})

        assert "df" in reading.loaded_names
        assert "summary" not in reading.loaded_names
        assert "df" in reading.bindings["summary"].inputs
        assert not reading.bindings["summary"].from_read

    def test_a_name_bound_only_in_a_branch_may_still_be_the_earlier_cells(self):
        reading = read_cell(textwrap.dedent("""
            if refresh:
                df = pd.read_csv("stock.csv")
            print(df.qty.sum())
        """), names={})

        assert reading.bindings["df"].from_read
        assert "df" in reading.bindings["df"].inputs
        assert "df" in reading.loaded_names

    def test_a_function_brings_what_its_body_loads_to_the_cell_that_reads_it(self):
        called = read_cell("def total():\n    return df.qty.sum()\nprint(total())", names={})
        defined = read_cell("def total():\n    return df.qty.sum()\n", names={})

        assert "df" in called.loaded_names
        assert "df" in called.bindings["total"].inputs
        assert "df" not in defined.loaded_names

    def test_a_function_that_reads_makes_what_its_call_binds_from_a_read(self):
        functions = read_cell("def load():\n    return pd.read_csv('stock.csv')\n", names={}).functions

        reading = read_cell("df = load()", names={}, functions=functions)

        assert functions["load"].reads_data
        assert reading.bindings["df"].from_read
        assert "load" in reading.loaded_names

    def test_a_functions_parameters_and_locals_are_not_the_cells_names(self):
        reading = read_cell(textwrap.dedent("""
            def scale(df):
                factor = 2
                return df * factor
            out = [row for row in rows]
        """), names={})

        assert "df" not in reading.loaded_names
        assert "row" not in reading.loaded_names
        assert "rows" in reading.loaded_names
        assert set(reading.bindings) == {"scale", "out"}


class TestCodeInsidePlaceholders:
    def test_a_query_inside_an_fstring_placeholder_runs(self):
        reading = read_cell(
            """print(f"Total stock: {query_minds_data('SELECT SUM(qty) FROM stock').iloc[0, 0]}")""", names={},
        )

        assert reading.queries == ("SELECT SUM(qty) FROM stock",)
        assert reading.calls_query_minds_data

    def test_a_request_inside_an_fstring_placeholder_is_made(self):
        reading = read_cell("""print(f"{requests.get('https://api.example.com/v1/rates').json()['rate']}")""", names={})

        assert reading.requests == (HttpRequest("GET", "https://api.example.com/v1/rates"),)

    def test_an_environment_read_inside_an_fstring_placeholder_is_noted(self):
        reading = read_cell('''x = f"{os.environ['DS_ERP_URL'].rstrip('/')}/stock"''', names={})

        assert "DS_ERP_URL" in reading.environment_variables

    @pytest.mark.parametrize("code", [
        """print("Total: " + str(query_minds_data('SELECT SUM(qty) FROM stock').iloc[0, 0]) + " units")""",
        """print("Total: %s" % query_minds_data('SELECT SUM(qty) FROM stock').iloc[0, 0])""",
        """print("Total: " + f"{query_minds_data('SELECT SUM(qty) FROM stock').iloc[0, 0]}")""",
    ])
    def test_a_call_inside_a_concatenation_or_template_runs(self, code: str):
        reading = read_cell(code, names={})

        assert reading.calls_query_minds_data
        assert reading.queries == ("SELECT SUM(qty) FROM stock",)


class TestWhatIsNotAQuery:
    def test_a_printed_label_that_reads_like_sql_is_not_a_query(self):
        assert _queries("""
            df = pd.read_csv("stock.csv"); print("With rows from stock.csv:", int(df.qty.sum()))
            print("Select a row from the table below:")
            print("With data from the ERP")
        """) == []

    def test_a_printed_or_titled_label_without_closing_punctuation_is_not_a_query(self):
        assert _queries("""
            print('Select parts from S1 with qty above 100')
            plt.title("Select parts from S1 with qty above 100")
            ax.set_title(f"Select parts from {site} with qty above 100")
        """) == []

    def test_a_comment_hides_the_from_it_holds(self):
        assert not is_sql("SELECT 1 -- from x")

    def test_unclosed_comments_read_in_linear_time(self):
        start = time.perf_counter()

        assert not is_sql("/* a" * 50_000)
        assert time.perf_counter() - start < TIME_BOUND_SECONDS

    def test_a_comment_that_reads_like_sql_is_not_a_query(self):
        assert _queries("""
            # select the rows from df where qty is high
            print(df[df.qty > 5])
        """) == []

    def test_a_pandas_filter_is_not_a_query(self):
        assert _queries("print(df.query('qty > 5'))") == []

    def test_ddl_and_writes_are_not_queries(self):
        assert _queries("""
            con.execute("CREATE TABLE t AS SELECT * FROM stock")
            con.execute("INSERT INTO t SELECT * FROM stock")
            con.execute("ATTACH 'dbname=erp' AS pg (TYPE postgres)")
        """) == []

    def test_a_docstring_is_not_a_query(self):
        assert _queries('''
            """SELECT qty FROM stock is what this cell would run."""
            print(1)
        ''') == []

    def test_code_that_does_not_parse_reads_as_nothing_and_keeps_the_names(self):
        reading = read_cell("cur.execute('SELECT qty FROM stock'", names={"sql": "SELECT 1 FROM t"})

        assert reading.queries == ()
        assert reading.requests == ()
        assert reading.names == {"sql": "SELECT 1 FROM t"}


class TestQueryLimits:
    def test_at_most_three_distinct_queries_in_code_order(self):
        assert _queries("""
            cur.execute("SELECT a FROM t")
            cur.execute("SELECT a FROM t")
            cur.execute("SELECT b FROM t")
            cur.execute("SELECT c FROM t")
            cur.execute("SELECT d FROM t")
        """) == ["SELECT a FROM t", "SELECT b FROM t", "SELECT c FROM t"]


class TestStatementsInOneString:
    @pytest.mark.parametrize("second", [
        "CREATE USER bob IDENTIFIED BY 'XyZhunter2'",
        "ALTER USER root WITH PASSWORD 'XyZhunter2'",
        "SET PASSWORD FOR bob = 'XyZhunter2'",
    ])
    def test_a_statement_after_a_query_is_not_listed(self, second: str):
        sql = f"SELECT qty FROM stock; {second}"

        assert read_cell(f"cur.execute({sql!r})", names={}).queries == ("SELECT qty FROM stock;",)

    def test_each_query_in_a_script_is_listed_on_its_own_in_order(self):
        assert _queries('''
            con.executescript("""
                SELECT qty FROM stock WHERE note = 'it''s';
                DELETE FROM stock WHERE qty = 0;
                SELECT site FROM sites
            """)
        ''') == ["SELECT qty FROM stock WHERE note = 'it''s';", "SELECT site FROM sites"]

    def test_a_function_lists_only_the_query_of_a_string_with_two_statements(self):
        reading = read_cell(textwrap.dedent("""
            def stock():
                return cur.execute("SELECT qty FROM stock; CREATE USER bob IDENTIFIED BY 'XyZhunter2'")
        """), names={})

        assert reading.functions["stock"].queries == ("SELECT qty FROM stock;",)

    @pytest.mark.parametrize("sql", [
        "SELECT qty FROM stock WHERE note = 'a; b' -- c; d\n AND site = 'S1'",
        "SELECT \"a;b\", `c;d` FROM stock /* e; f */ WHERE site = 'S1'",
        "SELECT qty FROM stock WHERE note = 'it''s; fine'",
    ])
    def test_a_semicolon_in_a_literal_a_quoted_name_or_a_comment_keeps_its_query_whole(self, sql: str):
        assert read_cell(f"cur.execute({sql!r})", names={}).queries == (sql,)

    @pytest.mark.parametrize("sql", [
        "SELECT $$ab$$ AS x FROM t",
        "SELECT $body$a$$b$body$ AS x FROM t",
        "SELECT [a;b] FROM t",
    ], ids=["dollar-quote", "tagged-dollar-quote", "bracketed-name"])
    def test_dollar_quotes_with_common_boundaries_and_bracketed_names_keep_the_query(self, sql: str):
        assert read_cell(f"cur.execute({sql!r})", names={}).queries == (sql,)

    @pytest.mark.parametrize(("query", "listed"), [
        ("SELECT $body$ab$body$ AS x FROM t;", "SELECT $body$ab$body$ AS x FROM t;"),
        ("SELECT qty FROM stock WHERE id = $1;", "SELECT qty FROM stock WHERE id = $1;"),
        ("SELECT a$b$c FROM t;", "SELECT a$b$c FROM t;"),
        ("SELECT [a;b] FROM t;", "SELECT [a;b] FROM t;"),
    ], ids=["after-a-dollar-quote", "positional-parameter", "dollar-in-a-name", "after-a-bracketed-name"])
    def test_a_statement_after_a_dollar_or_a_bracket_is_not_listed(self, query: str, listed: str):
        sql = f"{query} CREATE USER bob IDENTIFIED BY 'XyZhunter2'"

        assert read_cell(f"cur.execute({sql!r})", names={}).queries == (listed,)

    @pytest.mark.parametrize("sql", [
        "SELECT $$a;b$$ AS x FROM t",
        "SELECT $body$a;$$;b$body$ AS x FROM t",
        "SELECT qty FROM stock AS $tag$; CREATE USER demo IDENTIFIED BY 'SyntheticCredential'; "
        "SELECT qty FROM stock AS $tag$",
    ], ids=["untagged-string", "tagged-string", "mysql-alias-batch"])
    def test_dollar_dependent_statement_boundaries_omit_the_batch(self, sql: str):
        assert read_cell(f"cur.execute({sql!r})", names={}).queries == ()

    def test_dollar_spelling_inside_a_quoted_literal_keeps_its_query(self):
        sql = "SELECT qty FROM stock WHERE note = '$tag$a;b$tag$'"

        assert read_cell(f"cur.execute({sql!r})", names={}).queries == (sql,)

    # MySQL reads a backslash in a literal or a quoted name as escaping the
    # next character; Postgres and SQLite read it as an ordinary one, so
    # 'C:\' is a closed literal there and the ; after it ends the query.
    @pytest.mark.parametrize("sql", [
        "SELECT qty FROM stock WHERE p = 'C:\\'; ALTER USER bob WITH PASSWORD 'XyZhunter2'",
        'SELECT "a\\" FROM t; ALTER USER bob WITH PASSWORD \'XyZhunter2\'',
        "SELECT qty FROM stock WHERE note = 'it\\'s; fine'",
    ], ids=["literal", "quoted-name", "escaped-quote"])
    def test_a_string_a_backslash_splits_differently_lists_nothing(self, sql: str):
        assert read_cell(f"cur.execute({sql!r})", names={}).queries == ()

    @pytest.mark.parametrize("sql", [
        "SELECT qty FROM stock WHERE p = 'C:\\'",
        "SELECT qty FROM stock WHERE note = 'it\\'s fine'",
        "SELECT qty FROM stock WHERE p = 'C:\\new;'",
    ])
    def test_a_backslash_that_moves_no_split_keeps_its_query(self, sql: str):
        assert read_cell(f"cur.execute({sql!r})", names={}).queries == (sql,)

    @pytest.mark.parametrize("sql", [
        "SELECT qty FROM stock WHERE 1=1--0; CREATE USER demo IDENTIFIED BY 'SyntheticCredential'",
        "SELECT qty FROM stock # ' hidden quote\n; CREATE USER demo IDENTIFIED BY 'SyntheticCredential'",
        "SELECT qty FROM stock # ; hidden separator\nWHERE qty > 0",
        "SELECT qty FROM stock --0/* ; hidden comment */; SET PASSWORD = 'SyntheticCredential'",
        "SELECT qty FROM stock # \\\'\n; CREATE USER demo IDENTIFIED BY 'SyntheticCredential'",
    ], ids=["dash-arithmetic", "hash-quote", "hash-semicolon", "mixed-comments", "hash-backslash"])
    def test_dialect_comment_boundary_disagreement_omits_the_batch(self, sql: str):
        assert read_cell(f"cur.execute({sql!r})", names={}).queries == ()

    # Snowflake ends a // comment at the line end; BigQuery and ClickHouse
    # read # and a -- with no space after it as comments; Postgres escapes a
    # quote with a backslash in an E'' or e'' string but not in a
    # double-quoted name or a standard string (all three read a doubled quote
    # as one), and an E that ends a name or a keyword opens no E'' string
    # (LIKE'\', €E'\', t$E'\'); Postgres reads [ as an array, whose string
    # may hold a ]; Postgres takes any non-ASCII character in a dollar tag,
    # and reads one before $$ as part of a name. Each database ends the query
    # at the ; and runs the statement that sets a password.
    @pytest.mark.parametrize("sql", [
        "SELECT qty FROM stock // it's\n; CREATE USER bob PASSWORD 'XyZhunter2'",
        "SELECT qty FROM stock //* note\n; CREATE USER bob PASSWORD 'XyZhunter2' -- */",
        "SELECT qty FROM stock --it's\nWHERE a = 1 # /* note\n; CREATE USER bob IDENTIFIED BY 'XyZhunter2'",
        "SELECT qty FROM stock WHERE note = E'it\\'s' AND \"col\\\" = 1; CREATE USER bob PASSWORD 'XyZhunter2'",
        "SELECT E'\\'', '\\' FROM stock; ALTER ROLE bob PASSWORD $p$XyZhunter2$p$; SELECT ''",
        "SELECT e'\\'', '\\' FROM stock; ALTER ROLE bob PASSWORD $p$XyZhunter2$p$; SELECT ''",
        "SELECT E'\\'' AS \"\\\", '\\' FROM t; ALTER ROLE bob PASSWORD $p$XyZhunter2$p$; SELECT ''",
        "SELECT E'it''s\\'', '\\' FROM t; ALTER ROLE bob PASSWORD $p$XyZhunter2$p$; SELECT ''",
        "SELECT E'\\'' AS a FROM t WHERE b LIKE'\\'; ALTER ROLE bob PASSWORD $p$XyZhunter2$p$; SELECT ''",
        "SELECT E'\\'' AS a FROM t WHERE b = €E'\\'; ALTER ROLE bob PASSWORD $p$XyZhunter2$p$; SELECT ''",
        "SELECT E'\\'' AS a FROM t WHERE b = t$E'\\'; ALTER ROLE bob PASSWORD $p$XyZhunter2$p$; SELECT ''",
        "SELECT ARRAY['a]'] FROM stock; CREATE USER bob PASSWORD 'XyZhunter2'",
        'SELECT data["a]"] FROM stock; CREATE USER bob IDENTIFIED BY \'XyZhunter2\'',
        "SELECT $é$it's$é$ FROM stock; ALTER ROLE bob PASSWORD $p$XyZhunter2$p$; SELECT ''",
        "SELECT $a€$it's$a€$ FROM t; ALTER ROLE bob PASSWORD $p$XyZhunter2$p$; SELECT ''",
        "SELECT $$/*$$ AS €$$ FROM stock; ALTER ROLE bob PASSWORD 'XyZhunter2'; SELECT $$*/$$ FROM stock",
    ], ids=[
        "double-slash-comment", "double-slash-before-a-block-comment", "hash-and-dashdash-comments",
        "e-string-beside-a-quoted-name", "e-string-beside-a-standard-string",
        "lowercase-e-string-beside-a-standard-string", "e-string-beside-a-quoted-name-and-a-standard-string",
        "e-string-with-a-doubled-quote-beside-a-standard-string", "keyword-ending-in-e-before-a-string",
        "non-ascii-name-ending-in-e-before-a-string", "name-with-a-dollar-ending-in-e-before-a-string",
        "array-of-a-string-with-a-bracket", "subscript-of-a-name-with-a-bracket", "non-ascii-dollar-tag",
        "dollar-tag-ending-in-a-non-ascii-symbol", "non-ascii-name-before-a-dollar-quote",
    ])
    def test_a_statement_another_database_ends_at_a_semicolon_lists_nothing(self, sql: str):
        assert read_cell(f"cur.execute({sql!r})", names={}).queries == ()

    @pytest.mark.parametrize(("sql", "listed"), [
        (
            "SELECT qty FROM stock WHERE url = 'http://x'; DELETE FROM stock",
            "SELECT qty FROM stock WHERE url = 'http://x';",
        ),
        ("SELECT qty FROM stock WHERE note = 'a#b'; DELETE FROM stock", "SELECT qty FROM stock WHERE note = 'a#b';"),
        ("SELECT ARRAY[1, 2] FROM t; DELETE FROM t", "SELECT ARRAY[1, 2] FROM t;"),
    ], ids=["double-slash-in-a-literal", "hash-in-a-literal", "array-without-a-string"])
    def test_a_split_the_added_readings_agree_on_keeps_its_query(self, sql: str, listed: str):
        assert read_cell(f"cur.execute({sql!r})", names={}).queries == (listed,)

    @pytest.mark.parametrize("comment", [
        "/* outer /* inner */ ' */",
        "/*!50000 SELECT secret FROM credentials */",
        "/*M! SELECT secret FROM credentials */",
    ])
    def test_nested_or_executable_comments_omit_dialect_dependent_sql(self, comment: str):
        sql = f"SELECT qty FROM stock {comment}; CREATE USER demo IDENTIFIED BY 'SyntheticCredential'"

        assert read_cell(f"cur.execute({sql!r})", names={}).queries == ()

    @pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"])
    @pytest.mark.parametrize("space", [" ", "\t"])
    def test_dash_comments_with_whitespace_end_at_line_boundaries(self, ending: str, space: str):
        query = f"SELECT qty FROM stock --{space}comment with ;{ending};"
        sql = query + " CREATE USER demo IDENTIFIED BY 'SyntheticCredential'"

        assert read_cell(f"cur.execute({sql!r})", names={}).queries == (query,)

    @pytest.mark.parametrize("sql", [
        "SELECT qty FROM stock WHERE note IN ('# ;', '--0;')",
        'SELECT qty FROM stock WHERE "# ;" = `--0;`',
        "SELECT qty FROM stock /* # ; --0; */ WHERE qty > 0",
        "SELECT qty FROM stock # harmless comment\nWHERE qty > 0",
        "SELECT qty FROM stock WHERE qty = 1--0",
    ])
    def test_comment_spelling_that_moves_no_boundary_keeps_the_query(self, sql: str):
        assert read_cell(f"cur.execute({sql!r})", names={}).queries == (sql,)

    def test_a_label_that_reads_like_sql_up_to_a_semicolon_is_not_a_query(self):
        assert _queries("""
            ax.set_xlabel("Select a site from the list; units are pallets.")
        """) == []

    # Each filler opens a comment, a literal, a quoted name or a dollar-quoted
    # string and never closes it, so the rest of the string is one statement.
    # A splitter that tries each opener again reads the rest of the string
    # once per opener.
    @pytest.mark.parametrize(("filler", "ambiguous"), [
        ("/*'';" * 40_000, False),
        ("\\'" * 60_000, False),
        ('\\"' * 60_000, False),
        ("".join(f"$t{i}x$;" for i in range(15_000)), True),
        ("[;" * 60_000, False),
    ], ids=["comment", "literal", "quoted-name", "dollar-quote", "bracketed-name"])
    def test_a_string_left_open_splits_into_statements_in_linear_time(self, filler: str, ambiguous: bool):
        sql = "SELECT qty FROM stock WHERE note = " + filler
        code = f"cur.execute({sql!r})"
        start = time.perf_counter()

        reading = read_cell(code, names={})

        assert time.perf_counter() - start < TIME_BOUND_SECONDS
        assert reading.queries == (() if ambiguous else (sql,))

    # Every reading splits this string alike, so each one scans it to the
    # end: 20,000 closed block comments, each checked for a nested /*, and
    # 20,000 statements after the query.
    def test_a_string_every_reading_splits_alike_splits_in_linear_time(self):
        sql = "SELECT qty FROM stock WHERE note = 'a'; " + "/**/;" * 20_000
        code = f"cur.execute({sql!r})"
        start = time.perf_counter()

        reading = read_cell(code, names={})

        assert time.perf_counter() - start < TIME_BOUND_SECONDS
        assert reading.queries == ("SELECT qty FROM stock WHERE note = 'a';",)

    # The same string after a literal that holds every character the
    # readings are told apart by, so every reading scans it to the end.
    def test_a_string_that_runs_every_reading_splits_in_linear_time(self, monkeypatch: pytest.MonkeyPatch):
        sql = "SELECT qty FROM stock WHERE note = 'a\\b #c //d --e [f] $g$'; " + "/**/;" * 20_000
        reads = _counting_reads(monkeypatch)
        start = time.perf_counter()

        reading = read_cell(f"cur.execute({sql!r})", names={})

        assert time.perf_counter() - start < TIME_BOUND_SECONDS
        assert reading.queries == ("SELECT qty FROM stock WHERE note = 'a\\b #c //d --e [f] $g$';",)
        assert len(reads) == len(cell_queries_mod._SQL_READINGS)

    # The readings differ only where a text holds a backslash (the quote
    # rules), -- or # (MySQL's comments), # (BigQuery's), // (Snowflake's),
    # [ (a bracket a quote ends) or $ (dollar quoting). A query with none of
    # them reads alike under every reading, so it is read once.
    @pytest.mark.parametrize(("where", "reads"), [
        ("", 1), ("AND note = 'a\\b'", 4), ("-- note\n", 2), ("AND note = 'a#b'", 3), ("AND note = 'a//b'", 2),
        ("AND a[1] = 2", 2), ("AND note = $$a$$", 2), ("AND note = 'a\\b #c //d --e [f] $g$'", 64),
    ], ids=["none", "backslash", "dashdash", "hash", "double-slash", "bracket", "dollar", "every-mark"])
    def test_only_the_readings_a_querys_characters_can_tell_apart_run(
        self, where: str, reads: int, monkeypatch: pytest.MonkeyPatch,
    ):
        sql = f"SELECT qty FROM stock WHERE qty > 0 {where}"
        counted = _counting_reads(monkeypatch)

        read_cell(f"cur.execute({sql!r})", names={})

        assert len(counted) == reads

    # 20 reads of a 64 KiB query whose characters no reading tells apart.
    # 20 reads with every reading take about 2 s on a laptop; with one
    # reading, about 0.05 s.
    def test_a_long_query_no_reading_tells_apart_reads_quickly(self):
        sql = "SELECT id FROM orders WHERE id IN (" + "'ord-0000001', " * 4_350 + "'x')"
        code = f"cur.execute({sql!r})"
        start = time.perf_counter()

        for _ in range(20):
            reading = read_cell(code, names={})

        assert time.perf_counter() - start < 1.0
        assert reading.queries == (sql,)

    # Skipping a reading never changes a split: on texts of the characters
    # the readings differ on, the readings that run agree exactly when all
    # of them would.
    def test_the_readings_that_run_split_every_text_as_all_of_them_do(self):
        rng = random.Random(3433)
        for alphabet in (
            "'\"`[]$;-#/*\\\n aSxE\x01é€", "';- \nSELECTFROMa", "'\";-/ \n", "[]';\"", "$;'a \né€", "\\';\"-#\n E",
        ):
            for _ in range(500):
                text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 60)))
                assert cell_queries_mod._statements(text) == _split_by_every_reading(text), text


def _counting_reads(monkeypatch: pytest.MonkeyPatch) -> list[re.Pattern[str]]:
    """The readings the SQL split runs from here on, one entry per scan."""
    reads: list[re.Pattern[str]] = []
    statement_ends = cell_queries_mod._statement_ends

    def counted(text: str, *, token: re.Pattern[str]) -> list[int] | None:
        reads.append(token)
        return statement_ends(text, token=token)

    monkeypatch.setattr(cell_queries_mod, "_statement_ends", counted)
    return reads


def _split_by_every_reading(text: str) -> list[str]:
    """``text`` split as _statements splits it, with every reading run."""
    tokens = [reading.token for reading in cell_queries_mod._SQL_READINGS]
    ends = cell_queries_mod._statement_ends(text, token=tokens[0])
    if ends is None or any(ends != cell_queries_mod._statement_ends(text, token=token) for token in tokens[1:]):
        return []
    return [text[start:end] for start, end in pairwise([0, *ends, len(text)])]


class TestRequestsACellMakes:
    def test_a_request_to_a_saved_connection_names_its_variable(self):
        assert _requests("""
            import os, requests
            r = requests.get(f"{os.environ['DS_ERP_URL']}/inventory", params={"site": "S1"})
            s = requests.get(os.environ["DS_ERP_URL"] + "/sites")
            t = requests.get(f"{DS_ERP_URL}/orders")
        """) == ["GET {DS_ERP_URL}/inventory", "GET {DS_ERP_URL}/sites", "GET {DS_ERP_URL}/orders"]

    def test_headers_auth_body_and_params_never_enter_a_request(self):
        code = """
            r = httpx.post(
                "https://api.example.com/v1/search",
                headers={"Authorization": "Bearer tok-123"},
                auth=("erp", "pw-456"),
                json={"q": "steel"},
                params={"key": "k-789"},
                cookies={"session": "c-000"},
            )
        """

        requests = _requests(code)

        assert requests == ["POST https://api.example.com/v1/search"]
        for secret in ("tok-123", "pw-456", "steel", "k-789", "c-000"):
            assert secret not in requests[0]

    def test_a_literal_url_loses_its_query_string_fragment_and_userinfo(self):
        assert _requests(
            "r = requests.get('https://erp:pw@api.example.com/v1/rates?api_key=abc&region=eu#top')"
        ) == ["GET https://api.example.com/v1/rates"]

    def test_a_client_instance_request_counts_when_its_url_reads_as_one(self):
        assert _requests("""
            with httpx.Client() as client:
                a = client.get(f"{base}/stock")
                b = client.request("patch", "https://api.example.com/v1/stock/1")
            qty = row.get("qty")
            url = config.get("url_key")
        """) == ["GET {base}/stock", "PATCH https://api.example.com/v1/stock/1"]

    def test_a_bound_base_url_stays_a_string_and_its_placeholder_stays_as_written(self):
        reading = read_cell(textwrap.dedent("""
            base = "https://api.example.com"
            r = requests.get(f"{base}/stock")
        """), names={})

        assert reading.strings == ("https://api.example.com", "{base}/stock")
        assert [str(request) for request in reading.requests] == ["GET {base}/stock"]

    def test_a_get_on_a_name_holding_a_url_word_only_inside_another_word_is_not_a_request(self):
        assert _requests("""
            a = row.get(database)
            b = d.get(ghost)
            c = d.get(security)
            e = d.get(hostname)
        """) == []
        assert _requests("""
            a = session.get(apiUrl)
            b = session.get(api_url)
        """) == ["GET {apiUrl}", "GET {api_url}"]

    @pytest.mark.parametrize("code", [
        'requests.get(f"{proto}://admin:s3cret@erp.internal/api/stock")',
        'requests.get("{}://admin:s3cret@erp.internal/api/stock".format(p))',
        """requests.get(f"{scheme or 'https'}://admin:s3cret@erp.internal/api/stock")""",
        'requests.get(os.environ["DS_SCHEME"] + "://admin:s3cret@erp.internal/api/stock")',
    ])
    def test_a_user_and_password_after_a_placeholder_scheme_never_enter_a_request(self, code: str):
        requests = [str(request) for request in read_cell(code, names={}).requests]

        assert len(requests) == 1
        assert "s3cret" not in requests[0]
        assert "admin:" not in requests[0]

    def test_a_name_added_into_a_url_reads_as_a_placeholder(self):
        reading = read_cell(textwrap.dedent("""
            TOKEN = "tok-9f8e7d6c"
            a = requests.get("https://erp.example.com/api/" + TOKEN + "/orders")
            url = "https://api.telegram.org/bot" + TOKEN
            url += "/sendMessage"
            b = requests.post(url)
            path = "/orders/"
            path += TOKEN
            c = requests.get(BASE + path)
        """), names={"BASE": "https://erp.example.com/api"})

        requests = [str(request) for request in reading.requests]

        assert requests == [
            "GET https://erp.example.com/api/{TOKEN}/orders",
            "POST https://api.telegram.org/bot{TOKEN}/sendMessage",
            "GET https://erp.example.com/api{path}",
        ]
        assert not any("tok-9f8e7d6c" in request for request in requests)

    def test_a_name_path_joined_onto_a_url_reads_as_a_placeholder(self):
        assert _requests("""
            TOKEN = "tok-9f8e7d6c"
            a = requests.get(posixpath.join("https://erp.example.com/api", TOKEN, "orders"))
            b = requests.get(os.path.join(BASE, TOKEN))
        """, BASE="https://erp.example.com/api") == [
            "GET https://erp.example.com/api/{TOKEN}/orders",
            "GET https://erp.example.com/api/{TOKEN}",
        ]

    def test_a_base_url_a_connection_and_a_query_still_read_as_their_strings_in_a_join(self):
        assert _requests("""
            BASE = "https://erp.example.com/api"
            ERP = os.environ["DS_ERP_URL"]
            a = requests.get(BASE + "/stock")
            b = requests.get(ERP + "/sites")
        """) == ["GET https://erp.example.com/api/stock", "GET {DS_ERP_URL}/sites"]

    def test_a_percent_formatted_url_keeps_its_fields_as_placeholders(self):
        assert _requests("""
            a = requests.get("https://erp.example.com/api/%s/orders" % sku)
            b = requests.get("https://erp.example.com/api/%(site)s/stock" % params)
            c = requests.get("https://erp.example.com/%s/%s" % (region, sku))
        """) == [
            "GET https://erp.example.com/api/{sku}/orders",
            "GET https://erp.example.com/api/{site}/stock",
            "GET https://erp.example.com/{region}/{sku}",
        ]

    def test_a_base_url_clients_relative_request_is_made_against_its_base(self):
        assert _requests("""
            erp = httpx.Client(base_url=os.environ["DS_ERP_URL"])
            a = erp.get("/inventory")
            with httpx.Client(base_url="https://erp.example.com/") as api:
                b = api.post("orders")
            bot = httpx.Client(base_url="https://api.telegram.org/bot123456:ABC-secret")
            c = bot.post("/sendMessage")
        """) == [
            "GET {DS_ERP_URL}/inventory",
            "POST https://erp.example.com/orders",
            "POST {base_url}/sendMessage",
        ]

    @pytest.mark.parametrize(("base", "expected"), [
        ('os.environ["DS_ERP_URL"]', "{DS_ERP_URL}"),
        ('"https://stock.example.com/"', "https://stock.example.com"),
        ('"https://api.example.com/private-token"', "{base_url}"),
        ('"https://user:secret@stock.example.com/?token=secret"', "https://stock.example.com"),
    ])
    def test_client_metadata_carries_relative_requests_into_the_next_cell(self, base: str, expected: str):
        earlier = read_cell(f"erp = httpx.Client(base_url={base})", names={})

        later = read_cell("print(erp.get('/inventory').json()['qty'])", names=earlier.names, clients=earlier.clients)

        assert earlier.clients == {"erp": expected}
        assert later.clients == earlier.clients
        assert later.requests == (HttpRequest("GET", expected + "/inventory"),)
        assert later.bindings == {}

    @pytest.mark.parametrize("rebind", [
        "erp = {}", "del erp", "import other as erp", "erp += other", "erp, other = pair",
        'erp.base_url = "https://deliveries.example.com"',
    ])
    def test_rebinding_a_client_discards_its_saved_base(self, rebind: str):
        earlier = read_cell('erp = httpx.Client(base_url="https://stock.example.com")', names={})

        changed = read_cell(rebind, names=earlier.names, clients=earlier.clients)
        later = read_cell("erp.get('/inventory')", names=changed.names, clients=changed.clients)

        assert changed.clients == {}
        assert later.requests == ()

    def test_a_conditional_client_rebind_carries_no_uncertain_base(self):
        earlier = read_cell('erp = httpx.Client(base_url="https://stock.example.com")', names={})

        changed = read_cell(textwrap.dedent('''
            if changed:
                erp = httpx.Client(base_url="https://deliveries.example.com")
        '''), names=earlier.names, clients=earlier.clients)
        later = read_cell("erp.get('/inventory')", names=changed.names, clients=changed.clients)

        assert changed.clients == {}
        assert later.requests == ()

    def test_a_parameter_hides_a_carried_client_only_inside_its_function(self):
        earlier = read_cell('erp = httpx.Client(base_url="https://stock.example.com")', names={})

        reading = read_cell(textwrap.dedent('''
            def other(erp):
                return erp.get("/deliveries")
            other(local_client)
            erp.get("/inventory")
        '''), names=earlier.names, clients=earlier.clients)

        assert reading.functions["other"].requests == ()
        assert reading.clients == earlier.clients
        assert reading.requests == (HttpRequest("GET", "https://stock.example.com/inventory"),)

    def test_a_local_client_restores_the_carried_client_after_its_function(self):
        earlier = read_cell('erp = httpx.Client(base_url="https://stock.example.com")', names={})

        reading = read_cell(textwrap.dedent('''
            def other():
                erp = httpx.Client(base_url="https://deliveries.example.com")
                return erp.get("/deliveries")
            other()
            erp.get("/inventory")
        '''), names=earlier.names, clients=earlier.clients)

        assert reading.clients == earlier.clients
        assert reading.requests == (
            HttpRequest("GET", "https://deliveries.example.com/deliveries"),
            HttpRequest("GET", "https://stock.example.com/inventory"),
        )

    def test_rebinding_a_client_invalidates_a_helpers_frozen_request(self):
        earlier = read_cell(textwrap.dedent('''
            erp = httpx.Client(base_url="https://stock.example.com")
            def fetch():
                return erp.get("/inventory")
        '''), names={})

        reading = read_cell(textwrap.dedent('''
            erp = httpx.Client(base_url="https://deliveries.example.com")
            fetch()
            erp.get("/inventory")
        '''), names=earlier.names, functions=earlier.functions, clients=earlier.clients)

        assert reading.requests == (HttpRequest("GET", "https://deliveries.example.com/inventory"),)
        assert reading.functions["fetch"].requests == ()

    def test_a_failed_parse_preserves_carried_clients(self):
        earlier = read_cell('erp = httpx.Client(base_url="https://stock.example.com")', names={})

        reading = read_cell("erp = (", names=earlier.names, clients=earlier.clients)

        assert reading.clients == earlier.clients

    def test_client_metadata_is_bounded_in_count_and_url_length(self):
        code = "\n".join(
            f'client_{index} = httpx.Client(base_url="https://stock.example.com")'
            for index in range(MAX_CLIENTS + 1)
        )

        reading = read_cell(code, names={})
        long_base = read_cell(f'erp = httpx.Client(base_url={"https://" + "x" * MAX_STRING_CHARS!r})', names={})

        assert len(reading.clients) == MAX_CLIENTS
        assert f"client_{MAX_CLIENTS}" not in reading.clients
        assert long_base.clients == {"erp": "{base_url}"}

    def test_a_get_on_a_plural_url_name_is_a_request(self):
        assert _requests("""
            a = session.get(urls[i])
            b = session.get(endpoints["stock"])
            qty = row.get("qty")
        """) == ["GET {urls[i]}", 'GET {endpoints["stock"]}']

    def test_urllib_requests_take_their_method_from_data_or_method(self):
        assert _requests("""
            from urllib.request import Request, urlopen
            a = urlopen("https://example.com/a")
            b = urlopen(Request("https://example.com/b", data=payload))
            c = urllib.request.Request("https://example.com/c", method="PUT")
        """) == ["GET https://example.com/a", "POST https://example.com/b", "PUT https://example.com/c"]

    def test_at_most_three_requests(self):
        assert len(_requests("""
            for path in "abcd":
                requests.get("https://example.com/a")
                requests.get("https://example.com/b")
                requests.get("https://example.com/c")
                requests.get("https://example.com/d")
        """)) == 3


class TestFilesACellWrites:
    def test_saves_and_writing_opens_name_the_files_they_write(self):
        reading = read_cell(textwrap.dedent("""
            df.to_csv("a.csv")
            df.to_csv(path_or_buf="b.csv")
            df.to_excel(excel_writer="c.xlsx")
            plt.savefig(fname="d.png")
            open("e.csv", "w").write(text)
            with open("f.csv", mode="a") as f:
                f.write(text)
            out = "g.csv"
            df.to_csv(out)
            log = "h.txt"
            with open(log, mode="a") as f:
                f.write(text)
            Path("i.md").write_text("# Notes")
            with Path("j.json").open("w") as f:
                f.write(text)
        """), names={})

        assert reading.written_files == (
            "a.csv", "b.csv", "c.xlsx", "d.png", "e.csv", "f.csv", "g.csv", "h.txt", "i.md", "j.json",
        )

    def test_reads_write_no_file(self):
        reading = read_cell(textwrap.dedent("""
            a = open("a.csv").read()
            b = open("b.csv", "r").read()
            c = pd.read_csv("c.csv")
            d = Path("d.md").read_text()
            e = Path("e.json").open()
        """), names={})

        assert reading.written_files == ()

    def test_workbook_numpy_and_arrow_saves_name_the_files_they_write(self):
        reading = read_cell(textwrap.dedent("""
            wb.save("a.xlsx")
            with pd.ExcelWriter("b.xlsx") as writer:
                df.to_excel(writer, sheet_name="S1")
            np.savetxt("c.csv", totals)
            pq.write_table(table, "d.parquet")
            book = xlsxwriter.Workbook("e.xlsx")
        """), names={})

        assert reading.written_files == ("a.xlsx", "b.xlsx", "c.csv", "d.parquet", "e.xlsx")

    def test_a_joined_or_named_path_names_the_file_it_writes(self):
        reading = read_cell(textwrap.dedent("""
            df.to_excel(os.path.join(out_dir, "f.xlsx"))
            out = Path("g.md")
            out.write_text(text)
            (Path(out_dir) / "h.md").write_text(text)
            with out.open("a") as f:
                f.write(text)
        """), names={})

        assert reading.written_files == ("{out_dir}/f.xlsx", "g.md", "{Path(out_dir)}/h.md")

    def test_every_string_in_a_target_that_does_not_render_is_written(self):
        reading = read_cell('df.to_csv(build_path("summary.csv", folder="out"))', names={})

        assert reading.written_files == ("summary.csv", "out")


class TestPathJoins:
    def test_a_joined_path_reads_as_one_path_and_not_its_parts(self):
        assert _strings("""
            a = pd.read_csv(os.path.join("archive", "stock.csv"))
            b = pd.read_csv(Path("archive") / "orders.csv")
            c = pd.read_csv(Path("archive", "sites.csv"))
        """) == ("archive/stock.csv", "archive/orders.csv", "archive/sites.csv")

    def test_a_directory_the_code_does_not_spell_stays_a_placeholder(self):
        assert _strings("df = pd.read_csv(Path(ATT) / 'stock.csv')") == ("{Path(ATT)}/stock.csv",)

    def test_a_bound_directory_is_joined_as_its_string(self):
        assert _strings(
            "df = pd.read_csv(os.path.join(ATT, 'stock.csv'))", ATT="/workspace/attachments",
        ) == ("/workspace/attachments/stock.csv",)


class TestBoundedReading:
    @pytest.mark.parametrize("doubling", ["s = s + s", "s += s"])
    def test_a_string_doubled_on_every_line_stays_bounded(self, doubling: str):
        code = "s = 'https://e.example/'\n" + f"{doubling}\n" * 24

        reading = read_cell(code, names={})

        assert all(len(value) <= MAX_RENDERED_CHARS for value in reading.names.values())

    def test_renders_of_a_large_string_stay_within_the_cells_budget(self):
        code = (
            "s = 'https://e.example/'\n" + "s = s + s\n" * 12
            + "".join(f"s + s + '{i}'\n" for i in range(2_000))
        )

        reading = read_cell(code, names={})

        assert sum(len(text) for text in reading.strings) <= MAX_RENDERED_CHARS + len(code)

    def test_a_name_is_never_bound_to_a_string_longer_than_the_cap(self):
        url = "https://e.example/"
        # 13 doublings pass MAX_STRING_CHARS and stay inside the cell's
        # MAX_RENDERED_CHARS, so only the cap can unbind x.
        assert MAX_STRING_CHARS < len(url) * 2**13 and len(url) * 2**14 <= MAX_RENDERED_CHARS

        reading = read_cell(f"x = {url!r}\n" + "x += x\n" * 13 + "f(x)\n", names={})

        assert all(len(value) <= MAX_STRING_CHARS for value in reading.names.values())

    def test_a_query_built_past_the_render_budget_lists_no_fragment(self):
        spent = "s = 'https://e.example/'\n" + "s = s + s\n" * 11 + "s + s\n" * 20
        code = spent + textwrap.dedent("""
            base = "SELECT qty FROM stock"
            a = query_minds_data("SELECT qty FROM stock WHERE site = " + site)
            b = query_minds_data(base + " WHERE site = 'S1'")
        """)

        assert read_cell(code, names={}).queries == ()

    def test_a_long_concatenation_is_charged_once_for_what_it_renders(self):
        pieces = " + ".join(repr("a" * 1_000) for _ in range(60))
        code = f"sql = 'SELECT ' + {pieces} + ' FROM t'\ncur.execute(sql)\n"

        reading = read_cell(code, names={})

        assert reading.queries == ("SELECT " + "a" * 60_000 + " FROM t",)

    @pytest.mark.parametrize("depth", [3_000, 10_000])
    def test_a_cell_nested_past_what_the_parser_or_reader_can_walk_reads_as_nothing(self, depth: int):
        reading = read_cell("x = " + "-" * depth + "1", names={"k": "v"})

        assert reading.queries == ()
        assert reading.names == {"k": "v"}

    def test_a_large_name_read_in_many_placeholders_is_checked_for_sql_once(self, monkeypatch):
        base = "https://example.com/" + "a" * 60_000
        code = f"base = {base!r}\n" + 'x = f"' + "{base}" * 10_000 + '"\n'
        checked = 0
        check = cell_queries_mod.is_sql

        def counted(text: str) -> bool:
            nonlocal checked
            checked += 1
            return check(text)

        monkeypatch.setattr(cell_queries_mod, "is_sql", counted)
        start = time.perf_counter()

        read_cell(code, names={})

        # Checking the name once per placeholder reads 60,000 characters
        # 10,000 times; a check per distinct string reads them once.
        assert checked < 10_000
        assert time.perf_counter() - start < TIME_BOUND_SECONDS

    def test_a_large_query_read_in_many_placeholders_stays_within_the_cells_budget(self):
        base = "SELECT " + "a" * 60_000 + " FROM t"
        code = f"base = {base!r}\n" + 'x = f"' + "{base}" * 200 + '"\n'

        reading = read_cell(code, names={})

        assert sum(len(text) for text in reading.strings) <= MAX_RENDERED_CHARS + len(code)

    def test_many_placeholders_in_a_long_cell_read_in_linear_time(self):
        code = "a=1\n" * 10_000 + 'x = f"' + "{a}" * 10_000 + '"\n'
        start = time.perf_counter()

        reading = read_cell(code, names={})

        assert time.perf_counter() - start < TIME_BOUND_SECONDS
        assert reading.strings == ("{a}" * 10_000,)

    def test_placeholders_read_as_written_past_non_ascii_text_and_crlf_line_ends(self):
        code = (
            'note = "Coût de stock — été"\r\n'
            'label = f"Prix: {prix[\'clé\']} € — {total!r:>{width}}"\r\n'
            'q = f"""SELECT qty\r\n    FROM stock WHERE région = {région}"""\r\n'
            "cur.execute(q)\r\n"
        )

        reading = read_cell(code, names={})

        assert "Prix: {prix['clé']} € — {total!r:>{width}}" in reading.strings
        assert reading.queries == ("SELECT qty\n    FROM stock WHERE région = {région}",)

    def test_a_cell_that_cannot_be_encoded_reads_as_nothing(self):
        reading = read_cell("x = '\ud800' + f'{a}'", names={"k": "v"})

        assert reading.queries == ()
        assert reading.strings == ()
        assert reading.names == {"k": "v"}


class TestWhatACellCalls:
    def test_web_tools_the_data_helper_and_environment_reads_are_noted(self):
        reading = read_cell(textwrap.dedent("""
            import os
            # web_fetch( in a comment does not count
            results = web_search("steel price")
            client = Erp(token=os.environ["DS_ERP_TOKEN"], region=os.getenv("REGION"))
        """), names={})

        assert reading.calls_web_tool
        assert not reading.calls_query_minds_data
        assert reading.environment_variables == ("DS_ERP_TOKEN", "REGION")

    def test_a_comment_naming_a_web_tool_is_not_a_call(self):
        assert not read_cell("# web_search('x')\nprint(1)", names={}).calls_web_tool


def test_strip_url_keeps_scheme_host_and_path():
    assert strip_url("https://user:p@ss@example.com:8443/a/b?x=1#frag") == "https://example.com:8443/a/b"
    assert strip_url("{base}/search?q={term}") == "{base}/search"


@pytest.mark.parametrize("password", ["Xk9#2mQ", "Xk9?2mQ", "Xk9/2mQ", "p@ss"])
def test_strip_url_drops_a_password_that_holds_a_url_delimiter(password: str):
    assert strip_url(f"https://erp:{password}@example.com/rates?x=1") == "https://example.com/rates"


def test_strip_url_keeps_an_at_sign_in_the_path():
    assert strip_url("https://www.youtube.com/@mindsdb/videos") == "https://www.youtube.com/@mindsdb/videos"


class TestLabels:
    @pytest.mark.parametrize("same_cell", [False, True], ids=["earlier-binding", "same-cell-binding"])
    def test_a_printed_bound_name_is_a_label(self, same_cell: bool):
        if same_cell:
            reading = read_cell("caption = 'qty'\nprint(caption)\nprint(stock)", names={})
        else:
            earlier = read_cell("caption = 'qty'", names={})
            reading = read_cell("print(caption)\nprint(stock)", names=earlier.names)

        assert reading.labels == ("qty",)

    def test_a_name_rebound_before_printing_labels_only_its_current_string(self):
        reading = read_cell("caption = 'old'\ncaption = 'qty'\nprint(caption)", names={})

        assert reading.labels == ("qty",)

    def test_a_printed_identifier_remains_a_caption_when_it_also_names_a_column(self):
        reading = read_cell("print('qty')\nprint(stock['qty'])", names={})

        assert reading.labels == ("qty",)

    def test_conditional_values_of_a_printed_name_are_labels(self):
        reading = read_cell("caption = 'qty'\nif changed:\n    caption = 'count'\nprint(caption)", names={})

        assert reading.labels == ("qty", "count")

    @pytest.mark.parametrize("caption", ["'qty'", "caption"])
    @pytest.mark.parametrize("same_cell", [False, True], ids=["earlier-helper", "same-cell-helper"])
    def test_a_stable_helper_carries_its_printed_labels(self, caption: str, same_cell: bool):
        definition = f"caption = 'qty'\ndef show():\n    print({caption})\n    print(stock)\n"
        if same_cell:
            reading = read_cell(definition + "show()", names={})
        else:
            earlier = read_cell(definition, names={})
            reading = read_cell("show()", names=earlier.names, functions=earlier.functions)

        assert reading.labels == ("qty",)
        assert reading.functions["show"].labels == ("qty",)

    def test_a_wrapped_helper_carries_its_printed_labels(self):
        earlier = read_cell(textwrap.dedent('''
            def show():
                print('qty')
                print(stock)
            def wrapped():
                show()
        '''), names={})

        reading = read_cell("wrapped()", names=earlier.names, functions=earlier.functions)

        assert reading.labels == ("qty",)

    @pytest.mark.parametrize("call", ["show()", "wrapped()"])
    def test_a_literal_helper_caption_survives_rebinding_the_printed_data(self, call: str):
        earlier = read_cell(textwrap.dedent('''
            def show():
                print('qty')
                print(stock)
            def wrapped():
                show()
        '''), names={})

        reading = read_cell(
            "stock = pd.Series([250, 90, 250], name='qty')\n" + call,
            names=earlier.names, functions=earlier.functions,
        )

        assert reading.labels == ("qty",)
        assert reading.functions["show"].literal_labels == ("qty",)
        assert reading.functions["wrapped"].literal_labels == ("qty",)

    def test_literal_captions_survive_data_changes_while_global_captions_do_not(self):
        earlier = read_cell(textwrap.dedent('''
            caption = 'old'
            def show():
                print('qty')
                print(caption)
                print(stock)
        '''), names={})

        reading = read_cell("caption = 'new'\nstock = frame\nshow()", names=earlier.names, functions=earlier.functions)

        assert reading.labels == ("qty",)
        assert reading.functions["show"].labels == ("qty",)

    @pytest.mark.parametrize("rebind", ["show = other", "print = other"])
    def test_rebinding_a_caption_callable_clears_a_wrappers_inherited_literal_label(self, rebind: str):
        earlier = read_cell(textwrap.dedent('''
            def show():
                print('qty')
                print(stock)
            def wrapped():
                show()
        '''), names={})

        reading = read_cell(rebind + "\nwrapped()", names=earlier.names, functions=earlier.functions)

        assert reading.labels == ()
        assert reading.functions["wrapped"].literal_labels == ()

    def test_a_helper_with_a_changed_global_carries_no_stale_caption(self):
        earlier = read_cell("caption = 'qty'\ndef show():\n    print(caption)", names={})

        reading = read_cell("caption = 'different'\nshow()", names=earlier.names, functions=earlier.functions)

        assert reading.labels == ()
        assert reading.functions["show"].labels == ()

    def test_a_bound_printed_path_stays_a_source_when_it_is_also_read(self):
        reading = read_cell("path = 'stock.csv'\nprint(path)\ndf = pd.read_csv(path)", names={})

        assert reading.labels == ()
        assert "stock.csv" in reading.strings

    def test_a_helpers_printed_path_stays_a_source_when_it_is_also_read(self):
        definition = textwrap.dedent('''
            def load():
                path = 'stock.csv'
                print(path)
                return pd.read_csv(path)
        ''')
        earlier = read_cell(definition, names={})

        reading = read_cell("load()", names=earlier.names, functions=earlier.functions)

        assert earlier.strings == ("stock.csv",)
        assert reading.strings == ("stock.csv",)
        assert reading.labels == ()
        assert reading.functions["load"].labels == ()

    def test_a_string_only_printed_or_drawn_is_a_label(self):
        reading = read_cell(textwrap.dedent("""
            print('Saved to rates.csv')
            plt.title('Stock by site.png')
            print(f'Wrote {path}')
            df = pd.read_csv('stock.csv')
            print('stock.csv')
            print(open('notes.txt').read())
        """), names={})

        assert reading.labels == ("Saved to rates.csv", "Stock by site.png", "Wrote {path}")
