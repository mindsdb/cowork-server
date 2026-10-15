from __future__ import annotations

import textwrap
import time

import pytest

from cowork.harnesses.anton_harness.cell_queries import (
    MAX_RENDERED_CHARS,
    MAX_STRING_CHARS,
    FunctionReading,
    HttpRequest,
    is_sql,
    read_cell,
    strip_url,
)


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
            environment_variables=("DS_ERP_TOKEN",),
            calls_query_minds_data=True,
            reads_data=True,
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

        assert not is_sql("/* a" * 200_000)
        assert time.perf_counter() - start < 1.0

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

        assert "https://api.example.com" in reading.strings
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
        start = time.perf_counter()

        reading = read_cell('x = "https://e.example/"\n' + "x += x\n" * 17 + "f(x)\n", names={})

        assert all(len(value) <= MAX_STRING_CHARS for value in reading.names.values())
        assert time.perf_counter() - start < 1.0

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

    def test_a_large_name_read_in_many_placeholders_reads_in_linear_time(self):
        base = "https://example.com/" + "a" * 60_000
        code = f"base = {base!r}\n" + 'x = f"' + "{base}" * 10_000 + '"\n'
        start = time.perf_counter()

        read_cell(code, names={})

        assert time.perf_counter() - start < 1.0

    def test_a_large_query_read_in_many_placeholders_stays_within_the_cells_budget(self):
        base = "SELECT " + "a" * 60_000 + " FROM t"
        code = f"base = {base!r}\n" + 'x = f"' + "{base}" * 200 + '"\n'

        reading = read_cell(code, names={})

        assert sum(len(text) for text in reading.strings) <= MAX_RENDERED_CHARS + len(code)

    def test_many_placeholders_in_a_long_cell_read_in_linear_time(self):
        code = "a=1\n" * 10_000 + 'x = f"' + "{a}" * 10_000 + '"\n'
        start = time.perf_counter()

        reading = read_cell(code, names={})

        assert time.perf_counter() - start < 1.0
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
