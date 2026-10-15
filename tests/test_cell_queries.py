from __future__ import annotations

import textwrap
import time

import pytest

from cowork.harnesses.anton_harness.cell_queries import (
    MAX_RENDERED_CHARS,
    MAX_STRING_CHARS,
    HttpRequest,
    is_sql,
    read_cell,
    strip_url,
)


def _queries(code: str, **names: str) -> list[str]:
    return list(read_cell(textwrap.dedent(code), names=names).queries)


def _requests(code: str, **names: str) -> list[str]:
    return [str(request) for request in read_cell(textwrap.dedent(code), names=names).requests]


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


class TestBoundedReading:
    @pytest.mark.parametrize("doubling", ["s = s + s", "s += s"])
    def test_a_string_doubled_on_every_line_stays_bounded(self, doubling: str):
        code = "def unused():\n    s = 'ab'\n" + f"    {doubling}\n" * 24

        reading = read_cell(code, names={})

        assert all(len(value) <= MAX_RENDERED_CHARS for value in reading.names.values())

    def test_renders_of_a_large_string_stay_within_the_cells_budget(self):
        code = "s = 'ab'\n" + "s = s + s\n" * 12 + "".join(f"s + s + '{i}'\n" for i in range(2_000))

        reading = read_cell(code, names={})

        assert sum(len(text) for text in reading.strings) <= MAX_RENDERED_CHARS + len(code)

    def test_a_name_is_never_bound_to_a_string_longer_than_the_cap(self):
        start = time.perf_counter()

        reading = read_cell('x = "a"\n' + "x += x\n" * 17 + "f(x)\n", names={})

        assert all(len(value) <= MAX_STRING_CHARS for value in reading.names.values())
        assert time.perf_counter() - start < 1.0

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
