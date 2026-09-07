"""Unit tests for scripts/dump_routes.py, the ENG-1558 route-dump seed script
for the authorization map (auth/docs/endpoint-authorization-map.md).
"""

import csv
import io
from contextlib import redirect_stdout
from types import SimpleNamespace

from scripts.dump_routes import dependency_names, main


def _route(*calls):
    return SimpleNamespace(dependant=SimpleNamespace(dependencies=[SimpleNamespace(call=c) for c in calls]))


# Module-level, like real FastAPI dependency callables, so __qualname__ is
# just the bare name rather than carrying an enclosing test method's scope.
def _dep_get_principal():
    pass


def _dep_get_tenant_scope():
    pass


class TestDependencyNames:
    def test_empty_dependencies_returns_empty_list(self):
        assert dependency_names(_route()) == []

    def test_named_function_reports_qualname(self):
        assert dependency_names(_route(_dep_get_principal)) == ["_dep_get_principal"]

    def test_preserves_declaration_order(self):
        assert dependency_names(_route(_dep_get_principal, _dep_get_tenant_scope)) == [
            "_dep_get_principal",
            "_dep_get_tenant_scope",
        ]

    def test_falls_back_to_str_when_no_qualname(self):
        # A callable instance has no __qualname__ of its own (only its class
        # does), so this exercises the getattr fallback.
        class Checker:
            def __call__(self):
                pass

            def __str__(self):
                return "checker-instance"

        assert dependency_names(_route(Checker())) == ["checker-instance"]


class TestMain:
    def test_produces_csv_with_expected_header_and_health_route(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        rows = list(csv.DictReader(io.StringIO(buf.getvalue())))

        assert rows, "dump produced no rows"
        assert set(rows[0].keys()) == {"path", "methods", "checks"}

        health_rows = [r for r in rows if r["path"] == "/api/v1/health/"]
        assert health_rows, "expected /api/v1/health/ in the dump"
        assert health_rows[0]["methods"] == "GET"

    def test_no_duplicate_path_method_pairs(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main()
        rows = list(csv.DictReader(io.StringIO(buf.getvalue())))

        seen = set()
        dupes = []
        for r in rows:
            key = (r["path"], r["methods"])
            if key in seen:
                dupes.append(key)
            seen.add(key)
        assert dupes == []
