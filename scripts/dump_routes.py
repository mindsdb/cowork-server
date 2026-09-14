"""Dump every FastAPI route with its method and dependency chain.

Seed script for the ENG-1558 authorization map (auth/docs/endpoint-authorization-map.md),
run with the same tenancy mode the map is scoped to (org):

    COWORK_TENANCY_MODE=org python -m scripts.dump_routes > /tmp/cowork_routes.csv

`cowork.server.create_app()` runs at import time and branches its route set on
`COWORK_TENANCY_MODE` (default: `local`). Local mode mounts channel-plugin
webhook routers (`_install_channels` in `cowork/server.py`) that an org
deployment never exposes, so running this without `COWORK_TENANCY_MODE=org`
set dumps a superset of the surface the map's cowork section is scoped to.

Columns (`path, methods, permission, checks`). The first three match auth's,
mindshub_inference's, and mindshub_services's dump_routes.py; `checks` is
cowork-server's own and lists the rest of the route's `Depends()` chain.

`permission` is the ENG-2094 declaration -- the `Permission` class the route
passed to `require()`, read off the `permission_cls` stamp the dependency
carries. That is what AC7 re-derives the map's cowork-server section from, so
it has to name the class rather than the closure `require()` returns; a route
can list more than one, since a route-level `dependencies=[...]` is added to
its router's rather than substituted for it.

`checks` still under-reports what actually gates a request. The org-mode
identity gate is `TrustedHeaderMiddleware` (cowork/principal.py), registered
globally when COWORK_TENANCY_MODE=org, and middleware wraps every route
uniformly rather than appearing in per-route `Depends()` introspection. Nor
does either column show an in-handler role check like `can_manage_org`.
"""

from __future__ import annotations

import csv
import sys

from fastapi.routing import APIRoute, APIWebSocketRoute

from cowork.api.v1.route_walker import declared_permissions, route_key
from cowork.server import app

Route = APIRoute | APIWebSocketRoute


def dependency_names(route: Route) -> list[str]:
    """Every dependency on the route EXCEPT its permission declaration, which
    gets its own column."""
    return [
        getattr(dep.call, "__qualname__", str(dep.call))
        for dep in route.dependant.dependencies
        if getattr(dep.call, "permission_cls", None) is None
    ]


def main() -> None:
    rows = []
    for route in app.routes:
        # APIWebSocketRoute too: it carries a declaration the same way an
        # APIRoute does, so leaving it out would under-report the surface.
        if not isinstance(route, (APIRoute, APIWebSocketRoute)):
            continue
        path, methods = route_key(route)
        rows.append(
            {
                "path": path,
                "methods": ",".join(m for m in methods if m != "HEAD"),
                "permission": ",".join(cls.__name__ for cls in declared_permissions(route)),
                "checks": ",".join(dependency_names(route)),
            }
        )
    rows.sort(key=lambda r: (r["path"], r["methods"]))

    writer = csv.DictWriter(sys.stdout, fieldnames=["path", "methods", "permission", "checks"])
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
