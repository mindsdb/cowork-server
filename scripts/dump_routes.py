"""Dump every FastAPI route with its method and dependency chain.

Seed script for the ENG-1558 authorization map (auth/docs/endpoint-authorization-map.md),
run with the same tenancy mode the map is scoped to (org):

    COWORK_TENANCY_MODE=org python -m scripts.dump_routes > /tmp/cowork_routes.csv

`cowork.server.create_app()` runs at import time and branches its route set on
`COWORK_TENANCY_MODE` (default: `local`). Local mode mounts channel-plugin
webhook routers (`_install_channels` in `cowork/server.py`) that an org
deployment never exposes, so running this without `COWORK_TENANCY_MODE=org`
set dumps a superset of the surface the map's cowork section is scoped to.

Columns (`path, methods, checks`) match auth's, mindshub_inference's, and
mindshub_services's dump_routes.py. Cowork-server's real gate for org-mode
deployments is `TrustedHeaderMiddleware` (cowork/principal.py), registered
globally when COWORK_TENANCY_MODE=org -- middleware wraps every route
uniformly and isn't visible to per-route Depends() introspection at all, so
`checks` here is even less informative on its own than inference's: it only
shows a route's own `Depends()` chain (e.g. `get_principal`, which reads the
identity the middleware already built), not the middleware gate itself or
any in-handler role check like `can_manage_org`.
"""

from __future__ import annotations

import csv
import sys

from fastapi.routing import APIRoute

from cowork.server import app


def dependency_names(route: APIRoute) -> list[str]:
    return [getattr(dep.call, "__qualname__", str(dep.call)) for dep in route.dependant.dependencies]


def main() -> None:
    rows = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        rows.append(
            {
                "path": route.path,
                "methods": ",".join(sorted(route.methods - {"HEAD"})),
                "checks": ",".join(dependency_names(route)),
            }
        )
    rows.sort(key=lambda r: r["path"])

    writer = csv.DictWriter(sys.stdout, fieldnames=["path", "methods", "checks"])
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
