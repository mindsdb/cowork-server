"""No coroutine route takes the request's database session.

A coroutine route runs on the event loop, so a query on the request's session
runs there too. While the pool is full that query holds every request on the
pod for up to POOL_TIMEOUT, liveness included. Coroutine routes depend on the
caller's TenantScope instead and read through ``run_db`` units
(cowork/db/units.py), which wait for a connection in a worker thread.

The routes still on the request session are listed below by exact site, each
with its reason. The found routes are compared with the list both ways and
counted, as tests/test_no_subprocess_static.py does: a new offender fails, and
so does an entry whose route no longer takes the session.
"""
from collections import Counter

from fastapi import APIRouter, Depends, FastAPI
from fastapi.routing import APIRoute, APIWebSocketRoute
from sqlmodel import Session

from cowork.db.scoped import ScopedSession, get_scoped_session
from cowork.db.session import get_session
from cowork.server import create_app

_REQUEST_SESSIONS = (get_session, get_scoped_session)
_ENDPOINTS = "cowork.api.v1.endpoints."

#: (module under cowork.api.v1.endpoints, route function, reason). One entry
#: per route.
ALLOWLIST = (
    (
        "artifacts",
        "reveal_artifact",
        "desktop only (DesktopOnly): org mode refuses it before any I/O",
    ),
    (
        "coding",
        "inference_proxy",
        "desktop only (LoopbackDesktopOnly on the router)",
    ),
    (
        "coding_runtime",
        "runtime_inference_proxy",
        "desktop only (DesktopOnly on the router)",
    ),
    (
        "project_files",
        "upload_project_files",
        "lists the projects inside the pinned-directory inventory it writes the uploads "
        "through, on the loop; moving that inventory to the file store's threads moves both",
    ),
    (
        "settings",
        "write_raw_settings",
        "desktop only (LoopbackDesktopOnly): writes the deployment's dotenv",
    ),
)


def _takes_a_request_session(dependant) -> bool:
    return any(
        sub.call in _REQUEST_SESSIONS or _takes_a_request_session(sub)
        for sub in dependant.dependencies
    )


def coroutine_routes_with_a_request_session(app: FastAPI) -> Counter:
    """(module, function) of each coroutine route whose dependencies, at any
    depth, include the request's session."""
    found: Counter = Counter()
    for route in app.routes:
        if not isinstance(route, (APIRoute, APIWebSocketRoute)):
            continue
        if not route.dependant.is_coroutine_callable:
            continue
        if _takes_a_request_session(route.dependant):
            module = route.endpoint.__module__.removeprefix(_ENDPOINTS)
            found[(module, route.endpoint.__name__)] += 1
    return found


def test_the_coroutine_routes_on_the_request_session_are_the_listed_ones():
    found = coroutine_routes_with_a_request_session(create_app())

    listed = Counter((module, function) for module, function, _reason in ALLOWLIST)
    assert found - listed == Counter(), "a coroutine route takes the request's session"
    assert listed - found == Counter(), "a listed route no longer takes the request's session"


def _app_with(router: APIRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_a_new_coroutine_route_on_the_session_is_found():
    router = APIRouter()

    @router.get("/direct")
    async def direct(session: Session = Depends(get_session)):
        return {}

    @router.get("/scoped")
    async def scoped(session: ScopedSession = Depends(get_scoped_session)):
        return {}

    def nested(scoped: ScopedSession = Depends(get_scoped_session)):
        return scoped

    @router.get("/nested")
    async def through_a_dependency(value=Depends(nested)):
        return {}

    found = coroutine_routes_with_a_request_session(_app_with(router))

    assert found == Counter({(__name__, "direct"): 1, (__name__, "scoped"): 1, (__name__, "through_a_dependency"): 1})


def test_a_sync_route_on_the_session_is_not_counted():
    """A sync route runs in a worker thread, where a wait for the pool holds
    only that thread."""
    router = APIRouter()

    @router.get("/sync")
    def sync(session: Session = Depends(get_session)):
        return {}

    assert coroutine_routes_with_a_request_session(_app_with(router)) == Counter()
