"""Routes that run their work in a sync helper keep the route's contract.

A coroutine route ``x`` hands its work to a helper ``_x`` that runs in a unit
(cowork/db/units.py). The route, not the helper, is what FastAPI documents,
and the helper's authorization depends on the principal the route passes it.
"""
from __future__ import annotations

import inspect

import pytest
from fastapi.routing import APIRoute

from cowork.api.v1.endpoints import project_files, skills
from cowork.server import create_app


def test_a_route_whose_work_moved_into_its_helper_keeps_its_description():
    """FastAPI takes a route's OpenAPI description from the endpoint's own
    docstring, so a docstring left on the helper documents nothing."""
    undocumented = []
    for route in create_app().routes:
        if not isinstance(route, APIRoute):
            continue
        module = inspect.getmodule(route.endpoint)
        helper = getattr(module, f"_{route.endpoint.__name__}", None)
        if (
            inspect.isfunction(helper)
            and helper.__module__ == module.__name__
            and inspect.getdoc(helper)
            and not route.description
        ):
            undocumented.append(f"{sorted(route.methods)} {route.path}")

    assert undocumented == []


@pytest.mark.parametrize("module", [project_files, skills], ids=lambda module: module.__name__)
def test_a_helper_that_authorizes_by_principal_is_always_handed_one(module):
    """A default would let a caller that forgets the argument run with no
    principal, the weakest one, and nothing would fail."""
    defaulted = [
        name
        for name, function in vars(module).items()
        if inspect.isfunction(function)
        and function.__module__ == module.__name__
        and name.startswith("_")
        and "principal" in inspect.signature(function).parameters
        and inspect.signature(function).parameters["principal"].default is not inspect.Parameter.empty
    ]

    assert defaulted == []
