from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cowork.coding.contracts import SessionCreateRequest
from cowork.coding.reasoning import (
    advertised_levels,
    check_reasoning_effort,
    resolve_reasoning_effort,
)

# What MindsHub's /v1/models advertised on 2026-09-03: each family has its own
# vocabulary, and haiku has no levels at all.
LISTING = SimpleNamespace(
    ids=["gpt", "sonnet", "gemini", "haiku"],
    efforts={
        "gpt": {"efforts": ["none", "low", "medium", "high", "xhigh", "max"], "default": "medium"},
        "sonnet": {"efforts": ["low", "medium", "high", "max"], "default": "high"},
        "gemini": {"efforts": ["low", "medium", "high"], "default": "high"},
    },
)


def test_levels_come_from_the_listing_and_are_unknown_without_it() -> None:
    assert advertised_levels("gemini", LISTING) == ["low", "medium", "high"]
    assert advertised_levels("haiku", LISTING) == []
    assert advertised_levels("not-listed", LISTING) is None
    assert advertised_levels("gpt", None) is None
    assert advertised_levels("gpt", SimpleNamespace(ids=None, efforts={})) is None


@pytest.mark.parametrize(("model", "effort"), [("gpt", "max"), ("gpt", "none"), ("sonnet", "max"), ("gemini", "low")])
def test_a_level_the_model_advertises_passes(model: str, effort: str) -> None:
    check_reasoning_effort(model, effort, LISTING)


def test_a_level_the_model_does_not_advertise_names_the_levels_it_does() -> None:
    with pytest.raises(ValueError, match='Reasoning effort "max" isn\'t available for gemini. It offers: low, medium, high.'):
        check_reasoning_effort("gemini", "max", LISTING)


def test_a_model_without_levels_rejects_every_level_plainly() -> None:
    with pytest.raises(ValueError, match="haiku doesn't take a reasoning effort setting."):
        check_reasoning_effort("haiku", "high", LISTING)


def test_no_listing_and_no_effort_are_both_accepted() -> None:
    check_reasoning_effort("gemini", "max", None)
    check_reasoning_effort("gemini", None, LISTING)


def test_a_project_default_is_inherited_only_when_the_task_model_offers_it() -> None:
    assert resolve_reasoning_effort("gpt", None, "max", LISTING) == "max"
    assert resolve_reasoning_effort("gemini", None, "max", LISTING) is None
    assert resolve_reasoning_effort("gemini", "low", "max", LISTING) == "low"
    assert resolve_reasoning_effort("not-listed", None, "max", LISTING) == "max"
    assert resolve_reasoning_effort("gemini", None, None, LISTING) is None


def test_a_requested_level_is_checked_before_the_project_default_is_considered() -> None:
    with pytest.raises(ValueError):
        resolve_reasoning_effort("gemini", "max", "high", LISTING)


@pytest.mark.parametrize("effort", ["none", "minimal", "xhigh", "max", "ultra-2"])
def test_the_request_type_accepts_the_gateway_vocabulary(effort: str) -> None:
    assert SessionCreateRequest(prompt="x", path="/tmp/x", reasoning_effort=effort).reasoning_effort == effort


@pytest.mark.parametrize("effort", ["Max", "extra high", "", "x" * 40])
def test_the_request_type_still_rejects_things_that_are_not_a_level(effort: str) -> None:
    with pytest.raises(ValidationError):
        SessionCreateRequest(prompt="x", path="/tmp/x", reasoning_effort=effort)
