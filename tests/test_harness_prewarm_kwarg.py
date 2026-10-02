"""Scratchpad pre-warm is requested only from an anton that declares it."""
from dataclasses import dataclass

from anton.core.session import ChatSessionConfig

from cowork.harnesses.anton_harness.harness import _prewarm_scratchpad_kwarg


def test_requests_prewarm_when_anton_declares_the_field():
    if "prewarm_scratchpad" not in ChatSessionConfig.__dataclass_fields__:
        return  # older anton: the skew test below covers this build
    assert _prewarm_scratchpad_kwarg(ChatSessionConfig) == {"prewarm_scratchpad": True}


def test_omits_the_kwarg_on_an_anton_without_it():
    @dataclass
    class OldConfig:
        harness: str | None = None

    assert _prewarm_scratchpad_kwarg(OldConfig) == {}
