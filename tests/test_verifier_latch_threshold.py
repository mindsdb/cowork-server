"""The verifier latch threshold this server hands anton.

anton latches its completion verifier after N verdict calls that produce no
verdict. Its default is two, which suits a long-lived CLI session. This server
builds a fresh ChatSession per message, so the counter never survives to a
second sample and the latch could not engage: a verifier failing the same way
every time re-diagnosed on every message.
"""
from __future__ import annotations

from dataclasses import dataclass

from cowork.harnesses.anton_harness.harness import _verifier_latch_kwarg


@dataclass
class _CurrentConfig:
    verifier_latch_threshold: int | None = None


@dataclass
class _OlderAntonConfig:
    """An anton build predating the field."""

    initial_history: list | None = None


def test_the_threshold_is_passed_when_anton_declares_it():
    assert _verifier_latch_kwarg(_CurrentConfig) == {"verifier_latch_threshold": 1}


def test_an_anton_without_the_field_is_a_no_op():
    """Not a TypeError on every turn, which is what passing an unknown keyword
    to a plain dataclass would cause."""
    assert _verifier_latch_kwarg(_OlderAntonConfig) == {}


def test_a_non_dataclass_is_a_no_op():
    assert _verifier_latch_kwarg(object) == {}
