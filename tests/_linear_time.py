"""Check that a call's cost grows linearly with its input, on any machine.

A fixed time bound fails on a slow runner, under coverage or beside other
xdist workers, though the code did no extra work. The cost of one call at
two input sizes 10x apart doesn't depend on the machine: a linear call costs
about 10x more on the larger input, and a quadratic one about 100x.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Generic, NamedTuple, TypeVar

Input = TypeVar("Input")
Result = TypeVar("Result")

# For inputs 10x apart; it leaves 3x either way.
MAX_LINEAR_GROWTH = 30
# Windows' thread clock ticks every 15.6 ms; 20 ticks keep its rounding near 5%.
_MIN_TIMED_SECONDS = max(0.01, 20 * time.get_clock_info("thread_time").resolution)


class Growth(NamedTuple, Generic[Result]):
    ratio: float
    """How many times more one call cost on the large input than the small."""
    result: Result
    """What the call returned on the large input, for the test's own checks."""


def cpu_seconds_per_call(call: Callable[[], object]) -> float:
    """This thread's CPU time per call, repeating the call until the calls add
    up to _MIN_TIMED_SECONDS so a fast call still measures. CPU, not wall
    time: a busy machine delays the calls without adding work."""
    calls, started = 0, time.thread_time()
    while True:
        call()
        calls += 1
        if (elapsed := time.thread_time() - started) >= _MIN_TIMED_SECONDS:
            return elapsed / calls


def growth(run: Callable[[Input], Result], *, small: Input, large: Input) -> Growth[Result]:
    """How one ``run`` on ``large`` compares in cost with one on ``small``."""
    small_cost = min(cpu_seconds_per_call(lambda: run(small)) for _ in range(3))
    results: list[Result] = []
    large_cost = cpu_seconds_per_call(lambda: results.append(run(large)))
    return Growth(ratio=large_cost / small_cost, result=results[-1])
