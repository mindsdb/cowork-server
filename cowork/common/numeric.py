"""Numbers read from JSON that another service sends."""
from __future__ import annotations

import math
from typing import Any, Optional


def finite_float(value: Any) -> Optional[float]:
    """``value`` as a finite float, or None when it is not a real number.

    A bool, a string, None or any other non-number reads as None, and so do
    NaN and the infinities. A JSON integer too large for a float reads as None
    too, instead of raising ``OverflowError`` into the caller.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None
