"""When a connector id may take a handcrafted form instead of its stored one.

The request_credentials tool and the form submission both decide this, and
must agree: the tool renders the form, and the submission stages it.
"""
from __future__ import annotations

import pytest

from cowork.services.connectors.custom_connectors import handcrafted_form_allowed


@pytest.mark.parametrize("known, custom, extends, allowed", [
    (False, False, False, True),
    (False, False, True, True),
    (True, False, False, False),
    (True, False, True, False),
    (True, True, False, False),
    (True, True, True, True),
], ids=[
    "unknown id", "unknown id follow-up", "built-in", "built-in follow-up",
    "custom first step", "custom follow-up",
])
def test_the_rule(known, custom, extends, allowed):
    assert handcrafted_form_allowed(known=known, custom=custom, extends=extends) is allowed
