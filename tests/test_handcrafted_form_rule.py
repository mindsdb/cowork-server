"""When a connector id may take a handcrafted form instead of its stored one.

The request_credentials tool and the form submission both decide this, and
must agree: the tool renders the form, and the submission stages it.
"""
from __future__ import annotations

import pytest

from cowork.services.connectors import custom_connectors
from cowork.services.connectors.custom_connectors import handcrafted_form_allowed


@pytest.mark.parametrize("known, custom, extends, stored_test_request, allowed", [
    (False, False, False, None, True),
    (False, False, True, None, True),
    (True, False, False, None, False),
    (True, False, True, None, False),
    (True, True, False, True, False),
    (True, True, True, True, True),
    (True, True, False, False, True),
], ids=[
    "unknown id", "unknown id follow-up", "built-in", "built-in follow-up",
    "custom first step", "custom follow-up", "custom saved before connection tests",
])
def test_the_rule(known, custom, extends, stored_test_request, allowed):
    assert handcrafted_form_allowed(
        known=known, custom=custom, extends=extends, stored_test_request=stored_test_request,
    ) is allowed


def test_self_hosted_keeps_an_old_custom_connector_as_it_is(monkeypatch):
    monkeypatch.setattr(custom_connectors, "tenant_scope_required", lambda: True)
    assert handcrafted_form_allowed(known=True, custom=True, extends=False, stored_test_request=False) is False
