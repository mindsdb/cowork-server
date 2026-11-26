"""Host-name parsing shared by the loopback guard and the rebinding check.

``guards.require_local`` and ``host_middleware.RebindingHostMiddleware`` each
read a request's Host against the same question — is this a name a page
cannot be served from — so they share one parser and one trusted-name set
here rather than keeping their own copies in step by hand.
"""

from __future__ import annotations

from urllib.parse import urlparse

#: Names trusted without being an IP literal. A session-scoped test fixture
#: (tests/conftest.py) extends this for the lifetime of the test run, to
#: trust the Host the test HTTP client sends.
TRUSTED_HOSTS = frozenset({"localhost"})


def hostname(value: str) -> str:
    """The host part of a Host header value, case-folded; "" if unparsable.

    An unbalanced IPv6 literal such as "[::1" names no host.
    """
    try:
        return (urlparse(f"//{value}").hostname or "").casefold()
    except ValueError:
        return ""
