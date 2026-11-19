"""TLS verification setting for requests to the Minds services."""
from __future__ import annotations

import os

_OFF_VALUES = frozenset({"false", "0", "no", "off"})


def minds_ssl_verify() -> bool:
    """Whether to verify TLS certificates, from `ANTON_MINDS_SSL_VERIFY`.

    Only an explicit off value disables verification: an empty or unknown
    value must not drop TLS checks on a request that carries a credential.
    """
    return os.environ.get("ANTON_MINDS_SSL_VERIFY", "").strip().lower() not in _OFF_VALUES
