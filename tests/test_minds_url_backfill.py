"""Tests for backfill_minds_url — rewrite the legacy MindsHub host.

The MindsHub default base URL changed from https://mdb.ai to
https://api.mindshub.ai, but no migration updated existing rows. mdb.ai's
/api/v1 path now 404s, so a pre-flip user is stuck with a failing provider
and no UI field to fix it. backfill_minds_url heals those rows on boot.
"""

import json

from cowork.db.session import get_open_session
from cowork.migrations import backfill_minds_url
from cowork.services.settings import SettingService


def _seed(session, key, value):
    SettingService(session).upsert_setting(key, value)


def _value(session, key):
    row = SettingService(session)._fetch_row(key)
    return row.value if row else None


def test_backfill_rewrites_providers_json_mindsurl(tmp_path):
    session = get_open_session()
    try:
        providers = [
            {"type": "anthropic", "apiKey": "***", "isDefault": True},
            {"type": "minds-cloud", "apiKey": "***", "mindsUrl": "https://mdb.ai", "isDefault": False},
        ]
        _seed(session, "providers_json", json.dumps(providers))
        _seed(session, "minds_url", "https://mdb.ai")

        changed = backfill_minds_url(session)
        assert changed is True

        pj = json.loads(_value(session, "providers_json"))
        minds = next(p for p in pj if p["type"] == "minds-cloud")
        assert minds["mindsUrl"] == "https://api.mindshub.ai"
        assert "mdb.ai" not in _value(session, "providers_json")
        assert _value(session, "minds_url") == "https://api.mindshub.ai"
    finally:
        SettingService(session).delete_setting("providers_json")
        SettingService(session).delete_setting("minds_url")


def test_backfill_is_idempotent_and_noop_when_clean(tmp_path):
    session = get_open_session()
    try:
        providers = [{"type": "minds-cloud", "apiKey": "***", "mindsUrl": "https://api.mindshub.ai"}]
        _seed(session, "providers_json", json.dumps(providers))

        # Already canonical → no change reported.
        assert backfill_minds_url(session) is False
        assert "mdb.ai" not in _value(session, "providers_json")

        # Re-introduce the legacy host (simulating a stale "Save settings"),
        # confirm a subsequent boot self-heals, then is a no-op again.
        _seed(session, "providers_json", json.dumps(
            [{"type": "minds-cloud", "apiKey": "***", "mindsUrl": "https://mdb.ai"}]
        ))
        assert backfill_minds_url(session) is True
        assert backfill_minds_url(session) is False
    finally:
        SettingService(session).delete_setting("providers_json")


def test_backfill_handles_trailing_slash_and_http(tmp_path):
    session = get_open_session()
    try:
        _seed(session, "minds_url", "http://mdb.ai/")
        assert backfill_minds_url(session) is True
        assert _value(session, "minds_url") == "https://api.mindshub.ai/"
    finally:
        SettingService(session).delete_setting("minds_url")
