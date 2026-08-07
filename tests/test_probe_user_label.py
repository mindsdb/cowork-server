"""_extract_connection_label_fields() — the shared pop-out-of-credentials
helper both persist_connection call sites in probe.py use for label/user_label.

Full end-to-end coverage of the streaming handler's `response.completed`
payload is not exercised here — there's no existing test harness driving
`ProbeHandler.run()`'s generator (it needs a workspace, LLM client, and
submission store), and building one is out of scope for this unit-level
check. This covers the extraction logic itself, which is what both call
sites depend on.
"""
from cowork.handlers.probe import _extract_connection_label_fields


class TestExtractConnectionLabelFields:
    def test_pops_user_label_and_label_leaving_rest(self):
        credentials = {"host": "x", "label": "Old", "user_label": "prod-db"}
        label, user_label = _extract_connection_label_fields(credentials)
        assert label == "Old"
        assert user_label == "prod-db"
        assert credentials == {"host": "x"}

    def test_falls_back_to_underscore_prefixed_keys(self):
        credentials = {"host": "x", "_label": "Old", "_user_label": "prod-db"}
        label, user_label = _extract_connection_label_fields(credentials)
        assert label == "Old"
        assert user_label == "prod-db"
        assert credentials == {"host": "x"}

    def test_no_short_circuit_both_keys_popped_even_when_first_is_set(self):
        # Regression: an `a or b` pattern would short-circuit and leave the
        # second key sitting in `credentials` whenever the first is truthy,
        # so it would be saved as an ordinary (bogus) credential field.
        credentials = {"label": "Old", "_label": "Stale", "user_label": "New", "_user_label": "StaleToo"}
        label, user_label = _extract_connection_label_fields(credentials)
        assert label == "Old"
        assert user_label == "New"
        assert credentials == {}

    def test_missing_keys_return_empty_strings(self):
        credentials = {"host": "x"}
        label, user_label = _extract_connection_label_fields(credentials)
        assert label == ""
        assert user_label == ""
        assert credentials == {"host": "x"}
