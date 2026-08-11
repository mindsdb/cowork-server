from cowork.schemas.connectors import (
    ConnectionDetailResponse,
    ConnectionSummaryResponse,
    SaveConnectionResponse,
)


class TestSchemasHaveUserLabel:
    def test_summary_defaults_to_none(self):
        r = ConnectionSummaryResponse(engine="postgres", name="a1b2c3")
        assert r.user_label is None

    def test_detail_defaults_to_none(self):
        r = ConnectionDetailResponse(engine="postgres", name="a1b2c3")
        assert r.user_label is None

    def test_save_response_defaults_to_none(self):
        r = SaveConnectionResponse(status="ok", submission_id="s1", engine="postgres", name="a1b2c3", method=None)
        assert r.user_label is None
