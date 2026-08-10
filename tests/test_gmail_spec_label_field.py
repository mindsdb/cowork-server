from cowork.services.connectors.specs._registry import registry


class TestGmailSpecHasNoLabelField:
    def test_no_field_named_label_in_any_method(self):
        raw = registry.get_connectors()["gmail"]
        for method in raw["form"]["methods"]:
            names = [f["name"] for f in method.get("fields", [])]
            assert "label" not in names
