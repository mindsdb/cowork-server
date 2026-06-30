"""Connection identity: readable slugs, dedup, and secret classification.

Covers the ENG-508 fix — a saved connection gets a meaningful, stable name
derived from its identity field (gmail → email) instead of a random slug, and
the record carries an explicit ``secure_keys`` list so the email stays readable
while the app password is masked.
"""
from anton.core.datasources.data_vault import LocalDataVault

from cowork.handlers.probe import _save_connection_to_vault
from cowork.services.connectors.identity import (
    derive_connection_name,
    secure_keys_for,
    spec_secret_fields,
)

GMAIL_CREDS = {"email": "user@gmail.com", "app_password": "abcd efgh ijkl mnop"}


class TestDeriveConnectionName:
    def test_gmail_app_password_uses_email(self):
        assert (
            derive_connection_name("gmail", "app-password", GMAIL_CREDS)
            == "user-gmail-com"
        )

    def test_gmail_service_account_uses_impersonate_email(self):
        creds = {"impersonate_email": "admin@acme.com", "service_account_json": "{}"}
        assert (
            derive_connection_name("gmail", "service-account", creds)
            == "admin-acme-com"
        )

    def test_oauth_method_has_no_name_from_returns_none(self):
        # The OAuth method declares no name_from (identity comes from userinfo) →
        # caller keeps its random fallback.
        assert derive_connection_name("gmail", "oauth", {"client_id": "x"}) is None

    def test_missing_identity_field_returns_none(self):
        assert derive_connection_name("gmail", "app-password", {"app_password": "x"}) is None

    def test_unknown_connector_returns_none(self):
        assert derive_connection_name("does-not-exist", "m", GMAIL_CREDS) is None


class TestSecureKeys:
    def test_gmail_marks_only_app_password_secret(self):
        assert spec_secret_fields("gmail", "app-password") == ["app_password"]
        keys = secure_keys_for("gmail", "app-password", GMAIL_CREDS)
        assert "app_password" in keys
        assert "email" not in keys  # identity must stay readable

    def test_meta_fields_not_marked_secret(self):
        # _connector_id / _method are bookkeeping, not secrets.
        payload = {**GMAIL_CREDS, "_connector_id": "gmail", "_method": "app-password"}
        keys = secure_keys_for("gmail", "app-password", payload)
        assert "_connector_id" not in keys and "_method" not in keys


class TestSaveConnectionToVault:
    def test_saves_with_readable_slug_and_secure_keys(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        slug = _save_connection_to_vault(vault, "gmail", "app-password", "", GMAIL_CREDS)
        assert slug == "user-gmail-com"  # not a random gmail-<uuid6>
        rec = vault.read_record("gmail", slug)
        assert rec["fields"]["email"] == "user@gmail.com"
        assert rec["secure_keys"] == ["app_password"]

    def test_explicit_name_wins_over_derived(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        slug = _save_connection_to_vault(vault, "gmail", "app-password", "Support", GMAIL_CREDS)
        assert slug == "Support"

    def test_same_account_dedups_in_place(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        _save_connection_to_vault(vault, "gmail", "app-password", "", GMAIL_CREDS)
        # Re-connect the same address (rotated password) — must update in place,
        # not create a second random-slug duplicate.
        rotated = {**GMAIL_CREDS, "app_password": "zzzz yyyy xxxx wwww"}
        _save_connection_to_vault(vault, "gmail", "app-password", "", rotated)
        conns = vault.list_connections()
        assert len(conns) == 1
        assert vault.load("gmail", "user-gmail-com")["app_password"] == "zzzz yyyy xxxx wwww"

    def test_no_identity_field_falls_back_to_random_slug(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        # OAuth method: no name_from → random fallback (still saved, with secure_keys).
        slug = _save_connection_to_vault(
            vault, "gmail", "oauth", "", {"client_id": "abc", "client_secret": "shh"}
        )
        assert slug.startswith("gmail-")
        rec = vault.read_record("gmail", slug)
        assert "client_secret" in rec["secure_keys"]
        assert "client_id" not in rec["secure_keys"]


class TestNonDestructiveSave:
    """A save must never overwrite a *different* account's record."""

    def test_same_explicit_name_different_account_suffixes(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        a = {"email": "support@acme.com", "app_password": "aaaa bbbb cccc dddd"}
        b = {"email": "personal@gmail.com", "app_password": "eeee ffff gggg hhhh"}
        slug_a = _save_connection_to_vault(vault, "gmail", "app-password", "Inbox", a)
        slug_b = _save_connection_to_vault(vault, "gmail", "app-password", "Inbox", b)
        assert slug_a == "Inbox"
        assert slug_b == "Inbox-2"  # NOT overwritten
        assert len(vault.list_connections()) == 2
        assert vault.load("gmail", "Inbox")["email"] == "support@acme.com"
        assert vault.load("gmail", "Inbox-2")["email"] == "personal@gmail.com"

    def test_same_explicit_name_same_account_updates_in_place(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        a = {"email": "support@acme.com", "app_password": "old1 old1 old1 old1"}
        rotated = {"email": "support@acme.com", "app_password": "new2 new2 new2 new2"}
        s1 = _save_connection_to_vault(vault, "gmail", "app-password", "Inbox", a)
        s2 = _save_connection_to_vault(vault, "gmail", "app-password", "Inbox", rotated)
        assert s1 == s2 == "Inbox"  # same identity → update in place
        assert len(vault.list_connections()) == 1
        assert vault.load("gmail", "Inbox")["app_password"] == "new2 new2 new2 new2"

    def test_derived_distinct_emails_never_collide(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        _save_connection_to_vault(
            vault, "gmail", "app-password", "",
            {"email": "a@gmail.com", "app_password": "aaaa aaaa aaaa aaaa"},
        )
        _save_connection_to_vault(
            vault, "gmail", "app-password", "",
            {"email": "b@gmail.com", "app_password": "bbbb bbbb bbbb bbbb"},
        )
        names = {c["name"] for c in vault.list_connections()}
        assert names == {"a-gmail-com", "b-gmail-com"}  # distinct, no suffixes
