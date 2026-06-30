"""Connection identity: readable slugs, dedup, and secret classification.

Covers the ENG-508 fix — a saved connection gets a meaningful, stable name
derived from its identity field (gmail → email) instead of a random slug, and
the record carries an explicit ``secure_keys`` list so the email stays readable
while the app password is masked.
"""
from anton.core.datasources.data_vault import LocalDataVault

from cowork.handlers.probe import _save_connection_to_vault
from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.identity import (
    VAULT_KEEP_SENTINEL,
    derive_connection_name,
    resolve_keep_sentinels,
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

    def test_unknown_connector_with_no_identity_field_returns_none(self):
        # No name_from and no credential-unique field (email/host) → random fallback.
        assert derive_connection_name("does-not-exist", "m", {"api_token": "x"}) is None


class TestNarrowHeuristic:
    """When a connector declares no name_from, derive only from
    credential-unique fields (email, host[+database+username]) — never from
    tenant/project-level or config fields."""

    def test_email_used_when_no_name_from(self):
        assert derive_connection_name("zzz", "m", {"email": "u@acme.com"}) == "u-acme-com"

    def test_database_host_database_username_combo(self):
        assert (
            derive_connection_name(
                "postgres",
                "host-port",
                {"host": "db.acme.com", "database": "sales", "username": "ro", "password": "x"},
            )
            == "db-acme-com-sales-ro"
        )

    def test_host_alone(self):
        assert derive_connection_name("zzz", "m", {"host": "db.acme.com"}) == "db-acme-com"

    def test_tenant_level_fields_not_used(self):
        # Identify a tenant/project, not the specific credential → stay random.
        for f in ("project_id", "tenant_id", "subdomain", "account_id", "account_name"):
            assert derive_connection_name("zzz", "m", {f: "shared"}) is None

    def test_base_url_and_client_id_not_used(self):
        assert derive_connection_name("zzz", "m", {"base_url": "https://api.x.com"}) is None
        assert derive_connection_name("zzz", "m", {"client_id": "8a93f2c1-7d4e"}) is None

    def test_connection_string_method_has_no_clean_identity(self):
        assert (
            derive_connection_name(
                "postgres", "connection-string", {"connection_string": "postgres://u:p@h/db"}
            )
            is None
        )


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


class TestKeepSentinel:
    """Edit flow: an unchanged secret arrives as the keep-sentinel and must be
    resolved to the stored value, not persisted literally."""

    def test_resolve_keeps_prior_drops_orphan(self):
        prior = {"fields": {"email": "old@x.com", "app_password": "REALPW"}}
        resolved, had = resolve_keep_sentinels(
            {"email": "new@x.com", "app_password": VAULT_KEEP_SENTINEL}, prior
        )
        assert had is True
        assert resolved == {"email": "new@x.com", "app_password": "REALPW"}
        # sentinel with no prior value → dropped (never persisted)
        resolved2, had2 = resolve_keep_sentinels({"x": VAULT_KEEP_SENTINEL}, None)
        assert had2 is True and resolved2 == {}
        # no sentinel → unchanged, not an edit
        resolved3, had3 = resolve_keep_sentinels({"a": "b"}, None)
        assert had3 is False and resolved3 == {"a": "b"}

    def test_edit_keeps_secret_and_updates_in_place(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        _save_connection_to_vault(
            vault, "gmail", "app-password", "Inbox",
            {"email": "u@x.com", "app_password": "REALPW"},
        )
        # Edit: keep the password (sentinel), no other change.
        _save_connection_to_vault(
            vault, "gmail", "app-password", "Inbox",
            {"email": "u@x.com", "app_password": VAULT_KEEP_SENTINEL},
        )
        assert len(vault.list_connections()) == 1
        # The real password is preserved — NOT the literal sentinel.
        assert vault.load("gmail", "Inbox")["app_password"] == "REALPW"

    def test_edit_changing_identity_still_updates_named_record(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        _save_connection_to_vault(
            vault, "gmail", "app-password", "Inbox",
            {"email": "a@x.com", "app_password": "PW"},
        )
        # Edit changes the email but keeps the password (sentinel) → updates the
        # SAME record in place (an edit targets the named connection), not a suffix.
        _save_connection_to_vault(
            vault, "gmail", "app-password", "Inbox",
            {"email": "b@x.com", "app_password": VAULT_KEEP_SENTINEL},
        )
        assert len(vault.list_connections()) == 1
        rec = vault.load("gmail", "Inbox")
        assert rec["email"] == "b@x.com" and rec["app_password"] == "PW"


class TestGetMaskingFallback:
    """The detail endpoint must mask secrets even for legacy records saved
    before secure_keys was persisted (via the name heuristic)."""

    def test_legacy_record_without_secure_keys_is_masked(self, tmp_path, monkeypatch):
        vault = LocalDataVault(tmp_path)
        # Legacy save: no secure_keys written.
        vault.save(
            "gmail", "legacy",
            {"email": "u@x.com", "app_password": "PLAINTEXTPW", "_connector_id": "gmail"},
        )
        svc = ConnectionsService()
        monkeypatch.setattr(svc, "_vault", lambda: vault)
        detail = svc.get("gmail", "legacy")
        assert detail.fields["email"] == "u@x.com"            # identity stays readable
        assert detail.fields["app_password"] == VAULT_KEEP_SENTINEL  # secret masked
