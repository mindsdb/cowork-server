"""Connection identity: readable slugs, dedup, and secret classification.

Covers the ENG-508 fix — a saved connection gets a meaningful, stable name
derived from its identity field (gmail → email) instead of a random slug, and
the record carries an explicit ``secure_keys`` list so the email stays readable
while the app password is masked.
"""
from anton.core.datasources.data_vault import LocalDataVault

from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.identity import (
    VAULT_KEEP_SENTINEL,
    connection_display_name,
    derive_connection_name,
    resolve_keep_sentinels,
    secure_keys_for,
    spec_secret_fields,
)
from cowork.services.connectors.persist import (
    persist_connection,
    set_connection_label,
)


def _save(vault, *args):
    """Adapter: persist_connection takes vault as a keyword."""
    return persist_connection(*args, vault=vault)

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
        slug = _save(vault, "gmail", "app-password", "", GMAIL_CREDS)
        assert slug == "user-gmail-com"  # not a random gmail-<uuid6>
        rec = vault.read_record("gmail", slug)
        assert rec["fields"]["email"] == "user@gmail.com"
        assert rec["secure_keys"] == ["app_password"]

    def test_explicit_name_wins_over_derived(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        slug = _save(vault, "gmail", "app-password", "Support", GMAIL_CREDS)
        assert slug == "Support"

    def test_same_account_dedups_in_place(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        _save(vault, "gmail", "app-password", "", GMAIL_CREDS)
        # Re-connect the same address (rotated password) — must update in place,
        # not create a second random-slug duplicate.
        rotated = {**GMAIL_CREDS, "app_password": "zzzz yyyy xxxx wwww"}
        _save(vault, "gmail", "app-password", "", rotated)
        conns = vault.list_connections()
        assert len(conns) == 1
        assert vault.load("gmail", "user-gmail-com")["app_password"] == "zzzz yyyy xxxx wwww"

    def test_no_identity_field_falls_back_to_random_slug(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        # OAuth method: no name_from → random fallback (still saved, with secure_keys).
        slug = _save(
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
        slug_a = _save(vault, "gmail", "app-password", "Inbox", a)
        slug_b = _save(vault, "gmail", "app-password", "Inbox", b)
        assert slug_a == "Inbox"
        assert slug_b == "Inbox-2"  # NOT overwritten
        assert len(vault.list_connections()) == 2
        assert vault.load("gmail", "Inbox")["email"] == "support@acme.com"
        assert vault.load("gmail", "Inbox-2")["email"] == "personal@gmail.com"

    def test_same_explicit_name_same_account_updates_in_place(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        a = {"email": "support@acme.com", "app_password": "old1 old1 old1 old1"}
        rotated = {"email": "support@acme.com", "app_password": "new2 new2 new2 new2"}
        s1 = _save(vault, "gmail", "app-password", "Inbox", a)
        s2 = _save(vault, "gmail", "app-password", "Inbox", rotated)
        assert s1 == s2 == "Inbox"  # same identity → update in place
        assert len(vault.list_connections()) == 1
        assert vault.load("gmail", "Inbox")["app_password"] == "new2 new2 new2 new2"

    def test_derived_distinct_emails_never_collide(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        _save(
            vault, "gmail", "app-password", "",
            {"email": "a@gmail.com", "app_password": "aaaa aaaa aaaa aaaa"},
        )
        _save(
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
        _save(
            vault, "gmail", "app-password", "Inbox",
            {"email": "u@x.com", "app_password": "REALPW"},
        )
        # Edit: keep the password (sentinel), no other change.
        _save(
            vault, "gmail", "app-password", "Inbox",
            {"email": "u@x.com", "app_password": VAULT_KEEP_SENTINEL},
        )
        assert len(vault.list_connections()) == 1
        # The real password is preserved — NOT the literal sentinel.
        assert vault.load("gmail", "Inbox")["app_password"] == "REALPW"

    def test_edit_changing_identity_still_updates_named_record(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        _save(
            vault, "gmail", "app-password", "Inbox",
            {"email": "a@x.com", "app_password": "PW"},
        )
        # Edit changes the email but keeps the password (sentinel) → updates the
        # SAME record in place (an edit targets the named connection), not a suffix.
        _save(
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


class TestConnectionLabel:
    """A human label ("Support") names a connection without changing its
    identity/slug, settable at save time (form field) or after (agent tool)."""

    def test_label_param_stored_as_meta_not_in_slug_or_secrets(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        slug = persist_connection(
            "gmail", "app-password", "", GMAIL_CREDS, label="Support", vault=vault
        )
        assert slug == "user-gmail-com"  # identity slug, not the label
        rec = vault.read_record("gmail", slug)
        assert rec["fields"]["_label"] == "Support"
        assert "_label" not in rec["secure_keys"]

    def test_label_from_credentials_field_is_extracted(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        creds = {**GMAIL_CREDS, "label": "Personal"}
        slug = persist_connection("gmail", "app-password", "", creds, vault=vault)
        rec = vault.read_record("gmail", slug)
        assert rec["fields"]["_label"] == "Personal"
        # the raw "label" field is not persisted as a credential
        assert "label" not in rec["fields"]

    def test_label_preserved_when_later_save_omits_it(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        persist_connection("gmail", "app-password", "", GMAIL_CREDS, label="Support", vault=vault)
        # Re-save (rotate password) without a label → existing label carried forward.
        persist_connection(
            "gmail", "app-password", "",
            {**GMAIL_CREDS, "app_password": "zzzz zzzz zzzz zzzz"}, vault=vault,
        )
        assert vault.load("gmail", "user-gmail-com")["_label"] == "Support"

    def test_set_connection_label_updates_in_place(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        slug = persist_connection("gmail", "app-password", "", GMAIL_CREDS, vault=vault)
        assert set_connection_label("gmail", slug, "Support", vault=vault) is True
        assert vault.load("gmail", slug)["_label"] == "Support"
        # identity + secret untouched
        assert vault.load("gmail", slug)["email"] == "user@gmail.com"
        assert vault.read_record("gmail", slug)["secure_keys"] == ["app_password"]

    def test_set_connection_label_missing_connection_returns_false(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        assert set_connection_label("gmail", "nope", "X", vault=vault) is False


class TestDisplayName:
    """The card/detail display name: label → identity → (slug fallback client-side)."""

    def test_helper_priority(self):
        assert connection_display_name({"_label": "Support", "email": "a@x.com"}) == "Support"
        assert connection_display_name({"email": "a@x.com"}) == "a@x.com"
        assert connection_display_name({"account_email": "o@x.com"}) == "o@x.com"
        assert connection_display_name({"host": "h", "database": "d"}) == "h/d"
        assert connection_display_name({"client_id": "x"}) is None

    def test_list_display_name(self, tmp_path, monkeypatch):
        vault = LocalDataVault(tmp_path)
        persist_connection(
            "gmail", "app-password", "", {"email": "a@x.com", "app_password": "p"},
            label="Support", vault=vault,
        )
        persist_connection(
            "gmail", "app-password", "", {"email": "b@x.com", "app_password": "p"}, vault=vault,
        )
        svc = ConnectionsService()
        monkeypatch.setattr(svc, "_vault", lambda: vault)
        by_name = {s.name: s.display_name for s in svc.list()}
        assert by_name["a-x-com"] == "Support"   # label preferred
        assert by_name["b-x-com"] == "b@x.com"    # else the identity

    def test_get_surfaces_display_name_and_hides_label_field(self, tmp_path, monkeypatch):
        vault = LocalDataVault(tmp_path)
        persist_connection(
            "gmail", "app-password", "", {"email": "a@x.com", "app_password": "p"},
            label="Support", vault=vault,
        )
        svc = ConnectionsService()
        monkeypatch.setattr(svc, "_vault", lambda: vault)
        detail = svc.get("gmail", "a-x-com")
        assert detail.display_name == "Support"
        assert "_label" not in detail.fields            # not rendered as a raw `_`-field row
        assert detail.fields["label"] == "Support"      # echoed back so the edit form pre-fills
        assert detail.fields["app_password"] == VAULT_KEEP_SENTINEL  # still masked


class TestOAuthIdentity:
    """OAuth connections store the account email under `account_email`; it should
    drive a readable slug, and the email fetch is best-effort (never blocks)."""

    def test_account_email_drives_slug_and_token_masked(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        slug = _save(
            vault, "google_drive", None, "",
            {"account_email": "u@acme.com", "access_token": "toktoktok", "auth_type": "oauth"},
        )
        assert slug == "u-acme-com"  # not a random google_drive-<uuid6>
        rec = vault.read_record("google_drive", slug)
        assert "access_token" in rec["secure_keys"]
        assert "account_email" not in rec["secure_keys"]  # identity stays readable

    def test_no_account_email_falls_back_to_random(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        slug = _save(
            vault, "google_drive", None, "",
            {"access_token": "toktoktok", "auth_type": "oauth"},
        )
        assert slug.startswith("google_drive-")  # graceful random fallback

    def test_account_email_helper_is_best_effort(self, monkeypatch):
        from cowork.services.connectors.oauth.google import google_service

        assert google_service.account_email("") == ""
        monkeypatch.setattr(google_service, "_fetch_userinfo", lambda t: {"email": "u@acme.com"})
        assert google_service.account_email("tok") == "u@acme.com"

        def _boom(_t):
            raise RuntimeError("email scope not granted")

        monkeypatch.setattr(google_service, "_fetch_userinfo", _boom)
        assert google_service.account_email("tok") == ""  # never raises
