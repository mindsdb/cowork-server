"""Connection identity: readable slugs, dedup, and secret classification.

Covers the ENG-508 fix — a saved connection gets a meaningful, stable name
derived from its identity field (gmail → email) instead of a random slug, and
the record carries an explicit ``secure_keys`` list so the email stays readable
while the app password is masked.
"""
from datetime import datetime, timezone
from pathlib import Path

from anton.core.datasources.data_vault import LocalDataVault

from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.services.connectors.connections import ConnectionsService
from cowork.services.connectors.oauth import google as oauth_google
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
        # `set_connection_label()` now writes `_user_label` (not `_label`) and
        # returns the stored value (post-deduplication), not a bool.
        assert set_connection_label("gmail", slug, "Support", vault=vault) == "Support"
        assert vault.load("gmail", slug)["_user_label"] == "Support"
        # identity + secret untouched
        assert vault.load("gmail", slug)["email"] == "user@gmail.com"
        assert vault.read_record("gmail", slug)["secure_keys"] == ["app_password"]

    def test_set_connection_label_missing_connection_returns_none(self, tmp_path):
        vault = LocalDataVault(tmp_path)
        assert set_connection_label("gmail", "nope", "X", vault=vault) is None


class TestDisplayName:
    """The card/detail display name: derived identity only (email/host) —
    no longer prefers `_label`/`_user_label`; the connection's title comes
    from `user_label` directly now (see TestConnectionDisplayNameNoLongerPrefersLabel
    and Task 17's ConnectionsService.list/get tests)."""

    def test_helper_priority(self):
        assert connection_display_name({"_label": "Support", "email": "a@x.com"}) == "a@x.com"
        assert connection_display_name({"email": "a@x.com"}) == "a@x.com"
        assert connection_display_name({"account_email": "o@x.com"}) == "o@x.com"
        assert connection_display_name({"host": "h", "database": "d"}) == "h/d"
        assert connection_display_name({"client_id": "x"}) is None

    def test_account_name_preferred_for_supabase_and_linear(self):
        # Supabase's account_email is a synthetic `org:<slug>` placeholder, so
        # the human org name is the more useful subtitle there.
        fields = {"account_name": "Acme", "account_email": "org:acme"}
        assert connection_display_name(fields, "supabase") == "Acme"
        # Linear's account_email is likewise synthetic (`<email>:<workspace_id>`,
        # see _fetch_userinfo_linear) — the workspace name is the useful
        # subtitle, not the raw colon-joined identity string.
        linear_fields = {"account_name": "Acme Workspace", "account_email": "user@example.com:org-1"}
        assert connection_display_name(linear_fields, "linear") == "Acme Workspace"
        # Every other engine populates a real account_email — account_name is
        # just a free-text display name there, and preferring it would make
        # two accounts with the same name but different emails
        # indistinguishable (e.g. google/github/posthog).
        assert connection_display_name(fields, "google") == "org:acme"
        assert connection_display_name(fields) == "org:acme"

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
        # display_name is identity-only now — the label no longer changes it.
        assert by_name["a-x-com"] == "a@x.com"
        assert by_name["b-x-com"] == "b@x.com"

    def test_get_surfaces_display_name_and_hides_label_field(self, tmp_path, monkeypatch):
        vault = LocalDataVault(tmp_path)
        persist_connection(
            "gmail", "app-password", "", {"email": "a@x.com", "app_password": "p"},
            label="Support", vault=vault,
        )
        svc = ConnectionsService()
        monkeypatch.setattr(svc, "_vault", lambda: vault)
        detail = svc.get("gmail", "a-x-com")
        assert detail.display_name == "a@x.com"          # identity-only now
        assert "_label" not in detail.fields            # not rendered as a raw `_`-field row
        assert "label" not in detail.fields             # no longer echoed into fields
        # persist_connection assigned a default user_label (the engine id) for
        # this brand-new connection since only the legacy `label` was given,
        # not `user_label` — the default takes precedence over the `_label`
        # fallback because `_user_label` is present (non-empty) on the record.
        assert detail.user_label == "gmail"
        assert detail.fields["app_password"] == VAULT_KEEP_SENTINEL  # still masked


class TestConnectionDisplayNameNoLongerPrefersLabel:
    def test_ignores_label_returns_identity_instead(self):
        fields = {"_label": "Support", "email": "reg@mail.com"}
        assert connection_display_name(fields) == "reg@mail.com"

    def test_still_returns_host_database_identity(self):
        fields = {"host": "db.example.com", "database": "prod_db"}
        assert connection_display_name(fields) == "db.example.com/prod_db"

    def test_returns_none_when_nothing_derivable(self):
        fields = {"_label": "Support", "_connector_id": "gmail"}
        assert connection_display_name(fields) is None


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


class TestOAuthCallbackDedup:
    """The browser OAuth callback must route through the same identity-derived
    slug convention persist_connection uses everywhere else. Regression test for
    the duplicate-Google-Drive-connections bug: reconnecting the same account
    (e.g. re-running the browser sign-in, or using a different Google OAuth
    entry point) must update the existing vault record in place, not leave a
    second "connected" entry behind.
    """

    def _connect(self, monkeypatch, tmp_path, *, state, access_token, email, scope=""):
        monkeypatch.setenv("COWORK_VAULT_DIR", str(tmp_path / "vault"))
        settings = OAuthSettings(
            google_drive_client_id="cid",
            google_drive_client_secret="csecret",
            state_path=str(tmp_path / "oauth_state.json"),
        )
        svc = oauth_google.OAuthService()
        store = svc._store(settings)
        store.set_pending(
            "google-drive",
            state=state,
            verifier="verifier",
            redirect_uri="http://127.0.0.1/callback",
            started_at=datetime.now(timezone.utc).isoformat(),
            client_id="cid",
            client_secret="csecret",
        )
        store.set_outcome(state, {"status": "pending"})

        monkeypatch.setattr(
            oauth_google.OAuthService,
            "_exchange_code",
            lambda self, **kw: {
                "access_token": access_token, "refresh_token": "reftok",
                "expires_in": 3600, "scope": scope,
            },
        )
        monkeypatch.setitem(
            oauth_google._USERINFO_FETCHERS, "google_drive",
            lambda access_token: {"email": email, "name": "User"},
        )
        svc.callback("google-drive", code="authcode", state=state, error="", settings=settings)

    def test_repeat_connection_same_account_dedups(self, monkeypatch, tmp_path):
        self._connect(monkeypatch, tmp_path, state="state-1", access_token="tok1", email="user@gmail.com")
        # Simulate reconnecting the same account (browser sign-in run again).
        self._connect(monkeypatch, tmp_path, state="state-2", access_token="tok2", email="user@gmail.com")

        vault = LocalDataVault(Path(ConnectorSettings().vault_dir))
        conns = vault.list_connections()
        assert len(conns) == 1
        assert conns[0]["name"] == "user-gmail-com"
        assert vault.load("google_drive", "user-gmail-com")["access_token"] == "tok2"

    def test_different_accounts_do_not_collide(self, monkeypatch, tmp_path):
        self._connect(monkeypatch, tmp_path, state="state-1", access_token="tok1", email="a@gmail.com")
        self._connect(monkeypatch, tmp_path, state="state-2", access_token="tok2", email="b@gmail.com")

        vault = LocalDataVault(Path(ConnectorSettings().vault_dir))
        names = {c["name"] for c in vault.list_connections()}
        assert names == {"a-gmail-com", "b-gmail-com"}

    def test_reconnect_with_reordered_scope_still_dedups(self, monkeypatch, tmp_path):
        # Reproduces a live duplicate: Google's token endpoint doesn't
        # guarantee stable word order in the returned `scope` string, so the
        # exact same granted scopes came back in a different order on the
        # second token exchange for the same account.
        scope_a = "https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile openid"
        scope_b = "https://www.googleapis.com/auth/drive.file openid https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile"
        self._connect(
            monkeypatch, tmp_path, state="state-1", access_token="tok1",
            email="martyna@mindsdb.com", scope=scope_a,
        )
        self._connect(
            monkeypatch, tmp_path, state="state-2", access_token="tok2",
            email="martyna@mindsdb.com", scope=scope_b,
        )

        vault = LocalDataVault(Path(ConnectorSettings().vault_dir))
        conns = vault.list_connections()
        assert len(conns) == 1
        assert conns[0]["name"] == "martyna-mindsdb-com"


class TestPersistConnectionUserLabel:
    def test_new_connection_gets_explicit_user_label(self, tmp_path):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        slug = persist_connection(
            "postgres", "host-port", "", {"host": "db.example.com"},
            user_label="prod-db", vault=vault,
        )
        record = vault.read_record("postgres", slug)
        assert record["fields"]["_user_label"] == "prod-db"

    def test_user_label_deduplicated_against_existing(self, tmp_path):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        persist_connection(
            "postgres", "host-port", "", {"host": "a"}, user_label="prod-db", vault=vault,
        )
        slug2 = persist_connection(
            "mysql", "host-port", "", {"host": "b"}, user_label="prod-db", vault=vault,
        )
        record2 = vault.read_record("mysql", slug2)
        assert record2["fields"]["_user_label"] == "prod-db 2"

    def test_editing_keeps_own_label_unbumped(self, tmp_path):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        # `persist_connection()`'s return value IS the vault "name" — it is
        # NOT prefixed with `{connector_id}-` the way anton's `save_connection()`
        # helper builds a slug. For a host-derived identity, `derive_connection_name`
        # returns something like "db-example-com" (no relation to the string
        # "postgres"), so `slug.split("-", 1)[1]` would NOT recover a usable
        # name — it would chop the derived slug itself in an arbitrary place.
        # Pass `slug` straight through as `name` on the second call instead.
        #
        # The second call must submit the SAME credentials (just `host`,
        # nothing added) as the first. `resolve_unique_slug()` only treats
        # this as "the same connection" when `is_same_account()` agrees the
        # non-secret identity matches — and identity comparison includes
        # every non-`_`-prefixed, non-secret field except `expires_at`/`scope`
        # (`_VOLATILE_IDENTITY_FIELDS`). Adding `"port": "5432"` on the second
        # call would make `is_same_account()` return False, so the second
        # `persist_connection()` call would create a sibling record instead
        # of updating the first one in place — `exclude=(connector_id, slug)`
        # would then never actually exclude anything relevant, and the test
        # would pass without exercising the `exclude` behavior at all.
        slug = persist_connection(
            "postgres", "host-port", "", {"host": "db.example.com"},
            user_label="prod-db", vault=vault,
        )
        persist_connection(
            "postgres", "host-port", slug, {"host": "db.example.com"},
            user_label="prod-db", vault=vault,
        )
        record = vault.read_record("postgres", slug)
        assert record["fields"]["_user_label"] == "prod-db"
        # Proves the second call updated the existing record rather than
        # silently creating a sibling.
        assert len(vault.list_connections()) == 1

    def test_random_fallback_is_8_chars(self, tmp_path):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        # Only the random-fallback branch produces a `{connector_id}-{hex}`
        # shaped string — this is the one case where splitting on the first
        # "-" recovers the hex suffix. (An unregistered engine with no
        # recognizable identity field, like `opaque` here, has nothing for
        # `derive_connection_name` to key off, so it falls through to random.)
        slug = persist_connection(
            "unknownengine", None, "", {"opaque": "1"}, vault=vault,
        )
        suffix = slug.split("-", 1)[1]
        assert len(suffix) == 8

    def test_new_connection_gets_default_label_when_none_passed(self, tmp_path):
        # Without this, a connection saved via cowork with no explicit
        # `user_label` in the request (the common case for a first-time
        # "Connect Postgres" through the GUI) would end up with NO label at
        # all — inconsistent with anton, where the CLI prompt always has a
        # default and can never be skipped entirely.
        vault = LocalDataVault(Path(tmp_path) / "vault")
        slug = persist_connection(
            "postgres", "host-port", "", {"host": "db.example.com"}, vault=vault,
        )
        record = vault.read_record("postgres", slug)
        assert record["fields"]["_user_label"] == "postgres"

    def test_editing_without_a_label_does_not_assign_one(self, tmp_path):
        # The `existing is None` guard on the default-label fix above: an
        # existing connection that has no label (e.g. migrated from before
        # this feature) must not silently get one assigned just because a
        # later save didn't pass `user_label` either.
        #
        # Same trap as `test_editing_keeps_own_label_unbumped` above — the
        # credentials on the second call must be IDENTICAL to what's already
        # stored (just `host`), not `host` + a newly-added `port`.
        vault = LocalDataVault(Path(tmp_path) / "vault")
        vault.save("postgres", "legacy", {"host": "db.example.com"})  # no _user_label
        persist_connection(
            "postgres", "host-port", "legacy", {"host": "db.example.com"},
            vault=vault,
        )
        record = vault.read_record("postgres", "legacy")
        assert "_user_label" not in record["fields"]
        assert len(vault.list_connections()) == 1

    def test_new_connection_uses_default_label_over_generic_engine_id(self, tmp_path):
        # ENG-2188: an OAuth connector's fetched account/org/workspace name
        # (e.g. Linear's workspace, Supabase's organization) should title a
        # brand-new connection's tile instead of the generic engine id.
        vault = LocalDataVault(Path(tmp_path) / "vault")
        slug = persist_connection(
            "linear", "browser_oauth_builtin", "", {"account_email": "a@x.com:org-1"},
            default_label="Acme Workspace", vault=vault,
        )
        record = vault.read_record("linear", slug)
        assert record["fields"]["_user_label"] == "Acme Workspace"

    def test_default_label_deduplicated_against_existing(self, tmp_path):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        persist_connection(
            "linear", "browser_oauth_builtin", "", {"account_email": "a@x.com:org-1"},
            default_label="Acme Workspace", vault=vault,
        )
        slug2 = persist_connection(
            "linear", "browser_oauth_builtin", "", {"account_email": "a@x.com:org-2"},
            default_label="Acme Workspace", vault=vault,
        )
        record2 = vault.read_record("linear", slug2)
        assert record2["fields"]["_user_label"] == "Acme Workspace 2"

    def test_default_label_never_overwrites_a_label_the_user_already_set(self, tmp_path):
        # Unlike `user_label`, `default_label` must only ever apply to a
        # genuinely new connection — a reconnect through the same OAuth
        # browser flow (e.g. after a token expires) must not silently
        # clobber a name the user typed in for the tile.
        vault = LocalDataVault(Path(tmp_path) / "vault")
        slug = persist_connection(
            "linear", "browser_oauth_builtin", "", {"account_email": "a@x.com:org-1"},
            default_label="Acme Workspace", vault=vault,
        )
        set_connection_label("linear", slug, "My Renamed Tile", vault=vault)
        persist_connection(
            "linear", "browser_oauth_builtin", slug, {"account_email": "a@x.com:org-1"},
            default_label="Acme Workspace", vault=vault,
        )
        record = vault.read_record("linear", slug)
        assert record["fields"]["_user_label"] == "My Renamed Tile"

    def test_explicit_user_label_still_wins_over_default_label(self, tmp_path):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        slug = persist_connection(
            "linear", "browser_oauth_builtin", "", {"account_email": "a@x.com:org-1"},
            user_label="Explicit", default_label="Acme Workspace", vault=vault,
        )
        record = vault.read_record("linear", slug)
        assert record["fields"]["_user_label"] == "Explicit"


class TestSetConnectionLabelReturnsValue:
    def test_returns_stored_value(self, tmp_path):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        vault.save("gmail", "support", {"email": "a@b.com"})
        result = set_connection_label("gmail", "support", "Support", vault=vault)
        assert result == "Support"

    def test_deduplicates_against_existing_labels(self, tmp_path):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        vault.save("gmail", "acct1", {"email": "a@b.com", "_user_label": "Support"})
        vault.save("gmail", "acct2", {"email": "c@d.com"})
        result = set_connection_label("gmail", "acct2", "Support", vault=vault)
        assert result == "Support 2"

    def test_returns_none_for_missing_connection(self, tmp_path):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        result = set_connection_label("gmail", "missing", "Support", vault=vault)
        assert result is None

    def test_returns_none_for_blank_label(self, tmp_path):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        vault.save("gmail", "support", {"email": "a@b.com"})
        result = set_connection_label("gmail", "support", "   ", vault=vault)
        assert result is None
        assert "_user_label" not in (vault.load("gmail", "support") or {})


class TestServicePopulatesUserLabel:
    def test_list_includes_user_label(self, tmp_path, monkeypatch):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        vault.save("postgres", "a1b2c3", {"host": "x", "_user_label": "prod-db"})
        svc = ConnectionsService()
        monkeypatch.setattr(svc, "_vault", lambda: vault)
        results = svc.list()
        assert results[0].user_label == "prod-db"

    def test_list_falls_back_to_legacy_label(self, tmp_path, monkeypatch):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        vault.save("gmail", "acct1", {"email": "a@b.com", "_label": "Support"})
        svc = ConnectionsService()
        monkeypatch.setattr(svc, "_vault", lambda: vault)
        results = svc.list()
        assert results[0].user_label == "Support"

    def test_get_includes_user_label(self, tmp_path, monkeypatch):
        vault = LocalDataVault(Path(tmp_path) / "vault")
        vault.save("postgres", "a1b2c3", {"host": "x", "_user_label": "prod-db"})
        svc = ConnectionsService()
        monkeypatch.setattr(svc, "_vault", lambda: vault)
        detail = svc.get("postgres", "a1b2c3")
        assert detail.user_label == "prod-db"

