from __future__ import annotations

import html as html_lib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from anton.core.datasources.data_vault import LocalDataVault
from fastapi import HTTPException

from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.schemas.connectors import OAuthConfig, OAuthStartResponse
from cowork.services.connectors.oauth import pkce as pkce_utils
from cowork.services.connectors.oauth.config import OAUTH_SERVICES
from cowork.services.connectors.oauth.state import OAuthStateStore
from cowork.services.connectors.persist import persist_connection
from cowork.services.connectors.specs._registry import registry as spec_registry

import logging as _logging

_log = _logging.getLogger("cowork.oauth")

_SERVICE_CREDENTIAL_ATTRS: dict[str, tuple[str, str]] = {
    "google-drive":     ("google_drive_client_id",     "google_drive_client_secret"),
    "google-calendar":  ("google_calendar_client_id",  "google_calendar_client_secret"),
    "gmail":            ("gmail_client_id",             "gmail_client_secret"),
    "google-ads":       ("google_ads_client_id",        "google_ads_client_secret"),
    "google-analytics": ("google_analytics_client_id",  "google_analytics_client_secret"),
    "linear":           ("linear_client_id",            "linear_client_secret"),
    "github":           ("github_client_id",            "github_client_secret"),
}

# engine name (e.g. "google_drive") → service id (e.g. "google-drive")
_ENGINE_TO_SERVICE: dict[str, str] = {cfg.engine: svc for svc, cfg in OAUTH_SERVICES.items()}


def _fetch_userinfo_google(access_token: str) -> dict[str, Any]:
    return _json_request(
        "https://openidconnect.googleapis.com/v1/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )


def _fetch_userinfo_linear(access_token: str) -> dict[str, Any]:
    """Linear has no REST userinfo endpoint — identity comes from a GraphQL
    query against the authenticated user (`viewer`)."""
    result = _json_request(
        "https://api.linear.app/graphql",
        method="POST",
        json_body={"query": "query { viewer { email name } }"},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    viewer = (result.get("data") or {}).get("viewer") or {}
    return {"email": viewer.get("email", ""), "name": viewer.get("name", "")}


def _fetch_userinfo_github(access_token: str) -> dict[str, Any]:
    """GitHub's `email` is frequently null — the app only requests `read:user`,
    not `user:email`, and even with that scope a user can keep their email
    private. `login` (the username) is always present and always unique, so it's
    the fallback identity — same intent as email elsewhere, just not a real
    email address."""
    result = _json_request(
        "https://api.github.com/user",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
        },
    )
    login = str(result.get("login") or "").strip()
    email = str(result.get("email") or "").strip()
    name = str(result.get("name") or "").strip()
    return {"email": email or login, "name": name or login}


# engine → identity-fetch function. The one piece of connector onboarding
# that can't be pure spec-JSON data — response shape (REST vs GraphQL) is
# genuinely provider-specific code, not configuration. New OAuth-builtin
# connectors add one entry here.
_USERINFO_FETCHERS: dict[str, Callable[[str], dict[str, Any]]] = {
    "google_drive": _fetch_userinfo_google,
    "google_calendar": _fetch_userinfo_google,
    "gmail": _fetch_userinfo_google,
    "google_ads": _fetch_userinfo_google,
    "google_analytics_4": _fetch_userinfo_google,
    "linear": _fetch_userinfo_linear,
    "github": _fetch_userinfo_github,
}


class OAuthService:
    def _resolve_credentials(self, service: str, settings: OAuthSettings) -> tuple[str, str]:
        id_attr, secret_attr = _SERVICE_CREDENTIAL_ATTRS[service]
        client_id = getattr(settings, id_attr)
        client_secret = getattr(settings, secret_attr)
        if not client_id or not client_secret:
            raise HTTPException(status_code=400, detail=f"OAuth credentials not configured for {service}.")
        return client_id, client_secret

    def get_outcome(self, state: str, settings: OAuthSettings) -> dict | None:
        return self._store(settings).get_outcome(state)

    def clear_outcome(self, state: str, settings: OAuthSettings) -> None:
        self._store(settings).clear_outcome(state)

    def _store(self, settings: OAuthSettings) -> OAuthStateStore:
        return OAuthStateStore(settings.state_path)

    def _redirect_uri(self, service: str, settings: OAuthSettings) -> str:
        return f"{settings.server_origin.rstrip('/')}/api/v1/connectors/oauth/{service}/callback"

    def _oauth_config_for(self, engine: str) -> OAuthConfig | None:
        """The connector's whole OAuth shape (auth_url/token_url/revoke_url/
        scopes/capability flags/...) — sourced from its own spec JSON's
        `browser_oauth_builtin` method, the single canonical description of
        a connector's OAuth behavior. Not a second Python-side copy that can
        silently drift out of sync with it."""
        spec = spec_registry.get_connector(engine)
        if spec is None:
            return None
        for method in spec.form.methods or []:
            if method.id == "browser_oauth_builtin" and method.oauth:
                return method.oauth
        return None

    def start(self, service: str, settings: OAuthSettings, *, client_id: str = "", client_secret: str = "", extra_fields: dict[str, str] | None = None) -> OAuthStartResponse:
        if client_id and client_secret:
            cid, csecret = client_id, client_secret
        else:
            cid, csecret = self._resolve_credentials(service, settings)
        client_id = cid

        cfg = OAUTH_SERVICES[service]
        oauth_cfg = self._oauth_config_for(cfg.engine)
        if oauth_cfg is None:
            raise HTTPException(status_code=500, detail=f"No OAuth configuration found in the spec for {service!r}.")

        verifier = pkce_utils.generate_verifier()
        challenge = pkce_utils.generate_challenge(verifier)
        state = pkce_utils.generate_state()
        redirect_uri = self._redirect_uri(service, settings)
        started_at = datetime.now(timezone.utc).isoformat()

        self._store(settings).set_pending(
            service,
            state=state,
            verifier=verifier,
            redirect_uri=redirect_uri,
            started_at=started_at,
            client_id=cid,
            client_secret=csecret,
            extra_fields=extra_fields,
        )
        self._store(settings).set_outcome(state, {"status": "pending"})

        query_params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(oauth_cfg.scopes),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            **oauth_cfg.extra_auth_params,
        }
        auth_url = oauth_cfg.auth_url + "?" + urlencode(query_params)

        return OAuthStartResponse(auth_url=auth_url, redirect_uri=redirect_uri, started_at=started_at, state=state)

    def callback(self, service: str, code: str, state: str, error: str, settings: OAuthSettings) -> str:
        cfg = OAUTH_SERVICES[service]
        store = self._store(settings)
        service_label = service.replace("-", " ").title()

        if error:
            err_msg = f"{service_label} sign-in returned: {error}"
            store.clear_pending(service, error=err_msg)
            if state:
                store.set_outcome(state, {"status": "error", "error": err_msg})
            return _callback_page(
                f"{service_label} connection was cancelled",
                "You can return to CoWork and try the connection again whenever you are ready.",
                success=False,
            )

        pending = store.get_pending(service)
        if not pending:
            return _callback_page(
                f"{service_label} sign-in expired",
                f"CoWork could not find a pending {service_label} sign-in request. Start the connection again.",
                success=False,
            )

        if not state or state != str(pending.get("state", "")).strip():
            store.clear_pending(service, error=f"{service_label} sign-in state did not match the pending request.")
            return _callback_page(
                f"{service_label} connection could not be verified",
                f"CoWork rejected the callback because the {service_label} sign-in state did not match.",
                success=False,
            )

        if not code:
            store.clear_pending(service, error=f"{service_label} sign-in did not return an authorization code.")
            return _callback_page(
                f"{service_label} connection could not be completed",
                f"{service_label} did not return an authorization code.",
                success=False,
            )

        pending_client_id = str(pending.get("clientId", "")).strip()
        pending_client_secret = str(pending.get("clientSecret", "")).strip()
        if pending_client_id and pending_client_secret:
            client_id, client_secret = pending_client_id, pending_client_secret
        else:
            try:
                client_id, client_secret = self._resolve_credentials(service, settings)
            except HTTPException:
                err_msg = f"OAuth credentials not configured for {service}."
                store.clear_pending(service, error=err_msg)
                store.set_outcome(state, {"status": "error", "error": err_msg})
                return _callback_page(
                    f"{service_label} connection is not configured",
                    f"{service_label} OAuth credentials are not configured on this server.",
                    success=False,
                )

        started_at = str(pending.get("startedAt", "")).strip()
        if started_at:
            try:
                started_dt = datetime.fromisoformat(started_at)
                if datetime.now(timezone.utc) - started_dt > timedelta(minutes=20):
                    store.clear_pending(service, error=f"{service_label} sign-in timed out before it completed.")
                    return _callback_page(
                        f"{service_label} sign-in expired",
                        f"That {service_label} sign-in request took too long. Start the connection again.",
                        success=False,
                    )
            except ValueError:
                pass

        oauth_cfg = self._oauth_config_for(cfg.engine)
        if oauth_cfg is None:
            err_msg = f"No OAuth configuration found in the spec for {service!r}."
            store.clear_pending(service, error=err_msg)
            store.set_outcome(state, {"status": "error", "error": err_msg})
            return _callback_page(f"{service_label} connection failed", err_msg, success=False)

        try:
            token_data = self._exchange_code(
                token_url=oauth_cfg.token_url,
                code=code,
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=str(pending.get("redirectUri") or self._redirect_uri(service, settings)),
                verifier=str(pending.get("verifier", "")),
            )
            access_token = str(token_data.get("access_token", "")).strip()
            if not access_token:
                raise HTTPException(status_code=502, detail="Token exchange did not return an access token.")

            fetch_userinfo = _USERINFO_FETCHERS.get(cfg.engine, _fetch_userinfo_google)
            userinfo = fetch_userinfo(access_token)
            account_email = str(userinfo.get("email", "")).strip()
            account_name = str(userinfo.get("name", "")).strip()

            expires_in = int(token_data.get("expires_in", 0) or 0)
            expires_at = (
                (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
                if expires_in else ""
            )

            extra = {k: v for k, v in (pending.get("extraFields") or {}).items() if v}
            new_fields = {
                "auth_type": "oauth",
                "access_token": access_token,
                "refresh_token": str(token_data.get("refresh_token", "")).strip(),
                "token_type": str(token_data.get("token_type", "Bearer")).strip(),
                "scope": str(token_data.get("scope", "")).strip(),
                "expires_at": expires_at,
                "account_email": account_email,
                "account_name": account_name,
                **extra,
            }
            # Routed through the shared persist_connection()/identity.py convention
            # (same one save_connection_direct uses for the Electron PKCE flow) so
            # reconnecting the same account resolves to the same slug — an
            # identity-derived match (is_same_account) updates the existing record
            # in place, carrying forward Google Picker grants and any label,
            # instead of leaving a stale duplicate connection behind.
            connection_name = persist_connection(cfg.engine, "browser_oauth_builtin", "", new_fields)
        except HTTPException as exc:
            err_msg = str(exc.detail)
            store.clear_pending(service, error=err_msg)
            store.set_outcome(state, {"status": "error", "error": err_msg})
            return _callback_page(
                f"{service_label} connection failed",
                "An error occurred during the sign-in flow. Return to CoWork and try again.",
                success=False,
            )
        except Exception as exc:
            err_msg = str(exc)
            _log.exception("OAuth callback failed for %s", service)
            store.clear_pending(service, error=err_msg)
            store.set_outcome(state, {"status": "error", "error": err_msg})
            # The detail goes to the log and the outcome store (both internal);
            # the browser page stays generic so a raw exception message can't
            # leak internals to the end user.
            return _callback_page(
                f"{service_label} connection failed",
                f"Cowork could not finish the {service_label} sign-in flow. Return to Cowork and try again.",
                success=False,
            )

        store.clear_pending(service)
        store.set_outcome(state, {"status": "success", "name": connection_name})
        return _callback_page(
            f"{service_label} connected",
            f"{account_name or account_email or f'Your {service_label} account'} is now connected. You can close this tab and return to CoWork.",
            success=True,
        )

    def revoke(self, engine: str, name: str, connector_settings: ConnectorSettings) -> None:
        if engine not in _ENGINE_TO_SERVICE:
            return
        oauth_cfg = self._oauth_config_for(engine)
        if oauth_cfg is None or not oauth_cfg.supports_revoke or not oauth_cfg.revoke_url:
            _log.debug("Revoke not supported for %s/%s — skipping, local cleanup only", engine, name)
            return
        _log.info("Revoking OAuth token for %s/%s", engine, name)
        try:
            fields = LocalDataVault(Path(connector_settings.vault_dir)).load(engine, name) or {}
        except Exception:
            return
        if fields.get("auth_type") != "oauth":
            return
        token = fields.get("refresh_token", "").strip() or fields.get("access_token", "").strip()
        if not token:
            return
        try:
            request = Request(
                oauth_cfg.revoke_url,
                data=urlencode({"token": token}).encode("utf-8"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with urlopen(request, timeout=10):
                pass
            _log.info("Revoked OAuth token for %s/%s", engine, name)
        except HTTPError as exc:
            _log.warning("Could not revoke OAuth token for %s/%s: %s %s", engine, name, exc.code, exc.reason)
        except Exception as exc:
            _log.warning("Could not revoke OAuth token for %s/%s: %s", engine, name, exc)

    def get_catalogue(self, connector_settings: ConnectorSettings, oauth_settings: OAuthSettings) -> list[dict]:
        try:
            vault = LocalDataVault(Path(connector_settings.vault_dir))
            all_connections = vault.list_connections() or []
        except Exception as exc:
            _log.warning("Could not load vault for catalogue: %s", exc)
            all_connections = []

        state_data = self._store(oauth_settings)._load()
        items = []

        for service_id, cfg in OAUTH_SERVICES.items():
            engine = cfg.engine
            id_attr, secret_attr = _SERVICE_CREDENTIAL_ATTRS[service_id]
            cid = getattr(oauth_settings, id_attr, "")
            csecret = getattr(oauth_settings, secret_attr, "")
            ready = bool(cid and csecret)
            config_error = "" if ready else f"OAuth credentials not configured for {service_id}."

            connections = [
                {"engine": engine, "name": c.get("name", ""), "label": c.get("name", "")}
                for c in all_connections if c.get("engine") == engine
            ]

            entry = state_data.get(service_id) or {}
            service_label = " ".join(w.capitalize() for w in service_id.replace("-", " ").split())

            items.append({
                "id": engine,
                "title": service_label,
                "engine": engine,
                "status": "connected" if connections else ("available" if ready else "needs_config"),
                "connections": connections,
                "connectionCount": len(connections),
                "oauth": {
                    "ready": ready,
                    "configError": config_error,
                    "pending": bool((entry.get("pending") or {}).get("state")),
                    "lastSuccessAt": entry.get("lastSuccessAt", ""),
                    "lastError": entry.get("lastError", ""),
                    "lastErrorAt": entry.get("lastErrorAt", ""),
                    "launchLabel": f"Connect {service_label}",
                    "redirectUri": self._redirect_uri(service_id, oauth_settings),
                    # The web-fallback route slug (e.g. "google-drive") — has
                    # already diverged from the engine name for some
                    # connectors, so callers use this instead of guessing.
                    "serviceId": service_id,
                },
            })

        return items

    def _exchange_code(
        self,
        *,
        token_url: str,
        code: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        verifier: str,
    ) -> dict[str, Any]:
        return _json_request(
            token_url,
            method="POST",
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
                "code_verifier": verifier,
            },
        )


def _json_request(
    url: str,
    *,
    method: str = "GET",
    data: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Accept": "application/json", **(headers or {})}
    body = None
    if json_body is not None:
        request_headers.setdefault("Content-Type", "application/json")
        body = json.dumps(json_body).encode("utf-8")
    elif data is not None:
        request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
        body = urlencode(data).encode("utf-8")
    request = Request(url, data=body, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        detail = raw
        try:
            payload = json.loads(raw)
            detail = (
                payload.get("error_description")
                or payload.get("error", {}).get("message")
                or payload.get("error")
                or raw
            )
        except json.JSONDecodeError:
            pass
        raise HTTPException(status_code=502, detail=f"OAuth request failed: {detail}") from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail="Could not reach OAuth provider") from exc


def _callback_page(title: str, message: str, *, success: bool) -> str:
    accent = "#0f766e" if success else "#b42318"
    safe_title = html_lib.escape(title)
    safe_message = html_lib.escape(message)
    safe_state = "Connected" if success else "Connection failed"
    return dedent(
        f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>{safe_title}</title>
          <style>
            body {{
              margin: 0;
              min-height: 100vh;
              display: grid;
              place-items: center;
              background: #f6f5f1;
              color: #161616;
              font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            main {{
              width: min(92vw, 520px);
              background: #fff;
              border: 1px solid #e7e3da;
              border-radius: 18px;
              padding: 28px 28px 24px;
              box-shadow: 0 20px 60px rgba(20, 17, 12, 0.08);
            }}
            h1 {{
              margin: 0 0 10px;
              font-size: 24px;
              line-height: 1.15;
            }}
            p {{
              margin: 0;
              font-size: 15px;
              line-height: 1.55;
              color: #55514a;
            }}
            .pill {{
              display: inline-flex;
              align-items: center;
              gap: 8px;
              margin-bottom: 14px;
              padding: 6px 10px;
              border-radius: 999px;
              background: #f7f2ee;
              color: {accent};
              font-size: 12px;
              font-weight: 600;
              letter-spacing: 0.02em;
              text-transform: uppercase;
            }}
          </style>
        </head>
        <body>
          <main>
            <span class="pill">{safe_state}</span>
            <h1>{safe_title}</h1>
            <p>{safe_message}</p>
          </main>
        </body>
        </html>
        """
    ).strip()


oauth_service = OAuthService()
