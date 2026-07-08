from __future__ import annotations

import html as html_lib
import json
from datetime import datetime, timedelta, timezone
from textwrap import dedent
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import HTTPException

from cowork.common.settings.app_settings import ConnectorSettings, OAuthSettings
from cowork.schemas.connectors import OAuthStartResponse
from cowork.services.connectors.oauth import pkce as pkce_utils
from cowork.services.connectors.oauth.config import GOOGLE_SERVICES
from cowork.services.connectors.oauth.state import OAuthStateStore

_GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"


class GoogleOAuthService:
    def _store(self, settings: OAuthSettings) -> OAuthStateStore:
        return OAuthStateStore(settings.state_path)

    def _redirect_uri(self, service: str, settings: OAuthSettings) -> str:
        return f"{settings.server_origin.rstrip('/')}/api/v1/connectors/oauth/{service}/callback"

    def start(self, service: str, settings: OAuthSettings) -> OAuthStartResponse:
        if not settings.google_client_id or not settings.google_client_secret:
            raise HTTPException(status_code=400, detail="Google OAuth credentials are not configured.")

        cfg = GOOGLE_SERVICES[service]
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
        )

        auth_url = _GOOGLE_AUTH_ENDPOINT + "?" + urlencode({
            "client_id": settings.google_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "access_type": "offline",
            "prompt": "consent",
            "scope": " ".join(cfg.scopes),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        })

        return OAuthStartResponse(auth_url=auth_url, redirect_uri=redirect_uri, started_at=started_at)

    def callback(self, service: str, code: str, state: str, error: str, settings: OAuthSettings) -> str:
        cfg = GOOGLE_SERVICES[service]
        store = self._store(settings)
        service_label = service.replace("-", " ").title()

        if error:
            store.clear_pending(service, error=f"Google sign-in returned: {error}")
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
            store.clear_pending(service, error="Google sign-in state did not match the pending request.")
            return _callback_page(
                f"{service_label} connection could not be verified",
                "CoWork rejected the callback because the Google sign-in state did not match.",
                success=False,
            )

        if not code:
            store.clear_pending(service, error="Google sign-in did not return an authorization code.")
            return _callback_page(
                f"{service_label} connection could not be completed",
                "Google did not return an authorization code.",
                success=False,
            )

        if not settings.google_client_id or not settings.google_client_secret:
            store.clear_pending(service, error="Google OAuth credentials are not configured.")
            return _callback_page(
                f"{service_label} connection is not configured",
                "Google OAuth credentials are not configured on this server.",
                success=False,
            )

        started_at = str(pending.get("startedAt", "")).strip()
        if started_at:
            try:
                started_dt = datetime.fromisoformat(started_at)
                if datetime.now(timezone.utc) - started_dt > timedelta(minutes=20):
                    store.clear_pending(service, error="Google sign-in timed out before it completed.")
                    return _callback_page(
                        f"{service_label} sign-in expired",
                        "That Google sign-in request took too long. Start the connection again.",
                        success=False,
                    )
            except ValueError:
                pass

        try:
            token_data = self._exchange_code(
                code=code,
                client_id=settings.google_client_id,
                client_secret=settings.google_client_secret,
                redirect_uri=str(pending.get("redirectUri") or self._redirect_uri(service, settings)),
                verifier=str(pending.get("verifier", "")),
            )
            access_token = str(token_data.get("access_token", "")).strip()
            if not access_token:
                raise HTTPException(status_code=502, detail="Token exchange did not return an access token.")

            userinfo = self._fetch_userinfo(access_token)
            account_email = str(userinfo.get("email", "")).strip()
            account_name = str(userinfo.get("name", "")).strip()
            connection_name = account_email or cfg.engine

            expires_in = int(token_data.get("expires_in", 0) or 0)
            expires_at = (
                (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
                if expires_in else ""
            )

            from anton.core.datasources.data_vault import LocalDataVault
            LocalDataVault(ConnectorSettings().vault_dir).save(
                cfg.engine,
                connection_name,
                {
                    "auth_type": "oauth",
                    "access_token": access_token,
                    "refresh_token": str(token_data.get("refresh_token", "")).strip(),
                    "token_type": str(token_data.get("token_type", "Bearer")).strip(),
                    "scope": str(token_data.get("scope", "")).strip(),
                    "expires_at": expires_at,
                    "account_email": account_email,
                    "account_name": account_name,
                },
            )
        except HTTPException as exc:
            store.clear_pending(service, error=str(exc.detail))
            return _callback_page(
                f"{service_label} connection failed",
                "An error occurred during the sign-in flow. Return to CoWork and try again.",
                success=False,
            )
        except Exception as exc:
            store.clear_pending(service, error=str(exc))
            return _callback_page(
                f"{service_label} connection failed",
                "CoWork could not finish the Google sign-in flow.",
                success=False,
            )

        store.clear_pending(service, connection_name=connection_name)
        return _callback_page(
            f"{service_label} connected",
            f"{account_name or account_email or 'Your Google account'} is now connected. You can close this tab and return to CoWork.",
            success=True,
        )

    def _exchange_code(
        self,
        *,
        code: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        verifier: str,
    ) -> dict[str, Any]:
        return _json_request(
            _GOOGLE_TOKEN_ENDPOINT,
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

    def _fetch_userinfo(self, access_token: str) -> dict[str, Any]:
        return _json_request(
            _GOOGLE_USERINFO_ENDPOINT,
            headers={"Authorization": f"Bearer {access_token}"},
        )


def _json_request(
    url: str,
    *,
    method: str = "GET",
    data: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Accept": "application/json", **(headers or {})}
    body = None
    if data is not None:
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
        raise HTTPException(status_code=502, detail=f"Google OAuth request failed: {detail}") from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail="Could not reach Google OAuth services") from exc


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
            <div class="pill">{safe_state}</div>
            <h1>{safe_title}</h1>
            <p>{safe_message}</p>
          </main>
        </body>
        </html>
        """
    ).strip()


google_service = GoogleOAuthService()
