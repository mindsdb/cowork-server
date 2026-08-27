"""Org-mode Google Drive Picker page — near-direct port of desktop's
`pickerPage()` (cowork/src/main/drive-picker-service.ts).

Desktop reports PICKED/CANCEL/ERROR back to its own loopback HTTP server
(`POST /result`) because the picker page and the Electron main process are
different processes with no shared JS context. Here the picker page and the
tab that opened it are both same-origin browser windows, so that HTTP round
trip is replaced with a plain `window.opener.postMessage(...)` call — see
`_REPORT_RESULT_JS` below. Message shape posted back to the opener:

    {source: "cowork-drive-picker", state, files: [...]}       # picked
    {source: "cowork-drive-picker", state, files: []}           # cancelled
    {source: "cowork-drive-picker", state, error: "..."}        # in-widget error
    {source: "cowork-drive-picker", state, cancelled: true}     # gapi/picker failed to load
    {source: "cowork-drive-picker", state, signal: "suspected-account-mismatch"}

Escaping is a hard requirement, not a nice-to-have (mirrors the comments in
the original TS). Every dynamic value (`state`, `access_token`, `api_key`,
`app_id`, `account_email`, `file_ids`) is carried into the page as an
`html.escape()`'d HTML attribute — never interpolated directly into inline
`<script>` source — and the client script reads them back via
`dataset`/`JSON.parse` instead of bare embedded literals. `html.escape()`
is a sanitizer static analysis tooling actually recognizes (unlike a
bespoke JS-string-escaping helper), so this closes CodeQL's reflected-XSS
finding for real rather than just being safe in practice: the values here
were already correctly neutralized before this change (JSON-encoded, with
every literal less-than sign additionally replaced by its unicode escape
to block a `</script>` breakout), CodeQL just couldn't verify that itself
since that substitution was bespoke, not a library call it recognizes.
`escapeHtml` is additionally used client-side anywhere a value is written
into innerHTML rather than read as a plain script value.

Message shape posted back to the opener is `{type: "drive-picker-result",
result: {...}}`, where `result` is exactly the `DrivePickerResult` shape
`cowork`'s web `pickDriveFilesWeb()` (host.ts) resolves its promise with
directly — `{ok: true, files: [...], newFiles: [...]}` or `{ok: false,
reason: "..."}`. `files` and `newFiles` are the same array here: the web
path has no persisted-grant merge step (unlike desktop's oauthPickDriveFiles
in main/index.ts, where `files` is the connection's full accumulated grant
and `newFiles` is just this pick), but useGoogleDrivePicker.js reads
`newFiles` unconditionally, so it must always be present. No
separate "main process" step exists on the web side to translate a raw
PICKED/CANCEL/ERROR event the way Electron's loopback server does, so this
page has to emit the final, already-interpreted shape itself. The mapping
mirrors Electron's own `/result` handler exactly for consistency between
platforms (`drive-picker-service.ts`): a Cancel click inside the widget, and
even the Google Picker script failing to load at all, both resolve as
`{ok: true, files: []}` rather than an error — only a genuine in-widget
Action.ERROR (most commonly an active-account mismatch) resolves as
`{ok: false, reason: "..."}`.
"""
from __future__ import annotations

import html
import json


def _json_for_script(value) -> str:
    return json.dumps(value).replace("<", "\\u003c")


def render_picker_error_page(message: str) -> str:
    """Shown in the popup itself when the session lookup fails (expired /
    already-used / unknown picker link) — there's no live token to embed
    here, so this never loads the Google Picker script at all."""
    safe = html.escape(message)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Pick Google Drive files</title>
<style>
  :root {{ color-scheme: light dark; }}
  html, body {{ margin: 0; padding: 0; height: 100%; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    display: grid; place-items: center; padding: 40px;
    background: #FAFAFA; color: #0E0F10;
  }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #080d18; color: #E8EDF7; }}
  }}
  .card {{ max-width: 420px; text-align: center; }}
  h1 {{ font-size: 20px; font-weight: 600; margin: 0 0 10px; letter-spacing: -0.01em; color: #d64545; }}
  p {{ font-size: 14px; line-height: 1.5; margin: 0; color: #6B6F73; }}
  .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         background: #d64545; margin-right: 8px; vertical-align: middle; }}
</style></head>
<body>
  <div class="card">
    <h1><span class="dot"></span>Could not open Google Drive</h1>
    <p>{safe}</p>
  </div>
<script>
(function () {{
  if (window.opener) {{
    window.opener.postMessage(
      {{ type: 'drive-picker-result', result: {{ ok: false, reason: {_json_for_script(message)} }} }},
      window.location.origin
    );
  }}
}})();
</script>
</body></html>"""


def render_picker_page(
    *,
    access_token: str,
    api_key: str,
    app_id: str,
    account_email: str,
    state: str,
    file_ids: list[str] | None = None,
) -> str:
    safe_account_email_html = html.escape(account_email)

    # Carried in as html.escape()'d attributes (a sanitizer static analysis
    # recognizes) instead of interpolated into inline <script> source — see
    # this module's docstring. json.dumps() on file_ids first so the client
    # gets the same array back via JSON.parse(dataset.fileIds).
    data_attrs = {
        "data-state": state,
        "data-access-token": access_token,
        "data-api-key": api_key,
        "data-app-id": app_id,
        "data-account-email": account_email,
        "data-file-ids": json.dumps(file_ids or []),
    }
    picker_attrs_html = " ".join(
        f'{attr_name}="{html.escape(value, quote=True)}"' for attr_name, value in data_attrs.items()
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Pick Google Drive files</title>
<style>
  :root {{ color-scheme: light dark; }}
  html, body {{ margin: 0; padding: 0; height: 100%; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    display: grid; place-items: center; padding: 40px;
    background: #FAFAFA; color: #0E0F10;
  }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #080d18; color: #E8EDF7; }}
    p {{ color: #8A97AE; }}
  }}
  .card {{ max-width: 420px; text-align: center; }}
  h1 {{ font-size: 20px; font-weight: 600; margin: 0 0 10px; letter-spacing: -0.01em; }}
  p {{ font-size: 14px; line-height: 1.5; margin: 0; color: #6B6F73; }}
  .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         background: #1F9CB0; margin-right: 8px; vertical-align: middle; }}
  .err .dot {{ background: #d64545; }}
  .err p {{ color: #d64545; }}
</style></head>
<body>
  <div class="card" id="status" {picker_attrs_html}>
    <h1><span class="dot"></span>Opening Google Drive picker…</h1>
    <p>A Google file picker will open in a moment, using your {safe_account_email_html} connection.</p>
  </div>
<script src="https://apis.google.com/js/api.js"></script>
<script>
(function () {{
  var CARD = document.getElementById('status');
  var STATE = CARD.dataset.state;
  var ACCESS_TOKEN = CARD.dataset.accessToken;
  var API_KEY = CARD.dataset.apiKey;
  var APP_ID = CARD.dataset.appId;
  var ACCOUNT_EMAIL = CARD.dataset.accountEmail;
  var FILE_IDS = JSON.parse(CARD.dataset.fileIds);
  var OPENER_ORIGIN = window.location.origin;

  // The browser HTML-decodes data-* attributes for us on the way into
  // .dataset, so these are already the real values here — no separate
  // <script>-breakout escaping step needed on this side (that was only
  // ever a concern for interpolating values directly into script source,
  // which this page no longer does). ACCOUNT_EMAIL still needs its own
  // HTML escaping below wherever it's written into innerHTML (see
  // escapeHtml()) — the attribute decode above only gets it back to its
  // real value, it doesn't make it safe for a second HTML context.
  function escapeHtml(value) {{
    return String(value).replace(/[&<>"']/g, function (c) {{
      return {{ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }}[c];
    }});
  }}

  function setStatus(title, body, isError) {{
    var card = document.getElementById('status');
    card.className = isError ? 'card err' : 'card';
    card.innerHTML = '<h1><span class="dot"></span>' + title + '</h1><p>' + body + '</p>';
  }}

  // Same-origin popup + opener, so the HTTP round trip desktop's loopback
  // server needed is unnecessary here — post straight back to the tab that
  // opened this one, scoped to our own origin (never '*'). `result` is the
  // final DrivePickerResult shape the opener's promise resolves with
  // directly — see this module's docstring for the PICKED/CANCEL/ERROR
  // mapping and why it has to be fully resolved here, not just relayed.
  function reportResult(result) {{
    if (!window.opener) return;
    window.opener.postMessage(
      {{ type: 'drive-picker-result', result: result }},
      OPENER_ORIGIN
    );
  }}

  // Most common cause of the in-widget ERROR action: the browser's active
  // Google account differs from the connected account (ACCOUNT_EMAIL) —
  // Google's picker widget renders under whichever account is ambient in
  // this browser session, not the one the access token above is scoped to,
  // and 403s instead of showing the file browser. This is a genuine,
  // widget-confirmed error (the widget loaded and then reported one), so
  // it's fine to end the flow here — unlike the load-timeout signal below,
  // which is only a guess.
  function reportPickerLoadFailure() {{
    setStatus(
      'Could not open Google Drive',
      'This is usually caused by ' + escapeHtml(ACCOUNT_EMAIL) + ' not being the active Google account in this browser. '
        + 'Switch to that account (check the avatar menu on a Google page), close this tab, and try again from Cowork.',
      true
    );
    // Not escapeHtml(ACCOUNT_EMAIL) here — the receiving side renders this
    // as plain text, not innerHTML, so escaping would show literal HTML
    // entities in it.
    reportResult({{ ok: false, reason: 'Google Picker could not open — the browser\\u2019s active Google account may not match ' + ACCOUNT_EMAIL + '.' }});
  }}

  // A static Google 403 error page rendered inside the picker's iframe has
  // no picker JS running in it, so it can never emit PICKED/CANCEL/ERROR —
  // Action.ERROR only fires once the widget itself has loaded and then hit
  // a problem, so it can't catch this case. This can't be told apart, from
  // here, from a widget that loaded fine and is just sitting there while
  // the user browses — the iframe is cross-origin, onload fires identically
  // for both, and there is no Action.LOADED. So on timeout this only
  // signals a suspicion to the opener — it must never close the picker or
  // end the flow itself, or a user who simply takes longer than this to
  // pick a file loses their in-progress selection.
  var PICKER_LOAD_TIMEOUT_MS = 9000;

  function buildAndShowPicker() {{
    var google = window.google;
    var views = [];

    if (FILE_IDS.length > 0) {{
      views.push(new google.picker.DocsView(google.picker.ViewId.DOCS).setFileIds(FILE_IDS));
    }}
    views.push(new google.picker.DocsView(google.picker.ViewId.DOCS));
    views.push(new google.picker.DocsView(google.picker.ViewId.DOCS).setOwnedByMe(false));
    views.push(new google.picker.DocsView(google.picker.ViewId.DOCS).setEnableDrives(true));

    var userActed = false;
    var loadTimeoutId = null;
    function markUserActed() {{
      userActed = true;
      if (loadTimeoutId !== null) {{ clearTimeout(loadTimeoutId); loadTimeoutId = null; }}
    }}

    var builder = new google.picker.PickerBuilder()
      .setOAuthToken(ACCESS_TOKEN)
      .setDeveloperKey(API_KEY)
      .setAppId(APP_ID)
      .setTitle('Choose files from ' + ACCOUNT_EMAIL)
      .enableFeature(google.picker.Feature.MULTISELECT_ENABLED)
      .enableFeature(google.picker.Feature.SUPPORT_DRIVES)
      .setCallback(function (data) {{
        if (data.action === google.picker.Action.PICKED) {{
          markUserActed();
          var files = (data.docs || []).map(function (doc) {{
            return {{ id: doc.id, name: doc.name, mimeType: doc.mimeType, iconUrl: doc.iconUrl, url: doc.url, resourceKey: doc.resourceKey || null }};
          }});
          setStatus(
            files.length + ' file' + (files.length === 1 ? '' : 's') + ' selected',
            'You can close this tab and return to MindsHub Cowork.'
          );
          // `newFiles` mirrors desktop's oauthPickDriveFiles shape (see
          // main/index.ts): there, `files` is the connection's full
          // accumulated grant and `newFiles` is just this pick. The web
          // path has no persisted-grant merge step, so the two are the
          // same array here — but useGoogleDrivePicker.js reads `newFiles`
          // unconditionally (it's the "this action" scope), so it must
          // always be present, not just `files`.
          reportResult({{ ok: true, files: files, newFiles: files }});
        }} else if (data.action === google.picker.Action.CANCEL) {{
          markUserActed();
          setStatus('Picker closed', 'You can close this tab and return to MindsHub Cowork.');
          reportResult({{ ok: true, files: [], newFiles: [] }});
        }} else if (data.action === google.picker.Action.ERROR) {{
          markUserActed();
          reportPickerLoadFailure();
        }}
      }});
    views.forEach(function (v) {{ builder.addView(v); }});
    var picker = builder.build();
    picker.setVisible(true);

    loadTimeoutId = setTimeout(function () {{
      if (userActed) return;
      loadTimeoutId = null;
      // Advisory only — must NOT go through reportResult()/'drive-picker-
      // result', which the opener treats as terminal. This is only a
      // suspicion (see the comment above buildAndShowPicker on why it can't
      // be told apart from a widget the user is just still browsing), and
      // ending the flow here would drop an in-progress selection. Nothing
      // consumes this distinct type today; it's a hook for the opener to
      // surface a hint later without risking a false-terminal resolve.
      if (window.opener) {{
        window.opener.postMessage({{ type: 'drive-picker-suspected-mismatch' }}, OPENER_ORIGIN);
      }}
    }}, PICKER_LOAD_TIMEOUT_MS);
  }}

  window.onload = function () {{
    if (!window.gapi) {{
      setStatus('Could not load Google Picker', 'Your connection to Google may be blocked. Close this tab and try again.', true);
      // Matches desktop's own drive-picker-service.ts exactly: the script
      // failing to load resolves as "picked nothing" (ok: true, no files),
      // not an error — kept consistent across platforms even though the
      // status card above visibly flags it.
      reportResult({{ ok: true, files: [], newFiles: [] }});
      return;
    }}
    window.gapi.load('picker', {{
      callback: buildAndShowPicker,
      onerror: function () {{
        setStatus('Could not load Google Picker', 'Close this tab and try again.', true);
        reportResult({{ ok: true, files: [], newFiles: [] }});
      }},
    }});
  }};
}})();
</script>
</body></html>"""
