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
the original TS): `_json_for_script` escapes `<` so a value containing
`</script>` can't break out of the inline script (this page holds a live
access token), and `escapeHtml` is additionally used anywhere a value is
interpolated into visible markup/innerHTML.
"""
from __future__ import annotations

import html as html_lib
import json


def _json_for_script(value) -> str:
    return json.dumps(value).replace("<", "\\u003c")


def render_picker_page(
    *,
    access_token: str,
    api_key: str,
    app_id: str,
    account_email: str,
    state: str,
    file_ids: list[str] | None = None,
) -> str:
    safe_account_email_html = html_lib.escape(account_email)
    js_state = _json_for_script(state)
    js_access_token = _json_for_script(access_token)
    js_api_key = _json_for_script(api_key)
    js_app_id = _json_for_script(app_id)
    js_account_email = _json_for_script(account_email)
    js_file_ids = _json_for_script(file_ids or [])

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
  <div class="card" id="status">
    <h1><span class="dot"></span>Opening Google Drive picker…</h1>
    <p>A Google file picker will open in a moment, using your {safe_account_email_html} connection.</p>
  </div>
<script src="https://apis.google.com/js/api.js"></script>
<script>
(function () {{
  var STATE = {js_state};
  var ACCESS_TOKEN = {js_access_token};
  var API_KEY = {js_api_key};
  var APP_ID = {js_app_id};
  var ACCOUNT_EMAIL = {js_account_email};
  var FILE_IDS = {js_file_ids};
  var OPENER_ORIGIN = window.location.origin;

  // ACCOUNT_EMAIL ends up in innerHTML below (not just a JS string literal),
  // so it needs HTML escaping too, on top of the <script>-breakout escaping
  // jsonForScript already did when embedding it above.
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
  // opened this one, scoped to our own origin (never '*').
  function reportResult(payload) {{
    if (!window.opener) return;
    window.opener.postMessage(
      Object.assign({{ source: 'cowork-drive-picker', state: STATE }}, payload),
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
    reportResult({{ error: 'Google Picker could not open — the browser\\u2019s active Google account may not match ' + ACCOUNT_EMAIL + '.' }});
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
          reportResult({{ files: files }});
        }} else if (data.action === google.picker.Action.CANCEL) {{
          markUserActed();
          setStatus('Picker closed', 'You can close this tab and return to MindsHub Cowork.');
          reportResult({{ files: [] }});
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
      reportResult({{ signal: 'suspected-account-mismatch' }});
    }}, PICKER_LOAD_TIMEOUT_MS);
  }}

  window.onload = function () {{
    if (!window.gapi) {{
      setStatus('Could not load Google Picker', 'Your connection to Google may be blocked. Close this tab and try again.', true);
      reportResult({{ cancelled: true }});
      return;
    }}
    window.gapi.load('picker', {{
      callback: buildAndShowPicker,
      onerror: function () {{
        setStatus('Could not load Google Picker', 'Close this tab and try again.', true);
        reportResult({{ cancelled: true }});
      }},
    }});
  }};
}})();
</script>
</body></html>"""
