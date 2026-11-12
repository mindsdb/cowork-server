# anton's browser contract fixtures

anton owns these files, in its `tests/fixtures/browser_contract/`. They are copied here verbatim, so a change there is a change here: copy the new file in and run `tests/test_browser.py`.

- `browser_config.json`: a turn request's `params.browser` (the router block this server writes with `BrowserConfig.to_dict`), the in-app browser's constructor arguments (what `desktop_browser_config` passes), and the forms both refuse.
- `session_event.json`: the cloud turn's `{"kind": "browser", "session_id"}` line. The turn queue relays it as a `browser` step, and this server keeps the session id only.

Each file is `{description, examples: {<name>: {where, kind, refused?, body}}}`, as anton's own README describes.
