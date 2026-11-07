# Browser router contract fixtures

These files pin what this server sends the browser router and what it reads back. The router keeps copies of the same files and checks its routes against them, so a change here is a contract change on both sides.

`tests/test_browser.py` builds each request with this server's own code and compares it to the file, and parses each answer into the model `cowork/services/browser.py` reads it with.

Each file is `{description, examples: {<name>: {where, kind, status?, body}}}`, the same format as anton's fixtures in `../browser_contract/`. `kind` is `request` (what this server sends) or `response` (what the router answers).

| File | Covers |
|---|---|
| `scope_request.json` | `POST _scope`, sent with a web turn's key and the scope key before the turn is queued, a turn with no browser included |
| `warm_request.json` | `POST _warm` and the person's time zone and language |
| `status_answer.json` | `_status` and `_warm` answers |
| `embed_answer.json` | `_embed`, a single-use viewer link |
| `approve.json` | `POST sessions/{id}/approve`, the person's approval of a held step, and its answer |
| `decline.json` | `POST sessions/{id}/decline`, the person's Skip of a held step, and its answer |
| `logins_answer.json` | `DELETE logins`, signing the person out of every site |
| `error_answer.json` | The error body, and the status each one reaches the app with |
