# Jev route eval: 2026-11-03

Rerun of `jev_proposed` after the PR 761 review changes: the routing question
gained ENG-3896's rule (change, send, confirm or approve requests, and replies
to an earlier question, need the agent), and the same `/decisions` call now also
asks `search` (`jev_route.SEARCH_QUESTION`), which sets how long the direct
answer may take to start. Raw results: `2026-11-03-search-question.json`.

```sh
uv run python -m scripts.eval_jev_route.run --env-file <file> --only jev_proposed --repeat 3 --out scripts/eval_jev_route/results/2026-11-03-search-question.json
```

| Strategy | Correct | False direct | Missed direct | Latency |
|---|---|---|---|---|
| `jev_proposed`, both questions | **96/96** | **0** | **0** | p50 766ms, p95 1.04s |

Routing is unchanged from the 2026-10-22 baseline at every threshold (0.95
moves from 88 to 87 of 96). Adding the second question did not slow the call.

Search answers on the cases that go direct, identical on all 3 repeats:

- `search_web`: `weather`, `stock_price`, `latest_news`, `pricing`. These keep
  the 30s first-text wait.
- `no_search`: every recall, small-talk, general-knowledge and identity case.
  These get 10s before a wrong direct route reaches the agent.
