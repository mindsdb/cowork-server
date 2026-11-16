# Jev route eval: 2026-11-05, through the shipped code

`jev_proposed` now runs `decide_route(jev=...)` calling `jev_route.decide`: the
production code checks (ENG-3896's write/approval check included), Jev's
history view (`response_routing.jev_history`), and the production request,
parse and 3s budget. Earlier runs called Jev directly with the eval's own
checks, so they did not measure ENG-3896. Raw results:
`2026-11-05-shipped-path.json`.

| Strategy | Correct | False direct | Missed direct | Latency |
|---|---|---|---|---|
| `jev_proposed`, shipped path | 89/96 | **0** | 7 | p50 712ms, p95 982ms |

The 7 missed direct answers:

- 6 are ENG-3896's English change-word check, before Jev is asked:
  `update_app` ("how do I **update** the desktop app?", 3 of 3) and
  `churn_recall` ("the churn for the price **increase** cohort", 3 of 3). Jev
  itself routed both direct (p 0.91 and p 0.97 to 0.98). This is the trade-off
  ENG-3896 accepted: one more hop for a harmless question using a change word.
- 1 is `pricing` at p 0.48 on one repeat, the known borderline case.

Jev alone, ignoring the code checks, is unchanged from the earlier runs.
