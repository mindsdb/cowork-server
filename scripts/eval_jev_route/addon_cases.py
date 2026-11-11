"""Hand-written cases for Jev's `addon_route` question (ENG-4194).

Each case is a message the person sent while a turn was running: the running
task's request, its recent steps as `[done] ...` / `[in progress] ...` lines (the state
`jev_route.addon_state` builds), the messages already added to it, and the new
message. `expect` is "add" (fold it into the running task) or "queue" (run it
next); "either" cases are reported but never counted as errors. Queueing is
today's behaviour, so a wrong "queue" costs a restart while a wrong "add" sends
an unrelated request into the wrong task: `false_add` is the headline number.
"""
from __future__ import annotations

_SALES = "Build me a dashboard of our 2025 sales from the warehouse connection."
_SALES_STEPS = [
    "[done] List the warehouse tables (list_datasources)",
    "[in progress] Query 2025 sales by month and region (query_datasource)",
]
_REPORT = "Write a two-page summary of last quarter's churn for the board."
_REPORT_STEPS = [
    "[done] Pull last quarter's cancellations and active accounts (query_datasource)",
    "[in progress] Compute churn by segment and draft the summary (scratchpad)",
]
_RESEARCH = "Research our top 5 competitors' pricing pages and build a comparison table."
_RESEARCH_STEPS = [
    "[done] Find the top 5 competitors (web_search)",
    "[done] Read the first competitor's pricing page (web_fetch)",
    "[in progress] Read the second competitor's pricing page (web_fetch)",
]


def case(id_: str, category: str, expect: str, message: str, *, task: str = _SALES,
         progress: list[str] | None = None, added: list[str] | None = None) -> dict:
    return {
        "id": id_, "category": category, "expect": expect,
        "task": task, "progress": _SALES_STEPS if progress is None else progress,
        "added": added or [], "message": message,
    }


ADDON_CASES: list[dict] = [
    # ── corrections: add ──
    case("other_source", "correction", "add", "actually use the sales DB, not marketing"),
    case("wrong_year", "correction", "add", "wait, I meant 2024 not 2025"),
    case("wrong_metric", "correction", "add", "use net revenue, gross is misleading",
         task=_REPORT, progress=_REPORT_STEPS),
    case("typo_fix", "correction", "add", "*warehouse_prod, the other one is stale"),

    # ── narrowing or extra detail: add ──
    case("also_include", "refine", "add", "also include Q3 by region"),
    case("skip_part", "refine", "add", "skip the chart, just the table"),
    case("shorter", "refine", "add", "make it one page, not two", task=_REPORT, progress=_REPORT_STEPS),
    case("only_some", "refine", "add", "only the EU competitors", task=_RESEARCH, progress=_RESEARCH_STEPS),
    case("format", "refine", "add", "put the prices in EUR", task=_RESEARCH, progress=_RESEARCH_STEPS),
    case("follow_up_addon", "refine", "add", "and the same for EMEA",
         added=["break revenue down by product line"]),
    case("answer_for_task", "refine", "add", "the fiscal year starts in April"),
    case("non_english", "refine", "add", "agrega también el margen bruto"),

    # ── new or dependent requests: queue ──
    case("email_result", "next_task", "queue", "now email this to the team"),
    case("share_after", "next_task", "queue", "when it's done, publish it and send me the link"),
    case("unrelated", "next_task", "queue", "what's the weather in Paris tomorrow?"),
    case("new_analysis", "next_task", "queue", "after this, build a churn model for the same customers"),
    case("other_project", "next_task", "queue", "separately, can you clean up my Notion connection?",
         task=_REPORT, progress=_REPORT_STEPS),
    case("thanks", "next_task", "queue", "thanks!"),

    # ── defensible both ways ──
    case("also_slides", "ambiguous", "either", "also turn it into slides", task=_REPORT, progress=_REPORT_STEPS),
    case("how_long", "ambiguous", "either", "how long will this take?"),
]
