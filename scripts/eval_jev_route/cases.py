"""Hand-written routing cases for the Jev route eval.

Each history is in the shape the gate reads from persisted rows: plain string
content for ordinary messages, and tool_use / tool_result blocks for agent
work, so `prior_turn_ran_tools` and `_text_history` see what they would in
production. The last message is the turn being routed.

`expect` is the correct route for the planned direct path: one call to the
user's own model, with the conversation's visible text, the deployment's
identity facts, a `web_search` tool, and a `delegate` tool to hand the turn
back to Anton. "direct" is anything that call can answer well; "agent" is
anything needing the user's data, files, code or multi-step work; "either"
cases are defensible both ways, reported but never counted as errors. Since
the direct model can still delegate, the labels lean direct.
"""
from __future__ import annotations

from itertools import count

_ids = count(1)


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def tool_work(steps: list[tuple[str, str]], final: str) -> list[dict]:
    """An agent turn: each (tool name, output) step, then the visible answer."""
    rows: list[dict] = []
    for name, output in steps:
        tool_id = f"toolu_{next(_ids):04d}"
        rows.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
        ]})
        rows.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool_id, "content": output},
        ]})
    rows.append(assistant(final))
    return rows


_SALES_OUTPUT = (
    "total_revenue=4,812,330 yoy_growth=0.142 gross_margin=0.387 "
    "top_region=EMEA (1,904,220) new_customers=1,214 churn=0.031"
)

# A dashboard built with tools whose final reply STATES the headline numbers.
DASHBOARD_VISIBLE = [
    user("Build me a dashboard of our 2025 sales from the warehouse connection."),
    *tool_work(
        [("query_datasource", _SALES_OUTPUT), ("write_dashboard", "dashboard.html written")],
        "Your 2025 sales dashboard is ready. Headline numbers: total revenue "
        "$4.81M (up 14.2% year over year), gross margin 38.7%, 1,214 new "
        "customers, and churn of 3.1%. EMEA was the top region at $1.90M.",
    ),
]

# The same work, but the reply never states the numbers: they exist only in
# tool output, which the router and the direct answer never see.
DASHBOARD_HIDDEN = [
    user("Build me a dashboard of our 2025 sales from the warehouse connection."),
    *tool_work(
        [("query_datasource", _SALES_OUTPUT), ("write_dashboard", "dashboard.html written")],
        "Your 2025 sales dashboard is ready. It has revenue, margin, customer "
        "and regional breakdowns. Open it from the artifacts panel.",
    ),
]

# A plain exchange with numbers, no tools anywhere.
PLAIN_NUMBERS = [
    user("If I have 3 plans at $12, $29 and $99 a month, how would you describe the tiers?"),
    assistant(
        "A typical framing: $12 is the entry tier for individuals, $29 is the "
        "team tier and your likely volume driver, and $99 is the premium tier "
        "for power users or small businesses."
    ),
]

_CHURN_ANALYSIS = [
    user("Why did churn spike in March? Use the subscriptions table."),
    *tool_work(
        [("query_datasource", "march_churn=0.058 feb_churn=0.029 annual_plan_share=0.22 "
                              "price_increase_cohort_churn=0.121")],
        "Churn rose from 2.9% in February to 5.8% in March. Almost all of it came "
        "from customers hit by the March 1 price increase, whose churn was 12.1%. "
        "Customers on annual plans (22% of the base) were unaffected.",
    ),
]


CASES: list[dict] = [
    # ── small talk and general knowledge: direct ──
    {"id": "hello", "category": "small_talk", "expect": "direct",
     "history": [user("hey! how's it going?")]},
    {"id": "thanks_after_work", "category": "small_talk", "expect": "direct",
     "history": [*DASHBOARD_VISIBLE, user("perfect, thank you!")]},
    {"id": "define_ebitda", "category": "general", "expect": "direct",
     "history": [user("What does EBITDA stand for, in one line?")]},
    {"id": "explain_pvalue", "category": "general", "expect": "direct",
     "history": [user("Explain a p-value to a non-technical manager in two sentences.")]},
    {"id": "tier_names", "category": "general", "expect": "direct",
     "history": [*PLAIN_NUMBERS, user("Suggest catchy names for those three tiers.")]},

    # ── recall of numbers the assistant already stated: direct (the target case) ──
    {"id": "headline_again", "category": "recall_visible", "expect": "direct",
     "history": [*DASHBOARD_VISIBLE, user("what were those headline numbers again?")]},
    {"id": "top_region_again", "category": "recall_visible", "expect": "direct",
     "history": [*DASHBOARD_VISIBLE, user("which region came out on top?")]},
    {"id": "churn_recall", "category": "recall_visible", "expect": "direct",
     "history": [*_CHURN_ANALYSIS, user("remind me, what was the churn for the price increase cohort?")]},
    {"id": "summarize_findings", "category": "recall_visible", "expect": "direct",
     "history": [*_CHURN_ANALYSIS, user("Summarize that in one sentence I can paste into Slack.")]},
    {"id": "rephrase_for_exec", "category": "recall_visible", "expect": "direct",
     "history": [*DASHBOARD_VISIBLE, user("Rewrite those results as three bullet points for my CEO.")]},
    {"id": "recall_after_chat", "category": "recall_visible", "expect": "direct",
     "history": [*DASHBOARD_VISIBLE, user("nice"), assistant("Glad it helps! Anything else you want on it?"),
                 user("what was the gross margin again?")]},
    {"id": "recall_spanish", "category": "recall_visible", "expect": "direct",
     "history": [*DASHBOARD_VISIBLE, user("¿Cuáles eran los números principales otra vez?")]},

    # ── arithmetic over visible numbers: defensible either way ──
    {"id": "derive_revenue_2024", "category": "derive_visible", "expect": "either",
     "history": [*DASHBOARD_VISIBLE, user("so roughly what was 2024 revenue then?")]},
    {"id": "derive_churn_ratio", "category": "derive_visible", "expect": "either",
     "history": [*_CHURN_ANALYSIS, user("how many times higher was March churn than February?")]},

    # ── recall where the numbers are only in tool output: agent ──
    {"id": "hidden_headline", "category": "recall_hidden", "expect": "agent",
     "history": [*DASHBOARD_HIDDEN, user("what were the headline numbers?")]},
    {"id": "hidden_top_region", "category": "recall_hidden", "expect": "agent",
     "history": [*DASHBOARD_HIDDEN, user("which region came out on top?")]},
    {"id": "hidden_new_customers", "category": "recall_hidden", "expect": "agent",
     "history": [*DASHBOARD_VISIBLE, user("how many of the new customers were in EMEA?")]},
    {"id": "hidden_spanish", "category": "recall_hidden", "expect": "agent",
     "history": [*DASHBOARD_HIDDEN, user("¿Cuál fue el margen bruto?")]},

    # ── new work on top of earlier results: agent ──
    {"id": "add_chart", "category": "new_work", "expect": "agent",
     "history": [*DASHBOARD_VISIBLE, user("add a monthly revenue trend chart to it")]},
    {"id": "exclude_month", "category": "new_work", "expect": "agent",
     "history": [*_CHURN_ANALYSIS, user("rerun that excluding the annual plan customers")]},
    {"id": "refresh_data", "category": "new_work", "expect": "agent",
     "history": [*DASHBOARD_VISIBLE, user("are those numbers still current? refresh them")]},
    {"id": "export_csv", "category": "new_work", "expect": "agent",
     "history": [*DASHBOARD_VISIBLE, user("export the regional breakdown as a CSV")]},
    {"id": "fresh_query", "category": "new_work", "expect": "agent",
     "history": [user("How many active users did we have last week?")]},
    {"id": "new_work_french", "category": "new_work", "expect": "agent",
     "history": [*DASHBOARD_VISIBLE, user("Ajoute une ventilation par produit au tableau de bord.")]},

    # ── identity and product questions: direct (deployment facts + web search) ──
    {"id": "which_model", "category": "identity_product", "expect": "direct",
     "history": [user("which model are you running on right now?")]},
    {"id": "who_made_you", "category": "identity_product", "expect": "direct",
     "history": [user("who made you?")]},
    {"id": "update_app", "category": "identity_product", "expect": "direct",
     "history": [user("how do I update the desktop app?")]},
    {"id": "pricing", "category": "identity_product", "expect": "direct",
     "history": [user("how much does this cost per month?")]},
    # About this user's own setup, which neither web search nor deployment facts know.
    {"id": "my_connections", "category": "identity_product", "expect": "agent",
     "history": [user("which data sources do I have connected right now?")]},

    # ── time-sensitive facts: direct (one web search) ──
    {"id": "stock_price", "category": "time_sensitive", "expect": "direct",
     "history": [user("what's NVIDIA's stock price right now?")]},
    {"id": "latest_news", "category": "time_sensitive", "expect": "direct",
     "history": [user("what's the latest news on EU AI regulation?")]},
    {"id": "weather", "category": "time_sensitive", "expect": "direct",
     "history": [*DASHBOARD_VISIBLE, user("unrelated, but what's the weather in Paris tomorrow?")]},
    # Many searches plus a written comparison: more than one call should do.
    {"id": "research_report", "category": "time_sensitive", "expect": "agent",
     "history": [user("Research our top 5 competitors' current pricing pages and build a comparison table.")]},

    # ── names a file: agent (a code check catches these before any model) ──
    {"id": "file_mention", "category": "file", "expect": "agent",
     "history": [user("what's in q3_forecast.xlsx?")]},
]
