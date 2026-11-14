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

Each kind of case that must reach the agent has cases of the same shape beside
it that must stay direct: the same words in other senses, the same request in
Spanish, inside a long conversation, and as a follow-up. A rule that sends more
turns to the agent then shows what it costs as well as what it catches. The
set is built to hold these words, so its rates describe behavior, not
production shares.
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

# A plain conversation of nine exchanges, no tools. It is longer than the LLM
# gate's 16-message window, so the gate never sees its opening.
LONG_CHAT = [
    user("I'm putting together a pricing page for a small analytics tool. What makes a good one?"),
    assistant("Keep it to three tiers, name each for who it serves, put the recommended "
              "tier in the middle, and show the annual savings clearly."),
    user("Should the cheapest tier be free?"),
    assistant("A free tier helps when the product spreads by word of mouth. Otherwise a "
              "low-cost starter tier with a trial usually converts better."),
    user("How long should the trial be?"),
    assistant("14 days is the common default. Use 30 when setup takes real effort, such "
              "as connecting data sources."),
    user("What's a reasonable annual discount?"),
    assistant("15 to 20 percent, often framed as two months free."),
    user("Is it OK to hide enterprise pricing behind 'contact us'?"),
    assistant("Yes, for custom contracts it's standard. Show a starting price if you "
              "can: it turns away poor fits early."),
    user("How do I explain usage-based pricing simply?"),
    assistant("Say what one unit is, what it costs, and give one worked example of a "
              "typical monthly bill."),
    user("What's a good way to show a feature comparison?"),
    assistant("A table with tiers as columns, grouped feature rows, and checkmarks "
              "rather than long text."),
    user("Any common mistakes to avoid?"),
    assistant("Too many tiers, jargon in feature names, and hiding the price of the "
              "plan most people need."),
    user("Should I show prices in more than one currency?"),
    assistant("If a large share of visitors are abroad, yes: detect the region and "
              "let people switch."),
]

# The agent saved a preference with its memory tool and said so.
MEMORY_SAVED = [
    user("Please remember that I prefer teal for my charts."),
    *tool_work(
        [("memorize", "saved rule: use teal (#008080) for charts")],
        "Saved. Your charts will use teal from now on, in this and future conversations.",
    ),
]

_EBITDA_ANSWERED = [
    user("What does EBITDA stand for?"),
    assistant("Earnings before interest, taxes, depreciation and amortization."),
]

_PLANETS_ANSWERED = [
    user("What's a good trick to remember the order of the planets?"),
    assistant('Try "My Very Educated Mother Just Served Us Nachos": Mercury, Venus, '
              "Earth, Mars, Jupiter, Saturn, Uranus, Neptune."),
]

_ISO_ANSWERED = [
    user("What date format is clearest for international teams?"),
    assistant("ISO 8601, written YYYY-MM-DD. No one misreads it, and it sorts "
              "correctly as text."),
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

    # ── remember for later: agent (only the agent keeps anything past this
    #    conversation; a direct answer can only promise it, or say it can't) ──
    # With a listed memory word (a code check catches these before any model).
    {"id": "remember_chart_color", "category": "memory", "expect": "agent",
     "history": [user("Please remember this for my later chats: my preferred chart color is teal (#008080).")]},
    {"id": "keep_this_in_mind", "category": "memory", "expect": "agent",
     "history": [user("Keep this in mind: I prefer teal for charts.")]},
    {"id": "from_now_on_bullets", "category": "memory", "expect": "agent",
     "history": [user("From now on, answer me in bullet points.")]},
    # With no listed word: only the router's prompt sends these to the agent.
    {"id": "note_for_next_time", "category": "memory", "expect": "agent",
     "history": [user("Note for next time: I prefer concise replies.")]},
    {"id": "always_call_me", "category": "memory", "expect": "agent",
     "history": [user("Please always call me Sam.")]},
    {"id": "never_pie_charts", "category": "memory", "expect": "agent",
     "history": [user("I want you to know I never use pie charts.")]},
    # Spanish, with and without a listed word.
    {"id": "recuerda_color", "category": "memory", "expect": "agent",
     "history": [user("Recuerda que prefiero los gráficos en verde azulado.")]},
    {"id": "a_partir_de_ahora", "category": "memory", "expect": "agent",
     "history": [user("A partir de ahora respóndeme siempre en español.")]},
    {"id": "guarda_en_tu_memoria", "category": "memory", "expect": "agent",
     "history": [user("Guarda en tu memoria que mi jefa se llama Marta.")]},

    # ── the same shapes with nothing to keep past this reply: direct (the
    #    code check takes the ones with a listed word to the agent, at one
    #    hop each) ──
    {"id": "keep_in_mind_beginner", "category": "memory_word_general", "expect": "direct",
     "history": [user("Keep in mind I'm a beginner: what's compound interest, in two sentences?")]},
    {"id": "from_now_on_team", "category": "memory_word_general", "expect": "direct",
     "history": [user("How do I tell my team politely that from now on standups start at 9:30?")]},
    {"id": "this_answer_bullets", "category": "memory_word_general", "expect": "direct",
     "history": [user("Just for this answer, use bullet points: what are the pros and cons of index funds?")]},
    {"id": "next_time_overheats", "category": "memory_word_general", "expect": "direct",
     "history": [user("What should I do next time my laptop overheats?")]},
    {"id": "always_called_big_apple", "category": "memory_word_general", "expect": "direct",
     "history": [user("Why is New York always called the Big Apple?")]},
    {"id": "never_pie_charts_why", "category": "memory_word_general", "expect": "direct",
     "history": [user("Why do some analysts say you should never use pie charts?")]},
    # Says who they are and asks nothing to be kept.
    {"id": "nice_to_meet_sam", "category": "small_talk", "expect": "direct",
     "history": [user("Hi, I'm Sam, nice to meet you!")]},
    {"id": "eso_me_recuerda", "category": "memory_word_general", "expect": "direct",
     "history": [user("¿Qué es una muestra aleatoria?"),
                 assistant("Un subconjunto de la población elegido al azar, de modo que cada "
                           "elemento tiene la misma probabilidad de salir."),
                 user("Eso me recuerda: ¿qué es un intervalo de confianza?")]},
    {"id": "a_partir_de_ahora_equipo", "category": "memory_word_general", "expect": "direct",
     "history": [user("¿Cómo le digo a mi equipo que a partir de ahora las reuniones empiezan a las 9?")]},
    {"id": "memoria_ram", "category": "memory_word_general", "expect": "direct",
     "history": [user("¿Cuánta memoria RAM necesito para trabajar con hojas de cálculo grandes?")]},
    {"id": "memorizar_planetas", "category": "memory_word_general", "expect": "direct",
     "history": [user("¿Algún truco para memorizar el orden de los planetas?")]},

    # ── forget something saved: agent ──
    {"id": "forget_address", "category": "forget", "expect": "agent",
     "history": [user("Please forget my home address.")]},
    {"id": "wipe_employer", "category": "forget", "expect": "agent",
     "history": [user("Wipe what you know about my previous employer.")]},
    {"id": "olvida_presupuesto", "category": "forget", "expect": "agent",
     "history": [user("Olvida lo que te dije sobre el presupuesto.")]},

    # ── forget and wipe in other senses: direct ──
    {"id": "elephants_never_forget", "category": "memory_word_general", "expect": "direct",
     "history": [user("Is it true that elephants never forget?")]},
    {"id": "wipe_laptop", "category": "memory_word_general", "expect": "direct",
     "history": [user("How do I wipe my old laptop before selling it?")]},
    {"id": "elefante_nunca_olvida", "category": "memory_word_general", "expect": "direct",
     "history": [user("¿Es verdad que un elefante nunca olvida?")]},
    {"id": "olvide_kpi", "category": "memory_word_general", "expect": "direct",
     "history": [user("Se me olvidó qué significa KPI, ¿me lo recuerdas?")]},

    # ── what the assistant can keep: agent (a direct answer, with no product
    #    fact about memory, tends to say it can't) ──
    {"id": "long_term_memory", "category": "capability", "expect": "agent",
     "history": [user("Do you have long-term memory?")]},
    {"id": "store_preferences", "category": "capability", "expect": "agent",
     "history": [user("Can you store my preferences?")]},

    # ── memory as a topic, and what the direct call can answer about itself:
    #    direct ──
    {"id": "memory_psychology", "category": "capability", "expect": "direct",
     "history": [user("What's the difference between short-term and long-term memory?")]},
    {"id": "ai_memory_general", "category": "capability", "expect": "direct",
     "history": [user("How do AI assistants usually keep long-term memory between sessions?")]},
    {"id": "store_preferences_webapp", "category": "capability", "expect": "direct",
     "history": [user("What's a good way to store user preferences in a web app?")]},
    {"id": "speak_spanish", "category": "capability", "expect": "direct",
     "history": [user("Do you speak Spanish?")]},
    # The honest answer says "can't" about something the agent can't do either.
    {"id": "hug_rough_day", "category": "capability", "expect": "direct",
     "history": [user("Can you give me a hug? Rough day.")]},
    {"id": "read_my_mind", "category": "capability", "expect": "direct",
     "history": [user("Can you read my mind?")]},
    {"id": "divide_by_zero", "category": "general", "expect": "direct",
     "history": [user("Can you divide by zero?")]},

    # ── the user's own preferences: agent when said in another conversation
    #    (only the agent recalls them), direct when said in this one ──
    {"id": "my_saved_color", "category": "preference", "expect": "agent",
     "history": [user("What chart color did I tell you I prefer?")]},
    {"id": "my_color_this_chat", "category": "preference", "expect": "direct",
     "history": [user("My favorite color for charts is teal. What accent color goes well with it?"),
                 assistant("Coral or a warm orange pairs well with teal: both stand out without clashing."),
                 user("What did I say my favorite chart color was?")]},
    {"id": "colorblind_palette", "category": "general", "expect": "direct",
     "history": [user("Which chart colors work best for colorblind readers?")]},

    # ── the memory words in other senses: direct (general knowledge; the
    #    code check takes some of these to the agent, at one hop each) ──
    {"id": "remember_world_cup", "category": "memory_word_general", "expect": "direct",
     "history": [user("Do you remember who won the 1998 World Cup?")]},
    {"id": "remember_planets", "category": "memory_word_general", "expect": "direct",
     "history": [user("What's a good trick to remember the order of the planets?")]},
    {"id": "forgotten_gdpr", "category": "memory_word_general", "expect": "direct",
     "history": [user("What is the right to be forgotten under GDPR?")]},
    {"id": "forgot_ebitda", "category": "memory_word_general", "expect": "direct",
     "history": [user("I forgot, what does EBITDA stand for?")]},
    {"id": "keep_in_mind_salary", "category": "memory_word_general", "expect": "direct",
     "history": [user("What should I keep in mind when negotiating a salary?")]},
    {"id": "going_forward_rates", "category": "memory_word_general", "expect": "direct",
     "history": [user("What's the outlook for interest rates going forward?")]},
    {"id": "memorize_periodic_table", "category": "memory_word_general", "expect": "direct",
     "history": [user("How do I memorize the periodic table quickly?")]},
    {"id": "store_my_passwords", "category": "memory_word_general", "expect": "direct",
     "history": [user("Where's the safest place to store my passwords?")]},
    {"id": "flask_across_sessions", "category": "memory_word_general", "expect": "direct",
     "history": [user("How do I keep a user logged in across sessions in Flask?")]},

    # ── greetings and general questions: direct ──
    {"id": "good_morning", "category": "small_talk", "expect": "direct",
     "history": [user("Good morning!")]},
    {"id": "hola", "category": "small_talk", "expect": "direct",
     "history": [user("Hola, ¿qué tal?")]},
    {"id": "mean_vs_median", "category": "general", "expect": "direct",
     "history": [user("What's the difference between the mean and the median?")]},
    {"id": "explain_cte", "category": "general", "expect": "direct",
     "history": [user("Explain what a CTE is in SQL in two sentences.")]},
    {"id": "que_es_ebitda", "category": "general", "expect": "direct",
     "history": [user("¿Qué significa EBITDA?")]},
    {"id": "gracias_tras_trabajo", "category": "small_talk", "expect": "direct",
     "history": [*DASHBOARD_VISIBLE, user("¡Perfecto, muchas gracias!")]},
    {"id": "buenos_dias", "category": "small_talk", "expect": "direct",
     "history": [user("Buenos días, ¿cómo va todo?")]},
    {"id": "pvalor_es", "category": "general", "expect": "direct",
     "history": [user("Explícame qué es un p-valor en dos frases.")]},
    # Sounds like the user's own documents, but asks what is typical.
    {"id": "saas_refund_policy", "category": "general", "expect": "direct",
     "history": [user("What does a typical SaaS refund policy say?")]},

    # ── long conversations: the latest message decides, and memory work
    #    earlier on does not make a general question agent work ──
    {"id": "long_chat_general", "category": "long_history", "expect": "direct",
     "history": [*LONG_CHAT, user("One more: what's the difference between ARR and MRR?")]},
    {"id": "long_chat_thanks", "category": "long_history", "expect": "direct",
     "history": [*LONG_CHAT, user("This was really helpful, thanks!")]},
    {"id": "long_chat_currency", "category": "long_history", "expect": "direct",
     "history": [*LONG_CHAT, user("Which currency should I show first if most visitors are in Germany?")]},
    # With no listed word: only the router's prompt sends this to the agent.
    {"id": "long_chat_eur_always", "category": "long_history", "expect": "agent",
     "history": [*LONG_CHAT, user("By the way, I always work in EUR. Use EUR with me in every chat, please.")]},
    # The memory turn is outside the LLM gate's window; Jev and the direct answer see it.
    {"id": "long_chat_after_saved", "category": "long_history", "expect": "direct",
     "history": [*MEMORY_SAVED, *LONG_CHAT, user("One more: what's the difference between ARR and MRR?")]},

    # ── follow-ups: direct unless the follow-up itself asks to keep something ──
    {"id": "follow_up_ebit", "category": "follow_up", "expect": "direct",
     "history": [*_EBITDA_ANSWERED, user("And how is that different from EBIT?")]},
    {"id": "follow_up_mnemonic", "category": "follow_up", "expect": "direct",
     "history": [*_PLANETS_ANSWERED, user("Nice. Is there one for the order of operations too?")]},
    {"id": "follow_up_iso_why", "category": "follow_up", "expect": "direct",
     "history": [*_ISO_ANSWERED, user("Why does it sort correctly as text?")]},
    {"id": "follow_up_use_iso", "category": "follow_up", "expect": "agent",
     "history": [*_ISO_ANSWERED, user("Great, use that format with me from now on.")]},
    # Right after the agent saved a preference. The LLM gate sends all three
    # to the agent on `prior_turn_ran_tools`, so only Jev's path tells them apart.
    {"id": "after_saved_thanks", "category": "follow_up", "expect": "direct",
     "history": [*MEMORY_SAVED, user("great, thanks!")]},
    {"id": "after_saved_darker_teal", "category": "follow_up", "expect": "direct",
     "history": [*MEMORY_SAVED, user("What's the hex code for a slightly darker teal?")]},
    # With no listed word: only a model sends this to the agent.
    {"id": "after_saved_fiscal_year", "category": "follow_up", "expect": "agent",
     "history": [*MEMORY_SAVED, user("Also, my fiscal year starts in April.")]},
]
