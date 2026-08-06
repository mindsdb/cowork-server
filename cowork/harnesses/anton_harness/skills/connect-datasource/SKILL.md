---
name: connect-datasource
description: 'MANDATORY reading before connecting a data source, database, or service through the guided (interactive) flow — when the user wants to add/connect Postgres, MySQL, Snowflake, BigQuery, Gmail, HubSpot, Salesforce, Shopify, or any other service and has NOT pasted credentials directly into chat. Loads the connection tools (lookup_connector, request_credentials, label_connection) and the full Data Vault workflow: registry lookup, rendering the credential form, and labelling saved connections. Recall it when the user asks to connect/add a data source. (If the user pastes credentials directly, use connect_new_datasource instead — it is always available.)'
metadata:
  display_name: Connect a data source (guided flow)
  provenance: host
---
DATA VAULT WORKFLOW — when the user asks to connect a service or database and has not pasted credentials directly:

1. LOOKUP FIRST. Call `lookup_connector` with the user's wording (e.g. "google mail", "my postgres database"). The returned `form` blob is the SAME spec the in-app Connector Picker uses — the registry already encodes OAuth flows, `methods[]`, `how_to` markdown, help URLs, and the method-picker UI for services like Gmail. Handcrafting a form from memory produces a strictly worse result, so always look up first.

2. RENDER THE FORM. Pass the looked-up `form` spec to `request_credentials` VERBATIM — tweak only `selected_method` or `subtitle`, and copy `_connector_id` onto the spec. Never strip or paraphrase `methods[]`, OAuth blocks, `how_to`, or `help_url`. Include the markdown block the tool returns VERBATIM in your next message (blank lines around the fence) so the form renders in the side panel.
   - Handcraft a spec ONLY when the registry returns no match, using your own knowledge of the service's auth shape (host/port/user/password, API key, or OAuth). For engines with several auth options emit `methods[]` instead of `fields[]`, mark the simplest `recommended: true`, and pre-set `selected_method` if the user already signalled a preference.

3. STOP THERE. The server tests the connection and saves credentials on submit. Your job is done — do NOT call `request_credentials` again unless the user asks to connect a different service or explicitly requests a new form.

4. LABEL WHEN ASKED. Once a connection is saved and the user clarifies which account is which (e.g. two Gmail accounts, and `support@…` is the support address), call `label_connection(engine=…, name=<slug>, label=…)`. The label shows beside the connection in Connected Data Sources so the right account can be picked later. Never guess a label — ask the user first, then persist it.

STRICT: field VALUES never appear in chat — don't echo them, don't include them in any form spec, don't paraphrase them. The chat-emitted form must FEEL identical to the in-app Connector Picker; the registry lookup is what guarantees that parity.
