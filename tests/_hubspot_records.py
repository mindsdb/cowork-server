"""HubSpot records as its search_crm_objects tool returns them, for the
figure-source tests of the matcher and the formatter."""

import json

# Two open deals as HubSpot's search_crm_objects returns them.
HUBSPOT_DEALS = json.dumps({"total": 2, "results": [
    {"id": "18234567890", "properties": {"dealname": "Acme renewal", "amount": "48250.00", "dealstage": "contractsent"}},
    {"id": "18234567891", "properties": {"dealname": "Globex expansion", "amount": "12900.00", "dealstage": "qualified"}},
]})
SEARCH_DEALS = {"objectType": "deals", "limit": 10}
