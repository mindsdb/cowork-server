"""A doc's Markdown that uses every construct the docs format reads, for the docs tests."""

PLAN = """# Q3 launch plan

Q3 ships the connector to **all** workspaces and takes weekly active teams to 500.

## Workstreams

| Workstream | Owner | Status | Due | Budget ($k) |
| --- | --- | --- | --- | ---: |
| Connector | Platform | In progress | 2026-11-01 | 120 |
| Launch comms | Marketing | Not started | 2026-11-15 | 45 |
| Pricing | Finance | Done | 2026-10-10 | 10 |

```chart
{"type": "bar", "title": "Budget by workstream", "unit": "$k", "source": {"table": "previous", "x": "Workstream", "y": ["Budget ($k)"]}}
```

> [!WARNING] Legal review
> GA slips if legal review runs long.

- [ ] Pick the first workspaces
- [x] Draft pricing
    - nested note

1. First
2. Second

```metrics
[{"label": "WAT", "value": "312", "delta": "+18%", "tone": "good"}]
```

{{pending: Risks — checking the launch thread}}
"""
