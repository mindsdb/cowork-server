"""Assert coordinates on the actual serialized diagnostic stream."""

import json


def assert_log_fields(output, **fields):
    events = [json.loads(line) for line in output.splitlines()]
    assert any(
        all(event.get(key) == value for key, value in fields.items())
        for event in events
    ), events
