"""The fake launch hand-off the tests hold (cowork.app_secret.LaunchHandoff).

The ``handoff_held`` fixture in tests/conftest.py holds these values, and a
test that sends one of them imports it from here. Every value is a fixed
fake hex value.
"""
from __future__ import annotations

SECRET = "ab" * 32
TOKEN = "cd" * 32
PROOF = "ef" * 32
GOOD = {"app_secret": SECRET, "auth_token": TOKEN, "window_proof": PROOF}
