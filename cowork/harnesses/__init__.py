from . import anton_harness as _  # noqa: F401

try:
    from . import hermes_harness as __  # noqa: F401
except ImportError:
    # Does not cover hermes-agent being absent: both of its imports are lazy
    # (hermes_harness/harness.py), so the harness registers with or without it.
    pass
