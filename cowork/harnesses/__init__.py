from . import anton_harness as _  # noqa: F401

try:
    from . import hermes_harness as __  # noqa: F401
except ImportError:
    # Does not cover hermes-agent being absent: every hermes import under
    # hermes_harness is lazy (harness.py, tools.py), so it registers anyway.
    pass
