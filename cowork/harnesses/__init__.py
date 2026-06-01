from . import anton_harness as _  # noqa: F401

try:
    from . import hermes_harness as __  # noqa: F401
except ImportError:
    pass  # hermes-agent not installed; hermes harness unavailable
