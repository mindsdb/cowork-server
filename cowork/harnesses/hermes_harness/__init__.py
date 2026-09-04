from importlib.util import find_spec

from cowork.harnesses.hermes_harness import memory_adapter as _  # noqa: F401

# hermes-agent is an optional extra, and every hermes import under this package
# is lazy, so importing harness would register a harness that only fails once a
# turn reaches run_agent. memory_adapter stays unconditional: it needs no hermes
# import, and the legacy memory-file cleanup resolves it.
if find_spec("run_agent") is not None:
    from cowork.harnesses.hermes_harness import harness as __  # noqa: F401
