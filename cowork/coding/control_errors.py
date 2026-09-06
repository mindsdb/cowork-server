class StateConflict(ValueError):
    """A well-formed request that the target's current state does not allow."""


class ModelDiscoveryAuthenticationError(RuntimeError):
    """MindsHub rejected the credential used to discover coding models."""

    code = "coding_model_authentication_failed"


class RuntimeAuthenticationError(RuntimeError):
    """A runtime or delegated capability failed authentication."""


class StaleRuntimeEvent(RuntimeError):
    """A runtime event no longer belongs to the active fenced execution."""
