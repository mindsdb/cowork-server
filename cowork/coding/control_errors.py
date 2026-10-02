class StateConflict(ValueError):
    """A well-formed request that the target's current state does not allow."""


class ModelDiscoveryAuthenticationError(RuntimeError):
    """MindsHub rejected the credential used to discover coding models."""

    code = "coding_model_authentication_failed"


class ModelDiscoveryUnavailableError(RuntimeError):
    """MindsHub answered the model list request with a body that is not JSON."""

    code = "coding_model_list_unreadable"


class RuntimeAuthenticationError(RuntimeError):
    """A runtime or delegated capability failed authentication."""


class StaleRuntimeEvent(RuntimeError):
    """A runtime event no longer belongs to the active fenced execution."""
