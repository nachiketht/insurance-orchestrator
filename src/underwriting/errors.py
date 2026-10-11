"""Errors the pipeline can retry, and errors it must not."""


class UnderwritingError(Exception):
    """Base type for failures this package raises."""


class RetryableError(UnderwritingError):
    """A second attempt might succeed. Timeouts and connection drops land here."""


class LLMTimeoutError(RetryableError):
    """The model call exceeded its time limit."""


class LLMTransientError(RetryableError):
    """Ollama refused the connection or returned a server error."""


class AgentOutputError(UnderwritingError):
    """The model replied, but the reply cannot be used. Do not retry the same call."""


class ModelNotFoundError(UnderwritingError):
    """The named model is not installed. Retrying will not pull it."""


class LLMRequestError(UnderwritingError):
    """Ollama rejected the request. Retrying the same body will fail again."""
