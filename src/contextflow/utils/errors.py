"""
Custom exceptions for ContextFlow.

Provides a hierarchy of exceptions for different error types,
enabling precise error handling throughout the framework.
"""

from __future__ import annotations

from typing import Any


class ContextFlowError(Exception):
    """
    Base exception for all ContextFlow errors.

    All custom exceptions inherit from this class.
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Provider Errors
# =============================================================================


class ProviderError(ContextFlowError):
    """
    LLM provider errors.

    Raised when there's an issue with the LLM provider (API errors,
    rate limits, authentication failures, etc.).
    """

    def __init__(
        self,
        provider: str,
        message: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(f"[{provider}] {message}", details, cause)
        self.provider = provider
        self.status_code = status_code


class ProviderAuthenticationError(ProviderError):
    """Authentication failed for provider."""

    def __init__(self, provider: str, details: dict[str, Any] | None = None):
        super().__init__(
            provider,
            "Authentication failed. Check your API key.",
            status_code=401,
            details=details,
        )


class ProviderRateLimitError(ProviderError):
    """Rate limit exceeded for provider."""

    def __init__(
        self,
        provider: str,
        retry_after: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        message = "Rate limit exceeded."
        if retry_after:
            message += f" Retry after {retry_after} seconds."
        super().__init__(provider, message, status_code=429, details=details)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """Request timeout for provider."""

    def __init__(
        self,
        provider: str,
        timeout: int,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            provider,
            f"Request timed out after {timeout} seconds.",
            status_code=408,
            details=details,
        )
        self.timeout = timeout


class ProviderUnavailableError(ProviderError):
    """Provider service is unavailable."""

    def __init__(self, provider: str, details: dict[str, Any] | None = None):
        super().__init__(
            provider,
            "Service temporarily unavailable.",
            status_code=503,
            details=details,
        )


# =============================================================================
# Context Errors
# =============================================================================


class ContextOverflowError(ContextFlowError):
    """
    Context window exceeded.

    Raised when the input context exceeds the model's maximum context window.
    """

    def __init__(
        self,
        token_count: int,
        max_tokens: int,
        model: str,
        details: dict[str, Any] | None = None,
    ):
        message = (
            f"Context overflow: {token_count:,} tokens exceeds "
            f"maximum {max_tokens:,} tokens for model {model}"
        )
        super().__init__(message, details)
        self.token_count = token_count
        self.max_tokens = max_tokens
        self.model = model


class ContextCompressionError(ContextFlowError):
    """Failed to compress context."""

    def __init__(
        self,
        message: str = "Failed to compress context to fit within limits.",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)


# =============================================================================
# Strategy Errors
# =============================================================================


class StrategySelectionError(ContextFlowError):
    """
    Cannot select appropriate strategy.

    Raised when the strategy router cannot determine an appropriate
    strategy for the given context.
    """

    def __init__(
        self,
        message: str = "Could not determine appropriate strategy.",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)


class StrategyExecutionError(ContextFlowError):
    """Strategy execution failed."""

    def __init__(
        self,
        strategy: str,
        message: str,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(f"[{strategy}] {message}", details, cause)
        self.strategy = strategy


# =============================================================================
# RLM Errors
# =============================================================================


class RLMError(ContextFlowError):
    """
    RLM-specific errors.

    Base class for errors specific to Recursive Language Model execution.
    """

    pass


class RLMMaxIterationsError(RLMError):
    """Maximum iterations reached without convergence."""

    def __init__(
        self,
        iterations: int,
        max_iterations: int,
        details: dict[str, Any] | None = None,
    ):
        message = (
            f"RLM reached maximum iterations ({iterations}/{max_iterations}) "
            "without producing FINAL answer."
        )
        super().__init__(message, details)
        self.iterations = iterations
        self.max_iterations = max_iterations


class RLMDepthLimitError(RLMError):
    """Recursion depth limit exceeded."""

    def __init__(
        self,
        depth: int,
        max_depth: int,
        details: dict[str, Any] | None = None,
    ):
        message = f"RLM recursion depth limit exceeded ({depth}/{max_depth})."
        super().__init__(message, details)
        self.depth = depth
        self.max_depth = max_depth


class RLMCodeExecutionError(RLMError):
    """Error during REPL code execution."""

    def __init__(
        self,
        code: str,
        error: str,
        details: dict[str, Any] | None = None,
    ):
        message = f"REPL code execution failed: {error}"
        super().__init__(message, {**(details or {}), "code": code[:500]})
        self.code = code
        self.error = error


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(ContextFlowError):
    """
    Invalid configuration.

    Raised when configuration is invalid or missing required values.
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.config_key = config_key


class MissingAPIKeyError(ConfigurationError):
    """Required API key is missing."""

    def __init__(self, provider: str):
        super().__init__(
            f"API key not configured for provider: {provider}",
            config_key=f"{provider.upper()}_API_KEY",
        )
        self.provider = provider


# =============================================================================
# Token Errors
# =============================================================================


class TokenCountingError(ContextFlowError):
    """
    Token counting failed.

    Raised when token counting encounters an error.
    """

    def __init__(
        self,
        message: str = "Failed to count tokens.",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(ContextFlowError):
    """
    Input validation error.

    Raised when input validation fails.
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.field = field


class EmptyInputError(ValidationError):
    """Input is empty."""

    def __init__(self, field: str = "input"):
        super().__init__(f"{field} cannot be empty.", field=field)


class InvalidDocumentError(ValidationError):
    """Document is invalid or cannot be loaded."""

    def __init__(
        self,
        path: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(f"Invalid document '{path}': {reason}", details=details)
        self.path = path
        self.reason = reason


# =============================================================================
# RAG Errors
# =============================================================================


class RAGError(ContextFlowError):
    """RAG-related errors."""

    pass


class EmbeddingError(RAGError):
    """Embedding generation failed."""

    def __init__(
        self,
        provider: str,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(f"[{provider}] Embedding failed: {message}", details)
        self.provider = provider


class ChunkingError(RAGError):
    """Document chunking failed."""

    def __init__(
        self,
        message: str = "Failed to chunk document.",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)


# =============================================================================
# Agent Errors
# =============================================================================


class AgentError(ContextFlowError):
    """Sub-agent related errors."""

    pass


class AgentPoolExhaustedError(AgentError):
    """Agent pool has no available capacity."""

    def __init__(
        self,
        current_agents: int,
        max_agents: int,
        details: dict[str, Any] | None = None,
    ):
        message = f"Agent pool exhausted: {current_agents}/{max_agents} agents in use."
        super().__init__(message, details)
        self.current_agents = current_agents
        self.max_agents = max_agents


class AgentTimeoutError(AgentError):
    """Agent execution timed out."""

    def __init__(
        self,
        agent_id: str,
        timeout: int,
        details: dict[str, Any] | None = None,
    ):
        message = f"Agent {agent_id} timed out after {timeout} seconds."
        super().__init__(message, details)
        self.agent_id = agent_id
        self.timeout = timeout


# =============================================================================
# Pool Errors
# =============================================================================


class PoolError(ContextFlowError):
    """Agent pool related errors."""

    pass


class PoolExhaustedError(PoolError):
    """Pool has no available capacity."""

    def __init__(
        self,
        active_agents: int,
        max_agents: int,
        details: dict[str, Any] | None = None,
    ):
        message = (
            f"Pool exhausted: {active_agents}/{max_agents} agents active, " "no capacity available."
        )
        super().__init__(message, details)
        self.active_agents = active_agents
        self.max_agents = max_agents


class PoolShutdownError(PoolError):
    """Pool is shutting down or already stopped."""

    def __init__(
        self,
        status: str,
        details: dict[str, Any] | None = None,
    ):
        message = f"Pool is not available (status: {status})."
        super().__init__(message, details)
        self.status = status


# =============================================================================
# Aliases for backward compatibility
# =============================================================================

# Aliases for common error types
RateLimitError = ProviderRateLimitError
TokenLimitError = ContextOverflowError
