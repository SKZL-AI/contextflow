"""
Abstract base class for all LLM providers.

Defines the interface that all provider implementations must follow,
ensuring consistent behavior across different LLM APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from contextflow.core.types import (
    CompletionResponse,
    Message,
    ProviderCapabilities,
    ProviderType,
    StreamChunk,
)


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.

    All provider implementations (Claude, OpenAI, Ollama, etc.) must inherit
    from this class and implement the abstract methods.

    Example implementation:
        class MyProvider(BaseProvider):
            @property
            def name(self) -> str:
                return "my_provider"

            async def complete(self, messages, **kwargs) -> CompletionResponse:
                # Implementation here
                pass
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
    ):
        """
        Initialize the provider.

        Args:
            model: Model identifier (e.g., "claude-3-sonnet-20240229")
            api_key: API key for authentication
            base_url: Optional custom base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_backoff: Exponential backoff multiplier for retries
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Provider identifier.

        Returns:
            Unique provider name (e.g., "claude", "openai", "ollama")
        """
        pass

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """
        Provider category.

        Returns:
            ProviderType enum value
        """
        pass

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """
        Provider capabilities and limits.

        Returns:
            ProviderCapabilities with context limits, supported features, etc.
        """
        pass

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop_sequences: list[str] | None = None,
        **kwargs: object,
    ) -> CompletionResponse:
        """
        Execute a completion request.

        Args:
            messages: List of conversation messages
            system: Optional system prompt
            model: Optional model override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 - 1.0)
            top_p: Nucleus sampling parameter
            stop_sequences: Sequences that stop generation
            **kwargs: Provider-specific parameters

        Returns:
            CompletionResponse with generated text and metadata

        Raises:
            ProviderError: On API errors
            ProviderAuthenticationError: On auth failures
            ProviderRateLimitError: On rate limit exceeded
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        **kwargs: object,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a completion response.

        Args:
            messages: List of conversation messages
            system: Optional system prompt
            model: Optional model override
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Provider-specific parameters

        Yields:
            StreamChunk objects as they arrive

        Raises:
            ProviderError: On API errors
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str, model: str | None = None) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for
            model: Optional model for model-specific tokenization

        Returns:
            Number of tokens
        """
        pass

    async def validate_credentials(self) -> bool:
        """
        Validate API credentials.

        Returns:
            True if credentials are valid

        Raises:
            ProviderAuthenticationError: If validation fails
        """
        # Default implementation tries a simple completion
        try:
            await self.complete(
                messages=[Message(role="user", content="Hi")],
                max_tokens=5,
            )
            return True
        except Exception:
            return False

    def _convert_messages(
        self,
        messages: list[Message],
    ) -> list[dict[str, str]]:
        """
        Convert Message objects to provider-specific format.

        Default implementation returns simple role/content dicts.
        Override in subclasses for provider-specific formats.
        """
        return [msg.to_dict() for msg in messages]

    def get_model_info(self) -> dict[str, object]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model details
        """
        return {
            "provider": self.name,
            "model": self.model,
            "capabilities": {
                "max_context_tokens": self.capabilities.max_context_tokens,
                "max_output_tokens": self.capabilities.max_output_tokens,
                "supports_streaming": self.capabilities.supports_streaming,
                "supports_system_prompt": self.capabilities.supports_system_prompt,
            },
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
