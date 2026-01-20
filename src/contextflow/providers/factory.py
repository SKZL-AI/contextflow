"""
Provider factory for creating LLM provider instances.

Supports registration of custom providers and automatic
configuration from environment variables.
"""

from __future__ import annotations

from contextflow.core.config import ContextFlowConfig, get_config
from contextflow.providers.base import BaseProvider
from contextflow.utils.errors import ConfigurationError, MissingAPIKeyError

# Provider registry
_PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {}


def _register_builtin_providers() -> None:
    """Register built-in providers."""
    # Import here to avoid circular imports
    from contextflow.providers.claude import ClaudeProvider
    from contextflow.providers.gemini import GeminiProvider
    from contextflow.providers.groq import GroqProvider
    from contextflow.providers.mistral import MistralProvider
    from contextflow.providers.ollama import OllamaProvider
    from contextflow.providers.openai_provider import OpenAIProvider
    from contextflow.providers.vllm import VLLMProvider

    _PROVIDER_REGISTRY["claude"] = ClaudeProvider
    _PROVIDER_REGISTRY["openai"] = OpenAIProvider
    _PROVIDER_REGISTRY["mistral"] = MistralProvider
    _PROVIDER_REGISTRY["ollama"] = OllamaProvider
    _PROVIDER_REGISTRY["vllm"] = VLLMProvider
    _PROVIDER_REGISTRY["gemini"] = GeminiProvider
    _PROVIDER_REGISTRY["groq"] = GroqProvider


def register_provider(name: str, provider_class: type[BaseProvider]) -> None:
    """
    Register a custom provider.

    Args:
        name: Provider identifier
        provider_class: Provider class (must inherit from BaseProvider)

    Example:
        from contextflow.providers import register_provider, BaseProvider

        class MyProvider(BaseProvider):
            # Implementation
            pass

        register_provider("my_provider", MyProvider)
    """
    if not issubclass(provider_class, BaseProvider):
        raise TypeError(f"{provider_class} must inherit from BaseProvider")
    _PROVIDER_REGISTRY[name] = provider_class


def get_provider(
    name: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    config: ContextFlowConfig | None = None,
    **kwargs: object,
) -> BaseProvider:
    """
    Get a provider instance.

    Args:
        name: Provider name (e.g., "claude", "openai")
        model: Model to use (optional, uses config default)
        api_key: API key (optional, uses config/env)
        config: ContextFlowConfig (optional, uses global config)
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured provider instance

    Raises:
        ConfigurationError: If provider is unknown
        MissingAPIKeyError: If API key is required but not found

    Example:
        # Using defaults from config
        provider = get_provider("claude")

        # With explicit model
        provider = get_provider("openai", model="gpt-4o-mini")

        # With explicit API key
        provider = get_provider("claude", api_key="sk-ant-...")
    """
    # Ensure built-in providers are registered
    if not _PROVIDER_REGISTRY:
        _register_builtin_providers()

    # Use global config if not provided
    config = config or get_config()

    # Use default provider if not specified
    name = name or config.default_provider

    # Check if provider exists
    if name not in _PROVIDER_REGISTRY:
        available = ", ".join(_PROVIDER_REGISTRY.keys())
        raise ConfigurationError(
            f"Unknown provider: {name}. Available: {available}",
            config_key="provider",
        )

    # Get provider-specific config
    provider_config = config.get_provider_config(name)

    # Determine model
    final_model = model or provider_config.model

    # Determine API key
    final_api_key = api_key or provider_config.api_key

    # Check if API key is required (not for local providers)
    if name not in ("ollama", "vllm") and not final_api_key:
        raise MissingAPIKeyError(name)

    # Create provider instance
    provider_class = _PROVIDER_REGISTRY[name]
    return provider_class(
        model=final_model,
        api_key=final_api_key,
        base_url=provider_config.base_url,
        timeout=provider_config.timeout,
        max_retries=provider_config.max_retries,
        **kwargs,
    )


def list_providers() -> list[str]:
    """
    List all registered providers.

    Returns:
        List of provider names
    """
    if not _PROVIDER_REGISTRY:
        _register_builtin_providers()
    return list(_PROVIDER_REGISTRY.keys())


def is_provider_available(name: str) -> bool:
    """
    Check if a provider is registered.

    Args:
        name: Provider name

    Returns:
        True if provider is available
    """
    if not _PROVIDER_REGISTRY:
        _register_builtin_providers()
    return name in _PROVIDER_REGISTRY


def get_available_providers() -> list[str]:
    """
    Get list of available (registered) providers.

    Returns:
        List of available provider names

    Note:
        This is an alias for list_providers() for API compatibility.
    """
    return list_providers()


class ProviderFactory:
    """
    Factory class for creating providers.

    Alternative interface to get_provider() function.

    Example:
        factory = ProviderFactory()
        provider = factory.create("claude")
    """

    def __init__(self, config: ContextFlowConfig | None = None):
        """
        Initialize factory.

        Args:
            config: Optional configuration (uses global if not provided)
        """
        self.config = config or get_config()
        if not _PROVIDER_REGISTRY:
            _register_builtin_providers()

    def create(
        self,
        name: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: object,
    ) -> BaseProvider:
        """
        Create a provider instance.

        Args:
            name: Provider name
            model: Model to use
            api_key: API key
            **kwargs: Additional arguments

        Returns:
            Provider instance
        """
        return get_provider(
            name=name,
            model=model,
            api_key=api_key,
            config=self.config,
            **kwargs,
        )

    @staticmethod
    def register(name: str, provider_class: type[BaseProvider]) -> None:
        """Register a custom provider."""
        register_provider(name, provider_class)

    @staticmethod
    def list() -> list[str]:
        """List available providers."""
        return list_providers()

    @staticmethod
    def is_available(name: str) -> bool:
        """Check if provider is available."""
        return is_provider_available(name)
