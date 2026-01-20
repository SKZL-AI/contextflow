"""
Unit tests for Provider Factory and Provider implementations.

Tests provider functionality including:
- Provider factory registration
- Mock provider responses
- Token counting
- Error handling
- Provider capabilities
"""

from unittest.mock import MagicMock, patch

import pytest

from contextflow.core.config import ContextFlowConfig, ProviderConfig
from contextflow.core.types import (
    CompletionResponse,
    Message,
    ProviderCapabilities,
    ProviderType,
    StreamChunk,
)
from contextflow.providers.base import BaseProvider
from contextflow.providers.factory import (
    _PROVIDER_REGISTRY,
    ProviderFactory,
    get_available_providers,
    get_provider,
    is_provider_available,
    list_providers,
    register_provider,
)
from contextflow.utils.errors import ConfigurationError, MissingAPIKeyError

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config() -> ContextFlowConfig:
    """Create a mock configuration."""
    config = MagicMock(spec=ContextFlowConfig)
    config.default_provider = "claude"

    # Mock provider config
    provider_config = MagicMock(spec=ProviderConfig)
    provider_config.model = "claude-3-5-sonnet-20241022"
    provider_config.api_key = "test-api-key"
    provider_config.base_url = None
    provider_config.timeout = 120
    provider_config.max_retries = 3

    config.get_provider_config = MagicMock(return_value=provider_config)
    return config


@pytest.fixture
def mock_provider_class() -> type:
    """Create a mock provider class."""

    class MockProvider(BaseProvider):
        @property
        def name(self) -> str:
            return "mock"

        @property
        def provider_type(self) -> ProviderType:
            return ProviderType.PROPRIETARY

        @property
        def capabilities(self) -> ProviderCapabilities:
            return ProviderCapabilities(
                max_context_tokens=200000,
                max_output_tokens=4096,
                supports_streaming=True,
                supports_system_prompt=True,
                supports_tools=True,
                supported_models=["mock-model"],
            )

        async def complete(
            self,
            messages,
            system=None,
            model=None,
            max_tokens=4096,
            temperature=0.7,
            top_p=1.0,
            stop_sequences=None,
            **kwargs,
        ) -> CompletionResponse:
            return CompletionResponse(
                content="Mock response",
                tokens_used=50,
                input_tokens=30,
                output_tokens=20,
                model="mock-model",
                finish_reason="stop",
                cost_usd=0.001,
                latency_ms=100.0,
            )

        async def stream(self, messages, **kwargs):
            yield StreamChunk(content="Mock ", chunk_index=0)
            yield StreamChunk(content="response", chunk_index=1, is_final=True)

        def count_tokens(self, text: str, model=None) -> int:
            return len(text) // 4

    return MockProvider


@pytest.fixture
def sample_messages() -> list[Message]:
    """Create sample messages for testing."""
    return [
        Message(role="user", content="Hello, how are you?"),
    ]


@pytest.fixture
def mock_completion_response() -> CompletionResponse:
    """Create a mock completion response."""
    return CompletionResponse(
        content="I'm doing well, thank you!",
        tokens_used=25,
        input_tokens=10,
        output_tokens=15,
        model="test-model",
        finish_reason="stop",
        cost_usd=0.0001,
        latency_ms=100.0,
    )


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the provider registry before each test."""
    _PROVIDER_REGISTRY.clear()
    yield
    _PROVIDER_REGISTRY.clear()


# =============================================================================
# Provider Factory Tests
# =============================================================================


class TestProviderFactory:
    """Tests for ProviderFactory class."""

    def test_factory_initialization(self, mock_config: MagicMock) -> None:
        """Test factory initialization."""
        with patch("contextflow.providers.factory.get_config", return_value=mock_config):
            factory = ProviderFactory(config=mock_config)

            assert factory.config == mock_config

    def test_factory_register_provider(self, mock_provider_class: type) -> None:
        """Test registering a provider through factory."""
        ProviderFactory.register("test_provider", mock_provider_class)

        assert ProviderFactory.is_available("test_provider")

    def test_factory_list_providers(self, mock_provider_class: type) -> None:
        """Test listing providers through factory."""
        register_provider("test1", mock_provider_class)
        register_provider("test2", mock_provider_class)

        providers = ProviderFactory.list()

        assert "test1" in providers
        assert "test2" in providers


# =============================================================================
# Provider Registration Tests
# =============================================================================


class TestProviderRegistration:
    """Tests for provider registration functions."""

    def test_register_custom_provider(self, mock_provider_class: type) -> None:
        """Test registering a custom provider."""
        register_provider("custom", mock_provider_class)

        assert is_provider_available("custom")
        assert "custom" in list_providers()

    def test_register_non_provider_raises(self) -> None:
        """Test that registering non-provider raises TypeError."""

        class NotAProvider:
            pass

        with pytest.raises(TypeError, match="must inherit from BaseProvider"):
            register_provider("invalid", NotAProvider)

    def test_list_providers_includes_registered(self, mock_provider_class: type) -> None:
        """Test that list_providers includes registered providers."""
        register_provider("registered", mock_provider_class)

        providers = list_providers()

        assert "registered" in providers

    def test_is_provider_available(self, mock_provider_class: type) -> None:
        """Test provider availability check."""
        register_provider("available", mock_provider_class)

        assert is_provider_available("available") is True
        assert is_provider_available("not_registered") is False

    def test_get_available_providers_alias(self, mock_provider_class: type) -> None:
        """Test that get_available_providers is alias for list_providers."""
        register_provider("test", mock_provider_class)

        available = get_available_providers()
        listed = list_providers()

        assert available == listed


# =============================================================================
# Get Provider Tests
# =============================================================================


class TestGetProvider:
    """Tests for get_provider function."""

    def test_get_provider_unknown_raises(self) -> None:
        """Test that unknown provider raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Unknown provider"):
            get_provider("unknown_provider")

    def test_get_provider_creates_instance(
        self, mock_config: MagicMock, mock_provider_class: type
    ) -> None:
        """Test that get_provider creates provider instance."""
        register_provider("mock", mock_provider_class)

        with patch("contextflow.providers.factory.get_config", return_value=mock_config):
            provider = get_provider(name="mock", config=mock_config)

            assert provider is not None
            assert provider.name == "mock"

    def test_get_provider_with_explicit_api_key(
        self, mock_config: MagicMock, mock_provider_class: type
    ) -> None:
        """Test get_provider with explicit API key."""
        register_provider("mock", mock_provider_class)

        with patch("contextflow.providers.factory.get_config", return_value=mock_config):
            provider = get_provider(name="mock", api_key="explicit-key", config=mock_config)

            assert provider.api_key == "explicit-key"

    def test_get_provider_with_explicit_model(
        self, mock_config: MagicMock, mock_provider_class: type
    ) -> None:
        """Test get_provider with explicit model."""
        register_provider("mock", mock_provider_class)

        with patch("contextflow.providers.factory.get_config", return_value=mock_config):
            provider = get_provider(name="mock", model="custom-model", config=mock_config)

            assert provider.model == "custom-model"


# =============================================================================
# BaseProvider Tests
# =============================================================================


class TestBaseProvider:
    """Tests for BaseProvider abstract class."""

    def test_provider_initialization(self, mock_provider_class: type) -> None:
        """Test provider initialization with parameters."""
        provider = mock_provider_class(
            model="test-model",
            api_key="test-key",
            base_url="https://api.example.com",
            timeout=60,
            max_retries=5,
        )

        assert provider.model == "test-model"
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://api.example.com"
        assert provider.timeout == 60
        assert provider.max_retries == 5

    def test_provider_name_property(self, mock_provider_class: type) -> None:
        """Test provider name property."""
        provider = mock_provider_class(model="test")

        assert provider.name == "mock"

    def test_provider_capabilities_property(self, mock_provider_class: type) -> None:
        """Test provider capabilities property."""
        provider = mock_provider_class(model="test")

        caps = provider.capabilities

        assert isinstance(caps, ProviderCapabilities)
        assert caps.max_context_tokens > 0
        assert caps.supports_streaming is True

    def test_provider_repr(self, mock_provider_class: type) -> None:
        """Test provider string representation."""
        provider = mock_provider_class(model="test-model")

        repr_str = repr(provider)

        assert "MockProvider" in repr_str
        assert "test-model" in repr_str

    def test_provider_get_model_info(self, mock_provider_class: type) -> None:
        """Test getting model information."""
        provider = mock_provider_class(model="test-model")

        info = provider.get_model_info()

        assert "provider" in info
        assert "model" in info
        assert "capabilities" in info
        assert info["provider"] == "mock"
        assert info["model"] == "test-model"


# =============================================================================
# Provider Completion Tests
# =============================================================================


class TestProviderCompletion:
    """Tests for provider completion functionality."""

    @pytest.mark.asyncio
    async def test_complete_returns_response(
        self, mock_provider_class: type, sample_messages: list[Message]
    ) -> None:
        """Test that complete returns CompletionResponse."""
        provider = mock_provider_class(model="test")

        response = await provider.complete(messages=sample_messages)

        assert isinstance(response, CompletionResponse)
        assert response.content == "Mock response"
        assert response.tokens_used > 0

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(
        self, mock_provider_class: type, sample_messages: list[Message]
    ) -> None:
        """Test completion with system prompt."""
        provider = mock_provider_class(model="test")

        response = await provider.complete(
            messages=sample_messages, system="You are a helpful assistant."
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_complete_with_parameters(
        self, mock_provider_class: type, sample_messages: list[Message]
    ) -> None:
        """Test completion with various parameters."""
        provider = mock_provider_class(model="test")

        response = await provider.complete(
            messages=sample_messages, max_tokens=1000, temperature=0.5, top_p=0.9
        )

        assert response is not None


# =============================================================================
# Provider Streaming Tests
# =============================================================================


class TestProviderStreaming:
    """Tests for provider streaming functionality."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(
        self, mock_provider_class: type, sample_messages: list[Message]
    ) -> None:
        """Test that stream yields StreamChunk objects."""
        provider = mock_provider_class(model="test")

        chunks = []
        async for chunk in provider.stream(messages=sample_messages):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(isinstance(c, StreamChunk) for c in chunks)
        assert chunks[-1].is_final is True

    @pytest.mark.asyncio
    async def test_stream_content_accumulates(
        self, mock_provider_class: type, sample_messages: list[Message]
    ) -> None:
        """Test that stream content can be accumulated."""
        provider = mock_provider_class(model="test")

        content = ""
        async for chunk in provider.stream(messages=sample_messages):
            content += chunk.content

        assert len(content) > 0


# =============================================================================
# Provider Token Counting Tests
# =============================================================================


class TestProviderTokenCounting:
    """Tests for provider token counting."""

    def test_count_tokens_basic(self, mock_provider_class: type) -> None:
        """Test basic token counting."""
        provider = mock_provider_class(model="test")

        count = provider.count_tokens("This is a test sentence.")

        assert count > 0
        assert count < 100

    def test_count_tokens_empty_string(self, mock_provider_class: type) -> None:
        """Test token counting with empty string."""
        provider = mock_provider_class(model="test")

        count = provider.count_tokens("")

        assert count == 0

    def test_count_tokens_long_text(self, mock_provider_class: type) -> None:
        """Test token counting with long text."""
        provider = mock_provider_class(model="test")
        long_text = "This is a test. " * 1000

        count = provider.count_tokens(long_text)

        assert count > 100
        assert count < len(long_text)


# =============================================================================
# Provider Validation Tests
# =============================================================================


class TestProviderValidation:
    """Tests for provider validation."""

    @pytest.mark.asyncio
    async def test_validate_credentials_success(self, mock_provider_class: type) -> None:
        """Test successful credential validation."""
        provider = mock_provider_class(model="test")

        result = await provider.validate_credentials()

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_credentials_failure(self, mock_provider_class: type) -> None:
        """Test credential validation failure."""
        provider = mock_provider_class(model="test")

        # Make complete raise an error
        async def failing_complete(*args, **kwargs):
            raise Exception("Invalid credentials")

        provider.complete = failing_complete

        result = await provider.validate_credentials()

        assert result is False


# =============================================================================
# Message Conversion Tests
# =============================================================================


class TestMessageConversion:
    """Tests for message conversion functionality."""

    def test_convert_messages(
        self, mock_provider_class: type, sample_messages: list[Message]
    ) -> None:
        """Test message conversion to provider format."""
        provider = mock_provider_class(model="test")

        converted = provider._convert_messages(sample_messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello, how are you?"


# =============================================================================
# Provider Capabilities Tests
# =============================================================================


class TestProviderCapabilities:
    """Tests for ProviderCapabilities dataclass."""

    def test_capabilities_structure(self) -> None:
        """Test capabilities structure."""
        caps = ProviderCapabilities(
            max_context_tokens=200000,
            max_output_tokens=4096,
            supports_streaming=True,
            supports_system_prompt=True,
            supports_tools=True,
            supported_models=["model-a", "model-b"],
            rate_limit_rpm=60,
            rate_limit_tpm=100000,
        )

        assert caps.max_context_tokens == 200000
        assert caps.max_output_tokens == 4096
        assert caps.supports_streaming is True
        assert caps.supports_tools is True
        assert len(caps.supported_models) == 2


# =============================================================================
# CompletionResponse Tests
# =============================================================================


class TestCompletionResponse:
    """Tests for CompletionResponse dataclass."""

    def test_response_structure(self, mock_completion_response: CompletionResponse) -> None:
        """Test response structure."""
        assert mock_completion_response.content == "I'm doing well, thank you!"
        assert mock_completion_response.tokens_used == 25
        assert mock_completion_response.input_tokens == 10
        assert mock_completion_response.output_tokens == 15
        assert mock_completion_response.finish_reason == "stop"

    def test_response_cost(self, mock_completion_response: CompletionResponse) -> None:
        """Test response cost calculation."""
        assert mock_completion_response.cost_usd >= 0
        assert mock_completion_response.latency_ms >= 0


# =============================================================================
# StreamChunk Tests
# =============================================================================


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_chunk_structure(self) -> None:
        """Test chunk structure."""
        chunk = StreamChunk(content="Hello", is_final=False, chunk_index=0)

        assert chunk.content == "Hello"
        assert chunk.is_final is False
        assert chunk.chunk_index == 0

    def test_final_chunk(self) -> None:
        """Test final chunk."""
        chunk = StreamChunk(content="!", is_final=True, chunk_index=5)

        assert chunk.is_final is True


# =============================================================================
# Message Tests
# =============================================================================


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self) -> None:
        """Test message creation."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.metadata, dict)

    def test_message_with_metadata(self) -> None:
        """Test message with metadata."""
        msg = Message(role="assistant", content="Response", metadata={"timestamp": "2024-01-15"})

        assert msg.metadata["timestamp"] == "2024-01-15"

    def test_message_to_dict(self) -> None:
        """Test message serialization."""
        msg = Message(role="user", content="Hello")
        msg_dict = msg.to_dict()

        assert msg_dict["role"] == "user"
        assert msg_dict["content"] == "Hello"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in providers."""

    def test_missing_api_key_error(self) -> None:
        """Test MissingAPIKeyError."""
        error = MissingAPIKeyError("claude")

        assert "claude" in str(error).lower() or "api" in str(error).lower()

    def test_configuration_error(self) -> None:
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid configuration", config_key="test_key")

        assert "Invalid configuration" in str(error)


# =============================================================================
# Provider Type Tests
# =============================================================================


class TestProviderType:
    """Tests for ProviderType enum."""

    def test_provider_types_exist(self) -> None:
        """Test that all provider types exist."""
        expected = ["PROPRIETARY", "OPEN_SOURCE", "EXTERNAL_API", "LOCAL"]

        for type_name in expected:
            assert hasattr(ProviderType, type_name)

    def test_provider_type_values(self) -> None:
        """Test provider type values."""
        assert ProviderType.PROPRIETARY.value == "proprietary"
        assert ProviderType.OPEN_SOURCE.value == "open_source"
        assert ProviderType.LOCAL.value == "local"


# =============================================================================
# Integration-style Tests (with mocked internals)
# =============================================================================


class TestProviderIntegration:
    """Integration-style tests for provider functionality."""

    @pytest.mark.asyncio
    async def test_complete_and_count_tokens(
        self, mock_provider_class: type, sample_messages: list[Message]
    ) -> None:
        """Test completing a request and counting tokens."""
        provider = mock_provider_class(model="test")

        # Count input tokens
        input_text = " ".join(m.content for m in sample_messages)
        input_tokens = provider.count_tokens(input_text)

        # Get completion
        response = await provider.complete(messages=sample_messages)

        # Count output tokens
        output_tokens = provider.count_tokens(response.content)

        assert input_tokens > 0
        assert output_tokens > 0
        assert response.tokens_used > 0

    @pytest.mark.asyncio
    async def test_stream_complete_consistency(
        self, mock_provider_class: type, sample_messages: list[Message]
    ) -> None:
        """Test that stream produces content."""
        provider = mock_provider_class(model="test")

        # Stream content
        stream_content = ""
        async for chunk in provider.stream(messages=sample_messages):
            stream_content += chunk.content

        assert len(stream_content) > 0
