"""
Pytest configuration and fixtures for ContextFlow tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from contextflow.core.types import Message, CompletionResponse
from contextflow.core.config import ContextFlowConfig


@pytest.fixture
def mock_config() -> ContextFlowConfig:
    """Provide a test configuration."""
    return ContextFlowConfig(
        default_provider="claude",
    )


@pytest.fixture
def sample_messages() -> list[Message]:
    """Provide sample messages for testing."""
    return [
        Message(role="user", content="Hello, how are you?"),
    ]


@pytest.fixture
def mock_completion_response() -> CompletionResponse:
    """Provide a mock completion response."""
    return CompletionResponse(
        content="I'm doing well, thank you for asking!",
        tokens_used=25,
        input_tokens=10,
        output_tokens=15,
        model="claude-3-5-sonnet-20241022",
        finish_reason="stop",
        cost_usd=0.000045,
        latency_ms=500.0,
    )


@pytest.fixture
def mock_provider(mock_completion_response: CompletionResponse) -> MagicMock:
    """Provide a mock provider."""
    provider = MagicMock()
    provider.name = "mock"
    provider.model = "mock-model"
    provider.complete = AsyncMock(return_value=mock_completion_response)
    provider.stream = AsyncMock()
    provider.count_tokens = MagicMock(return_value=10)
    return provider


@pytest.fixture
def sample_text() -> str:
    """Provide sample text for token counting tests."""
    return """
    This is a sample text for testing the token estimator.
    It contains multiple sentences and should be representative
    of typical content that would be processed by ContextFlow.
    """


@pytest.fixture
def large_sample_text() -> str:
    """Provide a larger sample text."""
    paragraph = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.
    """
    return paragraph * 100  # ~30K characters
