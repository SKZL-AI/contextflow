"""
Integration test fixtures for ContextFlow.

Provides fully configured instances with mock providers for end-to-end testing.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from contextflow.core.config import ContextFlowConfig, RAGConfig, StrategyConfig
from contextflow.core.hooks import HookContext, HooksManager, HookType, reset_global_hooks_manager
from contextflow.core.orchestrator import ContextFlow, OrchestratorConfig
from contextflow.core.types import (
    CompletionResponse,
    Message,
    ProviderCapabilities,
    ProviderType,
    StreamChunk,
)
from contextflow.providers.base import BaseProvider

# =============================================================================
# Mock Provider Implementation
# =============================================================================


class MockProvider(BaseProvider):
    """
    Mock provider for integration testing.

    Simulates LLM behavior with configurable responses.
    """

    def __init__(
        self,
        model: str = "mock-model",
        responses: list[str] | None = None,
        verification_responses: list[dict[str, Any]] | None = None,
        stream_chunks: list[str] | None = None,
        fail_after: int | None = None,
        tokens_per_char: float = 0.25,
    ):
        """
        Initialize mock provider.

        Args:
            model: Model name
            responses: List of responses to return sequentially
            verification_responses: List of verification JSON responses
            stream_chunks: Chunks for streaming
            fail_after: Fail after this many calls
            tokens_per_char: Tokens per character for counting
        """
        super().__init__(model=model, api_key="mock-key")
        self._responses = responses or ["This is a mock response."]
        self._verification_responses = verification_responses or []
        self._stream_chunks = stream_chunks or ["Streaming ", "response ", "here."]
        self._fail_after = fail_after
        self._tokens_per_char = tokens_per_char
        self._call_count = 0
        self._complete_calls: list[dict[str, Any]] = []
        self._stream_calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "mock"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.LOCAL

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            max_context_tokens=200_000,
            max_output_tokens=8192,
            supports_streaming=True,
            supports_system_prompt=True,
            supports_tools=False,
            supported_models=["mock-model"],
            rate_limit_rpm=1000,
            rate_limit_tpm=100_000,
        )

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
        """Return mock completion response."""
        self._call_count += 1

        # Store call info
        self._complete_calls.append(
            {
                "messages": messages,
                "system": system,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "call_number": self._call_count,
            }
        )

        # Check for failure simulation
        if self._fail_after and self._call_count > self._fail_after:
            from contextflow.utils.errors import ProviderError

            raise ProviderError("Simulated provider failure")

        # Check if this is a verification call (system prompt contains verification keywords)
        if system and "verification" in system.lower():
            return self._create_verification_response(messages)

        # Get response
        response_idx = (self._call_count - 1) % len(self._responses)
        content = self._responses[response_idx]

        # Calculate tokens
        total_input = sum(len(m.content) for m in messages)
        if system:
            total_input += len(system)
        input_tokens = int(total_input * self._tokens_per_char)
        output_tokens = int(len(content) * self._tokens_per_char)

        return CompletionResponse(
            content=content,
            tokens_used=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model or self.model,
            finish_reason="stop",
            cost_usd=0.0001,
            latency_ms=50.0,
        )

    def _create_verification_response(
        self,
        messages: list[Message],
    ) -> CompletionResponse:
        """Create a verification response."""
        # Check for pre-configured verification responses
        if self._verification_responses:
            response_idx = min(
                len(self._complete_calls) - 1,
                len(self._verification_responses) - 1,
            )
            verification_data = self._verification_responses[response_idx]
        else:
            # Default: pass verification
            verification_data = {
                "passed": True,
                "score": 0.85,
                "message": "Output meets requirements",
                "issues": [],
            }

        content = json.dumps(verification_data)

        return CompletionResponse(
            content=content,
            tokens_used=100,
            input_tokens=80,
            output_tokens=20,
            model=self.model,
            finish_reason="stop",
            cost_usd=0.00001,
            latency_ms=30.0,
        )

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
        """Stream mock response chunks."""
        self._stream_calls.append(
            {
                "messages": messages,
                "system": system,
            }
        )

        for i, chunk in enumerate(self._stream_chunks):
            yield StreamChunk(
                content=chunk,
                is_final=(i == len(self._stream_chunks) - 1),
                chunk_index=i,
            )
            await asyncio.sleep(0.01)

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """Count tokens using character ratio."""
        return int(len(text) * self._tokens_per_char)

    @property
    def call_count(self) -> int:
        """Get number of complete calls made."""
        return self._call_count

    @property
    def complete_calls(self) -> list[dict[str, Any]]:
        """Get list of complete call details."""
        return self._complete_calls

    @property
    def stream_calls(self) -> list[dict[str, Any]]:
        """Get list of stream call details."""
        return self._stream_calls

    def reset(self) -> None:
        """Reset call tracking."""
        self._call_count = 0
        self._complete_calls = []
        self._stream_calls = []


class MockProviderWithVerification(MockProvider):
    """
    Mock provider that returns verification-ready responses.

    Configurable to fail verification initially, then pass.
    """

    def __init__(
        self,
        fail_verification_count: int = 0,
        **kwargs: Any,
    ):
        """
        Initialize with verification failure count.

        Args:
            fail_verification_count: Number of times to fail verification before passing
        """
        # Create verification responses
        verification_responses = []

        # Add failing responses
        for i in range(fail_verification_count):
            verification_responses.append(
                {
                    "passed": False,
                    "score": 0.4 + (i * 0.1),
                    "message": f"Verification failed (attempt {i + 1})",
                    "issues": [f"Issue {i + 1}"],
                    "suggestions": [f"Improve point {i + 1}"],
                }
            )

        # Add passing response
        verification_responses.append(
            {
                "passed": True,
                "score": 0.9,
                "message": "Verification passed",
                "issues": [],
                "suggestions": [],
            }
        )

        super().__init__(
            verification_responses=verification_responses,
            **kwargs,
        )


# =============================================================================
# Mock Embedding Provider
# =============================================================================


class MockEmbeddingProvider:
    """Mock embedding provider for RAG testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._call_count = 0

    def get_dimensions(self) -> int:
        return self._dimension

    async def embed_query(self, text: str) -> np.ndarray:
        """Generate mock embedding for query."""
        self._call_count += 1
        # Generate deterministic embedding based on text hash
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self._dimension).astype(np.float32)
        return embedding / np.linalg.norm(embedding)

    async def embed(self, texts: list[str]) -> MockEmbeddingResult:
        """Generate mock embeddings for multiple texts."""
        self._call_count += 1
        vectors = []
        for text in texts:
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(self._dimension).astype(np.float32)
            vectors.append(embedding / np.linalg.norm(embedding))
        return MockEmbeddingResult(vectors=vectors)


@dataclass
class MockEmbeddingResult:
    """Mock embedding result."""

    vectors: list[np.ndarray]


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def mock_provider() -> MockProvider:
    """Provide a basic mock provider."""
    return MockProvider()


@pytest.fixture
def mock_provider_with_responses() -> MockProvider:
    """Provide a mock provider with custom responses."""
    return MockProvider(
        responses=[
            "This is the summary of the document.",
            "Here are the key points identified.",
            "The analysis is complete.",
        ]
    )


@pytest.fixture
def mock_provider_with_verification() -> MockProviderWithVerification:
    """Provide a mock provider that fails verification initially."""
    return MockProviderWithVerification(fail_verification_count=1)


@pytest.fixture
def mock_provider_always_pass_verification() -> MockProvider:
    """Provide a mock provider that always passes verification."""
    return MockProvider(
        verification_responses=[
            {
                "passed": True,
                "score": 0.95,
                "message": "All checks passed",
                "issues": [],
            }
        ]
    )


@pytest.fixture
def mock_provider_always_fail_verification() -> MockProvider:
    """Provide a mock provider that always fails verification."""
    return MockProvider(
        verification_responses=[
            {
                "passed": False,
                "score": 0.3,
                "message": "Verification failed",
                "issues": ["Critical issue found"],
            }
        ]
    )


@pytest.fixture
def mock_embedding_provider() -> MockEmbeddingProvider:
    """Provide a mock embedding provider."""
    return MockEmbeddingProvider()


@pytest.fixture
def test_config() -> ContextFlowConfig:
    """Provide a test configuration."""
    return ContextFlowConfig(
        default_provider="mock",
        strategy=StrategyConfig(
            gsd_max_tokens=10_000,
            ralph_max_tokens=100_000,
            rlm_min_tokens=100_000,
        ),
        rag=RAGConfig(
            chunk_size=1000,
            chunk_overlap=100,
        ),
    )


@pytest.fixture
def orchestrator_config() -> OrchestratorConfig:
    """Provide orchestrator configuration for testing."""
    return OrchestratorConfig(
        enable_verification=True,
        verification_threshold=0.7,
        max_verification_iterations=3,
        enable_sessions=False,  # Disable sessions for unit tests
        enable_hooks=True,
        enable_cost_tracking=True,
        enable_streaming=True,
        default_timeout=30.0,
    )


@pytest.fixture
def orchestrator_config_no_verification() -> OrchestratorConfig:
    """Provide orchestrator configuration without verification."""
    return OrchestratorConfig(
        enable_verification=False,
        enable_sessions=False,
        enable_hooks=True,
    )


@pytest.fixture
def hooks_manager() -> HooksManager:
    """Provide a fresh hooks manager."""
    reset_global_hooks_manager()
    return HooksManager(name="test")


@pytest.fixture
async def configured_contextflow(
    mock_provider: MockProvider,
    test_config: ContextFlowConfig,
    orchestrator_config: OrchestratorConfig,
    hooks_manager: HooksManager,
) -> AsyncIterator[ContextFlow]:
    """
    Provide a fully configured ContextFlow instance.

    Uses mock provider and test configuration.
    """
    cf = ContextFlow(
        provider=mock_provider,
        config=test_config,
        orchestrator_config=orchestrator_config,
        hooks_manager=hooks_manager,
    )

    await cf.initialize()

    yield cf

    await cf.close()


@pytest.fixture
async def contextflow_with_verification(
    mock_provider_with_verification: MockProviderWithVerification,
    test_config: ContextFlowConfig,
    hooks_manager: HooksManager,
) -> AsyncIterator[ContextFlow]:
    """
    Provide ContextFlow with verification that fails then passes.
    """
    orchestrator_config = OrchestratorConfig(
        enable_verification=True,
        verification_threshold=0.7,
        max_verification_iterations=3,
        enable_sessions=False,
        enable_hooks=True,
    )

    cf = ContextFlow(
        provider=mock_provider_with_verification,
        config=test_config,
        orchestrator_config=orchestrator_config,
        hooks_manager=hooks_manager,
    )

    await cf.initialize()

    yield cf

    await cf.close()


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def small_context() -> str:
    """Provide small context (<10K tokens)."""
    return (
        """
    This is a small document for testing.
    It contains a few paragraphs of text.

    The main topic is about testing ContextFlow.
    We want to verify that small contexts are handled correctly.

    Key points:
    1. Small contexts should use GSD strategy
    2. Processing should be fast
    3. Verification should pass
    """
        * 10
    )  # ~1K tokens


@pytest.fixture
def medium_context() -> str:
    """Provide medium context (10K-100K tokens)."""
    base_paragraph = (
        """
    This is a medium-sized document that contains substantial content.
    It is designed to test the RALPH strategy which handles contexts
    between 10K and 100K tokens. The content includes various sections
    and detailed information that requires structured processing.

    Section A: Introduction
    The ContextFlow framework provides intelligent context orchestration
    for large language models. It automatically selects the optimal
    processing strategy based on context size and complexity.

    Section B: Implementation Details
    The implementation uses a router to analyze context characteristics
    and select from GSD, RALPH, or RLM strategies. Each strategy has
    specific optimizations for different context sizes.
    """
        * 80
    )  # ~11K tokens (reduced from 200, needs >10K for RALPH)
    return base_paragraph


@pytest.fixture
def large_context() -> str:
    """Provide large context (>100K tokens)."""
    base_content = (
        """
    This is a large document designed to test the RLM (Recursive Language Model)
    strategy. It contains extensive content that requires recursive processing
    with sub-agents to handle effectively.

    Chapter 1: Architecture Overview
    The ContextFlow architecture is designed for scalability and flexibility.
    It supports multiple LLM providers and can handle contexts of any size.
    The key components include the orchestrator, strategy router, and verification
    protocol that ensures output quality.

    Chapter 2: Strategy Selection
    The strategy selection process analyzes context characteristics including
    token count, information density, and task complexity. Based on this
    analysis, it recommends the optimal strategy for processing.

    Chapter 3: Verification Protocol
    Boris Step 13 emphasizes the importance of verification for quality.
    The verification protocol provides 2-3x quality improvement by enabling
    the LLM to check its own work and iterate if necessary.
    """
        * 50
    )  # ~15K tokens (reduced from 500 for RAM safety)
    return base_content


@pytest.fixture
def sample_task() -> str:
    """Provide a sample task."""
    return "Summarize the main points of this document."


@pytest.fixture
def complex_task() -> str:
    """Provide a complex task."""
    return """
    Analyze this document comprehensively. Provide:
    1. A detailed summary of all main topics
    2. Key insights and patterns identified
    3. Comparison of different sections
    4. Recommendations based on the content
    5. Any potential issues or areas for improvement
    """


@pytest.fixture
def sample_constraints() -> list[str]:
    """Provide sample constraints for verification."""
    return [
        "Include all main topics",
        "Keep the summary under 500 words",
        "Use bullet points for key findings",
    ]


# =============================================================================
# Hook Tracking Fixtures
# =============================================================================


@pytest.fixture
def hook_tracker() -> dict[str, list[HookContext]]:
    """Provide a tracker for hook executions."""
    return {hook_type.value: [] for hook_type in HookType}


@pytest.fixture
def hooks_manager_with_tracking(
    hook_tracker: dict[str, list[HookContext]],
) -> HooksManager:
    """Provide a hooks manager that tracks all executions."""
    manager = HooksManager(name="tracking")

    async def track_hook(context: HookContext) -> HookContext:
        hook_tracker[context.hook_type.value].append(context)
        return context

    # Register tracking hook for all types
    for hook_type in HookType:
        manager.register(
            hook_type=hook_type,
            callback=track_hook,
            priority=1000,  # Run last to capture final state
            name=f"tracker_{hook_type.value}",
        )

    return manager
