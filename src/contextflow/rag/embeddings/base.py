"""
Base Embedding Provider ABC for ContextFlow RAG.

All embedding providers must implement this interface for
consistent vector generation across the system.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

# =============================================================================
# Enums
# =============================================================================


class EmbeddingProviderType(Enum):
    """Types of embedding providers."""

    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OLLAMA = "ollama"
    COHERE = "cohere"
    CUSTOM = "custom"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EmbeddingResult:
    """
    Result from embedding generation.

    Attributes:
        vectors: 2D numpy array of shape (n_texts, embedding_dim)
        model: Name of the model used for embedding
        dimensions: Dimensionality of each embedding vector
        token_count: Total tokens processed
        processing_time: Time taken in seconds
        metadata: Additional provider-specific metadata
    """

    vectors: npt.NDArray[np.float32]  # Shape: (n_texts, embedding_dim)
    model: str
    dimensions: int
    token_count: int
    processing_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the embedding result."""
        if self.vectors.ndim != 2:
            raise ValueError(f"Expected 2D array for vectors, got {self.vectors.ndim}D")
        if self.vectors.shape[1] != self.dimensions:
            raise ValueError(
                f"Vector dimensions {self.vectors.shape[1]} do not match "
                f"declared dimensions {self.dimensions}"
            )


@dataclass
class EmbeddingCapabilities:
    """
    Capabilities of an embedding provider.

    Attributes:
        max_tokens_per_text: Maximum tokens allowed per single text
        max_batch_size: Maximum number of texts per batch request
        dimensions: Output embedding dimensions
        supports_batching: Whether provider supports batch requests
        supports_truncation: Whether provider auto-truncates long texts
        normalized: Whether vectors are L2 normalized by default
    """

    max_tokens_per_text: int
    max_batch_size: int
    dimensions: int
    supports_batching: bool = True
    supports_truncation: bool = True
    normalized: bool = True  # Whether vectors are L2 normalized


# =============================================================================
# Abstract Base Class
# =============================================================================


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    All providers must implement:
    - embed(): Generate embeddings for multiple texts
    - embed_query(): Embed a single query (may use different model)
    - get_dimensions(): Return embedding dimensions
    - validate_connection(): Verify provider connection/credentials

    Example implementation:
        class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
            @property
            def name(self) -> str:
                return "openai"

            @property
            def provider_type(self) -> EmbeddingProviderType:
                return EmbeddingProviderType.OPENAI

            async def embed(self, texts, **kwargs) -> EmbeddingResult:
                # Implementation here
                pass
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Provider identifier.

        Returns:
            Unique provider name (e.g., "openai", "sentence_transformers")
        """
        pass

    @property
    @abstractmethod
    def provider_type(self) -> EmbeddingProviderType:
        """
        Provider type enum.

        Returns:
            EmbeddingProviderType enum value
        """
        pass

    @property
    @abstractmethod
    def capabilities(self) -> EmbeddingCapabilities:
        """
        Provider capabilities and limits.

        Returns:
            EmbeddingCapabilities with dimensions, batch size, etc.
        """
        pass

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        batch_size: int | None = None,
    ) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: Optional model override
            batch_size: Optional batch size override

        Returns:
            EmbeddingResult with vectors and metadata

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If texts list is empty
        """
        pass

    @abstractmethod
    async def embed_query(
        self,
        query: str,
        model: str | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Embed a single query.

        Some providers use different models for queries vs documents
        to optimize retrieval quality.

        Args:
            query: Query text to embed
            model: Optional model override

        Returns:
            1D numpy array of embedding

        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass

    @abstractmethod
    def get_dimensions(self, model: str | None = None) -> int:
        """
        Get embedding dimensions for the model.

        Args:
            model: Optional model to get dimensions for

        Returns:
            Number of dimensions in the embedding vector
        """
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """
        Validate provider connection and credentials.

        Returns:
            True if connection is valid and working

        Raises:
            EmbeddingError: If validation fails with specific error
        """
        pass

    async def embed_single(self, text: str) -> npt.NDArray[np.float32]:
        """
        Convenience method to embed a single text.

        Args:
            text: Text to embed

        Returns:
            1D numpy array of embedding
        """
        result = await self.embed([text])
        return result.vectors[0]

    def validate_texts(self, texts: list[str]) -> list[str]:
        """
        Validate and clean texts before embedding.

        Performs the following validations:
        - Removes empty strings
        - Strips whitespace
        - Warns about texts exceeding max token limit

        Args:
            texts: List of texts to validate

        Returns:
            Cleaned list of texts

        Raises:
            ValueError: If all texts are empty after cleaning
        """
        cleaned: list[str] = []

        for text in texts:
            # Strip whitespace
            stripped = text.strip()

            # Skip empty strings
            if not stripped:
                continue

            cleaned.append(stripped)

        if not cleaned:
            raise ValueError("No valid texts to embed after cleaning")

        return cleaned

    def _create_result(
        self,
        vectors: npt.NDArray[np.float32],
        model: str,
        token_count: int,
        start_time: float,
        metadata: dict[str, Any] | None = None,
    ) -> EmbeddingResult:
        """
        Helper to create EmbeddingResult with timing.

        Args:
            vectors: Generated embedding vectors
            model: Model name used
            token_count: Total tokens processed
            start_time: Start time from time.perf_counter()
            metadata: Optional additional metadata

        Returns:
            EmbeddingResult with all fields populated
        """
        processing_time = time.perf_counter() - start_time

        return EmbeddingResult(
            vectors=vectors,
            model=model,
            dimensions=vectors.shape[1],
            token_count=token_count,
            processing_time=processing_time,
            metadata=metadata or {},
        )

    def normalize_vectors(
        self,
        vectors: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """
        L2 normalize vectors.

        Args:
            vectors: Vectors to normalize, shape (n, dim)

        Returns:
            Normalized vectors with unit L2 norm
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms > 0, norms, 1.0)
        return vectors / norms

    def get_provider_info(self) -> dict[str, Any]:
        """
        Get information about the provider.

        Returns:
            Dictionary with provider details
        """
        return {
            "provider": self.name,
            "type": self.provider_type.value,
            "capabilities": {
                "max_tokens_per_text": self.capabilities.max_tokens_per_text,
                "max_batch_size": self.capabilities.max_batch_size,
                "dimensions": self.capabilities.dimensions,
                "supports_batching": self.capabilities.supports_batching,
                "supports_truncation": self.capabilities.supports_truncation,
                "normalized": self.capabilities.normalized,
            },
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"type={self.provider_type.value!r})"
        )
