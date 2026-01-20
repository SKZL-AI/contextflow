"""
OpenAI Embedding Provider for ContextFlow.

Supports:
- text-embedding-3-small (1536 dims, cheapest)
- text-embedding-3-large (3072 dims, best quality)
- text-embedding-ada-002 (1536 dims, legacy)
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt
import tiktoken
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from contextflow.rag.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingCapabilities,
    EmbeddingProviderType,
    EmbeddingResult,
)
from contextflow.utils.errors import EmbeddingError
from contextflow.utils.logging import get_logger

# Logger
logger = get_logger(__name__)


# =============================================================================
# Model Configurations
# =============================================================================

OPENAI_EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "text-embedding-3-small": {
        "dimensions": 1536,
        "max_tokens": 8191,
        "price_per_1m": 0.02,
        "supports_dimension_reduction": True,
        "min_dimensions": 256,
    },
    "text-embedding-3-large": {
        "dimensions": 3072,
        "max_tokens": 8191,
        "price_per_1m": 0.13,
        "supports_dimension_reduction": True,
        "min_dimensions": 256,
    },
    "text-embedding-ada-002": {
        "dimensions": 1536,
        "max_tokens": 8191,
        "price_per_1m": 0.10,
        "supports_dimension_reduction": False,
        "min_dimensions": 1536,
    },
}

# Default model
DEFAULT_MODEL = "text-embedding-3-small"

# API constants
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 30.0
DEFAULT_BATCH_SIZE = 2048
MAX_BATCH_SIZE = 2048


# =============================================================================
# OpenAI Embedding Provider
# =============================================================================


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI Embedding Provider.

    Provides high-quality embeddings using OpenAI's embedding models.
    Supports batching, retry logic, and cost tracking.

    Models:
        - text-embedding-3-small: 1536 dims, $0.02/1M tokens (recommended)
        - text-embedding-3-large: 3072 dims, $0.13/1M tokens (best quality)
        - text-embedding-ada-002: 1536 dims, $0.10/1M tokens (legacy)

    Usage:
        # Basic usage
        provider = OpenAIEmbeddingProvider(api_key="sk-...")
        result = await provider.embed(["Hello world", "Another text"])
        print(result.vectors.shape)  # (2, 1536)

        # Embed query for search
        query_vec = await provider.embed_query("search query")

        # Use different model
        result = await provider.embed(
            texts=["text"],
            model="text-embedding-3-large"
        )

        # Reduce dimensions (text-embedding-3-* only)
        provider = OpenAIEmbeddingProvider(
            api_key="sk-...",
            model="text-embedding-3-large",
            dimensions=1024  # Reduce from 3072 to 1024
        )
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        dimensions: int | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        """
        Initialize OpenAI Embedding Provider.

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            model: Embedding model to use
            dimensions: Optional dimension reduction (text-embedding-3-* only)
            base_url: API base URL (for Azure or proxies)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts

        Raises:
            ValueError: If model is unknown or dimensions are invalid
        """
        # Get API key from env if not provided
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            logger.warning(
                "No API key provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Validate model
        if model not in OPENAI_EMBEDDING_MODELS:
            raise ValueError(
                f"Unknown model: {model}. "
                f"Supported models: {list(OPENAI_EMBEDDING_MODELS.keys())}"
            )

        self._model = model
        self._model_config = OPENAI_EMBEDDING_MODELS[model]
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries

        # Handle dimension reduction
        self._dimensions = self._validate_dimensions(dimensions)

        # Initialize HTTP client
        self._client: httpx.AsyncClient | None = None

        # Initialize tokenizer for token counting
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tokenizer = None
            logger.warning("Failed to initialize tiktoken. Token counting may be inaccurate.")

        logger.info(
            "Initialized OpenAI embedding provider",
            model=self._model,
            dimensions=self._dimensions,
            base_url=self._base_url,
        )

    def _validate_dimensions(self, dimensions: int | None) -> int:
        """Validate and return dimensions."""
        model_dims = self._model_config["dimensions"]
        min_dims = self._model_config["min_dimensions"]
        supports_reduction = self._model_config["supports_dimension_reduction"]

        if dimensions is None:
            return model_dims

        if not supports_reduction:
            if dimensions != model_dims:
                logger.warning(
                    f"Model {self._model} does not support dimension reduction. "
                    f"Using default dimensions: {model_dims}"
                )
            return model_dims

        if dimensions < min_dims:
            raise ValueError(
                f"Dimensions must be at least {min_dims} for {self._model}. " f"Got: {dimensions}"
            )

        if dimensions > model_dims:
            raise ValueError(
                f"Dimensions cannot exceed {model_dims} for {self._model}. " f"Got: {dimensions}"
            )

        return dimensions

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OpenAIEmbeddingProvider:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def name(self) -> str:
        """Provider name."""
        return "openai"

    @property
    def provider_type(self) -> EmbeddingProviderType:
        """Provider type."""
        return EmbeddingProviderType.OPENAI

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        """Provider capabilities."""
        return EmbeddingCapabilities(
            max_tokens_per_text=self._model_config["max_tokens"],
            max_batch_size=MAX_BATCH_SIZE,
            dimensions=self._dimensions,
            supports_batching=True,
            supports_truncation=True,
            normalized=True,  # OpenAI embeddings are L2 normalized
        )

    @property
    def model(self) -> str:
        """Current model."""
        return self._model

    @property
    def embedding_dimensions(self) -> int:
        """Current dimensions."""
        return self._dimensions

    # =========================================================================
    # Core Methods
    # =========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
        reraise=True,
    )
    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        batch_size: int | None = None,
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            model: Optional model override
            batch_size: Optional batch size (default: 2048)

        Returns:
            EmbeddingResult with vectors and metadata

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If texts are empty
        """
        # Validate inputs
        texts = self.validate_texts(texts)

        # Use provided or default values
        effective_model = model or self._model
        effective_batch_size = min(batch_size or DEFAULT_BATCH_SIZE, MAX_BATCH_SIZE)

        # Get model config
        if effective_model not in OPENAI_EMBEDDING_MODELS:
            raise ValueError(f"Unknown model: {effective_model}")
        model_config = OPENAI_EMBEDDING_MODELS[effective_model]

        start_time = time.perf_counter()
        all_embeddings: list[npt.NDArray[np.float32]] = []
        total_tokens = 0

        # Process in batches
        for i in range(0, len(texts), effective_batch_size):
            batch = texts[i : i + effective_batch_size]

            logger.debug(
                "Processing embedding batch",
                batch_index=i // effective_batch_size,
                batch_size=len(batch),
                total_texts=len(texts),
            )

            # Call API
            response = await self._call_api(batch, effective_model)

            # Extract embeddings
            batch_embeddings = self._extract_embeddings(response)
            all_embeddings.extend(batch_embeddings)

            # Track tokens
            if "usage" in response:
                total_tokens += response["usage"].get("total_tokens", 0)

        # Combine all embeddings into a 2D array
        vectors = np.vstack(all_embeddings).astype(np.float32)

        # Calculate cost
        cost_usd = (total_tokens / 1_000_000) * model_config["price_per_1m"]

        logger.info(
            "Generated embeddings",
            model=effective_model,
            count=len(texts),
            dimensions=vectors.shape[1],
            tokens=total_tokens,
            cost_usd=round(cost_usd, 6),
        )

        return self._create_result(
            vectors=vectors,
            model=effective_model,
            token_count=total_tokens,
            start_time=start_time,
            metadata={
                "cost_usd": cost_usd,
                "batch_count": (len(texts) + effective_batch_size - 1) // effective_batch_size,
            },
        )

    async def embed_query(
        self,
        query: str,
        model: str | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Embed a single query.

        Optimized for single query embedding (e.g., for search).

        Args:
            query: Query text to embed
            model: Optional model override

        Returns:
            1D numpy array with embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        result = await self.embed([query.strip()], model=model)
        return result.vectors[0]

    def get_dimensions(self, model: str | None = None) -> int:
        """
        Get embedding dimensions for a model.

        Args:
            model: Optional model (uses default if not provided)

        Returns:
            Number of dimensions
        """
        if model is None:
            return self._dimensions

        if model not in OPENAI_EMBEDDING_MODELS:
            raise ValueError(f"Unknown model: {model}")

        return OPENAI_EMBEDDING_MODELS[model]["dimensions"]

    async def validate_connection(self) -> bool:
        """
        Validate API key by making a test request.

        Returns:
            True if connection is valid, False otherwise

        Raises:
            EmbeddingError: If validation fails with specific error
        """
        if not self._api_key:
            logger.warning("No API key configured")
            return False

        try:
            await self.embed(["test"])
            return True
        except EmbeddingError as e:
            logger.warning("Connection validation failed", error=str(e))
            return False
        except Exception as e:
            logger.error("Unexpected error during validation", error=str(e))
            return False

    # =========================================================================
    # API Methods
    # =========================================================================

    async def _call_api(
        self,
        texts: list[str],
        model: str,
    ) -> dict[str, Any]:
        """
        Make API call to OpenAI embeddings endpoint.

        Args:
            texts: Texts to embed
            model: Model to use

        Returns:
            API response dictionary

        Raises:
            EmbeddingError: If API call fails
        """
        client = await self._get_client()

        # Build request payload
        payload: dict[str, Any] = {
            "model": model,
            "input": texts,
        }

        # Add dimensions if using dimension reduction
        model_config = OPENAI_EMBEDDING_MODELS.get(model, {})
        if model_config.get("supports_dimension_reduction", False):
            if self._dimensions != model_config.get("dimensions"):
                payload["dimensions"] = self._dimensions

        url = f"{self._base_url}/embeddings"

        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_body = e.response.text

            # Parse error message
            error_message = self._parse_error(error_body)

            logger.error(
                "OpenAI API error",
                status_code=status_code,
                error=error_message,
            )

            # Handle specific error codes
            if status_code == 401:
                raise EmbeddingError(
                    provider="openai",
                    message="Authentication failed. Check your API key.",
                    details={"status_code": status_code},
                ) from e

            elif status_code == 429:
                raise EmbeddingError(
                    provider="openai",
                    message="Rate limit exceeded. Please retry later.",
                    details={"status_code": status_code},
                ) from e

            elif status_code == 400:
                raise EmbeddingError(
                    provider="openai",
                    message=f"Bad request: {error_message}",
                    details={"status_code": status_code},
                ) from e

            else:
                raise EmbeddingError(
                    provider="openai",
                    message=f"API error: {error_message}",
                    details={"status_code": status_code},
                ) from e

        except httpx.TimeoutException as e:
            logger.error("OpenAI API timeout", timeout=self._timeout)
            raise EmbeddingError(
                provider="openai",
                message=f"Request timed out after {self._timeout}s",
                details={"timeout": self._timeout},
            ) from e

        except httpx.RequestError as e:
            logger.error("OpenAI API request error", error=str(e))
            raise EmbeddingError(
                provider="openai",
                message=f"Request failed: {str(e)}",
            ) from e

    def _parse_error(self, error_body: str) -> str:
        """Parse error message from API response."""
        try:
            import json

            data = json.loads(error_body)
            if "error" in data:
                error = data["error"]
                if isinstance(error, dict):
                    return error.get("message", str(error))
                return str(error)
            return error_body
        except Exception:
            return error_body

    def _extract_embeddings(self, response: dict[str, Any]) -> list[npt.NDArray[np.float32]]:
        """
        Extract embeddings from API response.

        Args:
            response: API response dictionary

        Returns:
            List of embedding vectors
        """
        if "data" not in response:
            raise EmbeddingError(
                provider="openai",
                message="Invalid API response: missing 'data' field",
            )

        embeddings: list[npt.NDArray[np.float32]] = []
        for item in response["data"]:
            if "embedding" not in item:
                raise EmbeddingError(
                    provider="openai",
                    message="Invalid API response: missing 'embedding' field",
                )
            embeddings.append(np.array(item["embedding"], dtype=np.float32))

        return embeddings

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def estimate_cost(self, texts: list[str]) -> float:
        """
        Estimate embedding cost.

        Args:
            texts: Texts to estimate cost for

        Returns:
            Estimated cost in USD
        """
        total_tokens = self.count_tokens(texts)
        price_per_1m = self._model_config["price_per_1m"]
        return (total_tokens / 1_000_000) * price_per_1m

    def count_tokens(self, texts: list[str]) -> int:
        """
        Count tokens in texts using tiktoken.

        Args:
            texts: Texts to count tokens for

        Returns:
            Total token count
        """
        if not texts:
            return 0

        if self._tokenizer:
            # Use tiktoken for accurate counting
            total = 0
            for text in texts:
                if text:
                    total += len(self._tokenizer.encode(text))
            return total
        else:
            # Fallback: estimate ~4 chars per token
            total_chars = sum(len(t) for t in texts if t)
            return total_chars // 4

    def get_model_info(self, model: str | None = None) -> dict[str, Any]:
        """
        Get information about a model.

        Args:
            model: Model name (uses current model if not provided)

        Returns:
            Model configuration dictionary
        """
        model = model or self._model
        if model not in OPENAI_EMBEDDING_MODELS:
            raise ValueError(f"Unknown model: {model}")
        return OPENAI_EMBEDDING_MODELS[model].copy()

    @staticmethod
    def list_models() -> list[str]:
        """
        List available models.

        Returns:
            List of model names
        """
        return list(OPENAI_EMBEDDING_MODELS.keys())


# =============================================================================
# Factory Function
# =============================================================================


def create_openai_embedding_provider(
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    dimensions: int | None = None,
    **kwargs: Any,
) -> OpenAIEmbeddingProvider:
    """
    Factory function to create OpenAI embedding provider.

    Args:
        api_key: OpenAI API key
        model: Embedding model to use
        dimensions: Optional dimension reduction
        **kwargs: Additional arguments

    Returns:
        Configured OpenAIEmbeddingProvider instance

    Example:
        provider = create_openai_embedding_provider(
            model="text-embedding-3-large",
            dimensions=1024
        )
    """
    return OpenAIEmbeddingProvider(
        api_key=api_key,
        model=model,
        dimensions=dimensions,
        **kwargs,
    )


__all__ = [
    "OpenAIEmbeddingProvider",
    "OPENAI_EMBEDDING_MODELS",
    "DEFAULT_MODEL",
    "create_openai_embedding_provider",
]
