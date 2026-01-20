"""
Ollama Embedding Provider for ContextFlow.

Local embeddings via Ollama server. Supports:
- nomic-embed-text (768 dims, best quality)
- mxbai-embed-large (1024 dims, multilingual)
- all-minilm (384 dims, fast)
- snowflake-arctic-embed (1024 dims)

No API key required - runs entirely local.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import numpy as np
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

logger = get_logger(__name__)

# Type alias for numpy array
NDArrayFloat32 = np.ndarray  # More explicit: npt.NDArray[np.float32]


# =============================================================================
# Model Configurations
# =============================================================================

OLLAMA_EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "nomic-embed-text": {
        "dimensions": 768,
        "max_tokens": 8192,
        "quality": "excellent",
        "speed": "normal",
        "description": "Best quality general-purpose embeddings",
    },
    "mxbai-embed-large": {
        "dimensions": 1024,
        "max_tokens": 512,
        "quality": "excellent",
        "speed": "normal",
        "multilingual": True,
        "description": "Multilingual embeddings with high quality",
    },
    "all-minilm": {
        "dimensions": 384,
        "max_tokens": 256,
        "quality": "good",
        "speed": "fast",
        "description": "Fast embeddings for quick prototyping",
    },
    "snowflake-arctic-embed": {
        "dimensions": 1024,
        "max_tokens": 512,
        "quality": "excellent",
        "speed": "normal",
        "description": "High-quality dense retrieval embeddings",
    },
    "bge-base-en": {
        "dimensions": 768,
        "max_tokens": 512,
        "quality": "good",
        "speed": "fast",
        "description": "BAAI general embedding model",
    },
    "bge-large-en": {
        "dimensions": 1024,
        "max_tokens": 512,
        "quality": "excellent",
        "speed": "normal",
        "description": "Large BAAI general embedding model",
    },
}

# Default model for Ollama embeddings
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


# =============================================================================
# Ollama Embedding Provider
# =============================================================================


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """
    Ollama Embedding Provider for local embeddings.

    Requires Ollama server running locally (default: http://localhost:11434).
    No API key required - all processing happens locally.

    Supported Models:
        - nomic-embed-text: 768 dims, 8K context, best general quality
        - mxbai-embed-large: 1024 dims, multilingual support
        - all-minilm: 384 dims, fastest option
        - snowflake-arctic-embed: 1024 dims, excellent for retrieval
        - bge-base-en / bge-large-en: BAAI embedding models

    Usage:
        # Initialize provider
        provider = OllamaEmbeddingProvider(model="nomic-embed-text")

        # Check connection
        if await provider.validate_connection():
            # Embed documents
            result = await provider.embed(["doc1", "doc2"])
            print(f"Embeddings shape: {result.embeddings.shape}")

            # Embed query
            query_vec = await provider.embed_query("search query")
            print(f"Query shape: {query_vec.shape}")

    Note:
        Ollama must be installed and running. Start with: `ollama serve`
        Pull embedding models with: `ollama pull nomic-embed-text`
    """

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_EMBEDDING_MODEL,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize Ollama Embedding Provider.

        Args:
            model: Ollama embedding model name (default: nomic-embed-text)
            base_url: Ollama server URL (default: http://localhost:11434)
            timeout: Request timeout in seconds (embeddings can be slow)
            max_retries: Maximum retry attempts for failed requests

        Raises:
            ValueError: If model is not in known model list
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries

        # Get model config or use defaults for unknown models
        self._model_config = OLLAMA_EMBEDDING_MODELS.get(
            model,
            {
                "dimensions": 768,
                "max_tokens": 512,
                "quality": "unknown",
                "speed": "normal",
            },
        )

        logger.info(
            "Initialized Ollama embedding provider",
            model=model,
            base_url=self._base_url,
            dimensions=self._model_config.get("dimensions"),
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Get provider name."""
        return "ollama"

    @property
    def provider_type(self) -> EmbeddingProviderType:
        """Get provider type."""
        return EmbeddingProviderType.OLLAMA

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        """Get provider capabilities based on current model."""
        return EmbeddingCapabilities(
            max_tokens_per_text=self._model_config.get("max_tokens", 512),
            max_batch_size=1,  # Ollama doesn't support batch API
            dimensions=self._model_config.get("dimensions", 768),
            supports_batching=False,  # Process one at a time
            supports_truncation=True,
            normalized=True,  # Ollama normalizes embeddings
        )

    # -------------------------------------------------------------------------
    # Core Methods
    # -------------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True,
    )
    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        Generate embeddings for multiple texts via Ollama API.

        Note: Ollama doesn't support batch embedding, so texts are
        processed sequentially. For large batches, consider using
        a provider with native batch support.

        Args:
            texts: List of texts to embed
            model: Optional model override
            **kwargs: Additional options (ignored for Ollama)

        Returns:
            EmbeddingResult with embeddings array (N x D)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            raise EmbeddingError(
                provider="ollama",
                message="Cannot embed empty text list",
            )

        use_model = model or self._model
        start_time = time.perf_counter()
        embeddings_list: list[list[float]] = []

        logger.debug(
            "Starting batch embedding",
            model=use_model,
            text_count=len(texts),
        )

        for idx, text in enumerate(texts):
            if not text or not text.strip():
                # Handle empty strings with zero vector
                dims = self.get_dimensions(use_model)
                embeddings_list.append([0.0] * dims)
                logger.warning(
                    "Empty text at index, using zero vector",
                    index=idx,
                )
                continue

            try:
                embedding = await self._call_api(text, use_model)
                embeddings_list.append(embedding)
            except Exception as e:
                logger.error(
                    "Failed to embed text",
                    index=idx,
                    text_preview=text[:50],
                    error=str(e),
                )
                raise EmbeddingError(
                    provider="ollama",
                    message=f"Failed to embed text at index {idx}: {e}",
                ) from e

        processing_time = time.perf_counter() - start_time
        vectors = np.array(embeddings_list, dtype=np.float32)

        logger.info(
            "Batch embedding completed",
            model=use_model,
            text_count=len(texts),
            processing_time_ms=round(processing_time * 1000, 2),
            dimensions=vectors.shape[-1],
        )

        return EmbeddingResult(
            vectors=vectors,
            model=use_model,
            dimensions=vectors.shape[-1],
            token_count=sum(len(t.split()) for t in texts),  # Approximate
            processing_time=processing_time,
            metadata={
                "provider": "ollama",
                "base_url": self._base_url,
            },
        )

    async def embed_query(
        self,
        query: str,
        model: str | None = None,
    ) -> NDArrayFloat32:
        """
        Embed a single query text.

        Args:
            query: Query text to embed
            model: Optional model override

        Returns:
            1D numpy array with query embedding

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not query or not query.strip():
            raise EmbeddingError(
                provider="ollama",
                message="Cannot embed empty query",
            )

        result = await self.embed([query], model=model)
        return result.vectors[0]

    def get_dimensions(self, model: str | None = None) -> int:
        """
        Get embedding dimensions for model.

        Args:
            model: Model name (uses default if None)

        Returns:
            Number of dimensions in embedding vectors
        """
        use_model = model or self._model
        config = OLLAMA_EMBEDDING_MODELS.get(use_model, self._model_config)
        return config.get("dimensions", 768)

    # -------------------------------------------------------------------------
    # API Methods
    # -------------------------------------------------------------------------

    async def _call_api(
        self,
        text: str,
        model: str,
    ) -> list[float]:
        """
        Make API call to Ollama embeddings endpoint.

        Args:
            text: Text to embed
            model: Model to use

        Returns:
            List of embedding floats

        Raises:
            EmbeddingError: If API call fails
            httpx.HTTPError: For retryable HTTP errors
        """
        url = f"{self._base_url}/api/embeddings"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    url,
                    json={
                        "model": model,
                        "prompt": text,
                    },
                )

                if response.status_code == 404:
                    raise EmbeddingError(
                        provider="ollama",
                        message=f"Model '{model}' not found. Pull it with: ollama pull {model}",
                    )

                response.raise_for_status()
                data = response.json()

                if "embedding" not in data:
                    raise EmbeddingError(
                        provider="ollama",
                        message="Invalid response from Ollama: missing 'embedding' field",
                    )

                return data["embedding"]

            except httpx.ConnectError as e:
                raise EmbeddingError(
                    provider="ollama",
                    message=(
                        f"Cannot connect to Ollama at {self._base_url}. "
                        "Ensure Ollama is running: ollama serve"
                    ),
                ) from e

            except httpx.TimeoutException as e:
                raise httpx.TimeoutException(
                    f"Ollama request timed out after {self._timeout}s"
                ) from e

    # -------------------------------------------------------------------------
    # Connection & Model Management
    # -------------------------------------------------------------------------

    async def validate_connection(self) -> bool:
        """
        Check if Ollama server is running and accessible.

        Returns:
            True if server is reachable, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.warning(
                "Ollama connection validation failed",
                base_url=self._base_url,
                error=str(e),
            )
            return False

    async def list_available_models(self) -> list[str]:
        """
        List embedding models available on Ollama server.

        Returns:
            List of model names that support embeddings

        Raises:
            EmbeddingError: If unable to fetch model list
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                response.raise_for_status()
                data = response.json()

                # Filter for known embedding models
                all_models = [m["name"] for m in data.get("models", [])]
                embedding_models = []

                for model in all_models:
                    # Check if model is a known embedding model
                    base_name = model.split(":")[0]
                    if base_name in OLLAMA_EMBEDDING_MODELS or "embed" in model.lower():
                        embedding_models.append(model)

                logger.debug(
                    "Listed available embedding models",
                    total_models=len(all_models),
                    embedding_models=len(embedding_models),
                )

                return embedding_models

        except httpx.ConnectError as e:
            raise EmbeddingError(
                provider="ollama",
                message=f"Cannot connect to Ollama at {self._base_url}",
            ) from e
        except Exception as e:
            raise EmbeddingError(
                provider="ollama",
                message=f"Failed to list models: {e}",
            ) from e

    async def pull_model(self, model: str) -> bool:
        """
        Pull a model from Ollama registry if not available locally.

        Note: This can take several minutes for large models.

        Args:
            model: Model name to pull (e.g., "nomic-embed-text")

        Returns:
            True if model was pulled successfully

        Raises:
            EmbeddingError: If pull fails
        """
        logger.info("Pulling Ollama model", model=model)

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:  # 10min timeout
                response = await client.post(
                    f"{self._base_url}/api/pull",
                    json={"name": model, "stream": False},
                )
                response.raise_for_status()

                logger.info("Model pulled successfully", model=model)
                return True

        except httpx.TimeoutException:
            raise EmbeddingError(
                provider="ollama",
                message=f"Timeout pulling model '{model}'. Try manually: ollama pull {model}",
            )
        except Exception as e:
            raise EmbeddingError(
                provider="ollama",
                message=f"Failed to pull model '{model}': {e}",
            ) from e

    async def ensure_model_available(self, model: str | None = None) -> bool:
        """
        Ensure a model is available, pulling if necessary.

        Args:
            model: Model to check/pull (uses default if None)

        Returns:
            True if model is available after check/pull
        """
        use_model = model or self._model
        available = await self.list_available_models()

        # Check with and without tag
        base_name = use_model.split(":")[0]
        is_available = any(m == use_model or m.startswith(base_name) for m in available)

        if is_available:
            logger.debug("Model already available", model=use_model)
            return True

        logger.info("Model not found, attempting to pull", model=use_model)
        return await self.pull_model(use_model)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "provider": self.name,
            "model": self._model,
            "base_url": self._base_url,
            "config": self._model_config,
            "capabilities": {
                "dimensions": self.capabilities.dimensions,
                "max_tokens_per_text": self.capabilities.max_tokens_per_text,
                "max_batch_size": self.capabilities.max_batch_size,
                "supports_batching": self.capabilities.supports_batching,
                "supports_truncation": self.capabilities.supports_truncation,
                "normalized": self.capabilities.normalized,
            },
        }

    @staticmethod
    def list_supported_models() -> dict[str, dict[str, Any]]:
        """
        Get list of known Ollama embedding models with their configs.

        Returns:
            Dictionary mapping model names to their configurations
        """
        return OLLAMA_EMBEDDING_MODELS.copy()

    def __repr__(self) -> str:
        return (
            f"OllamaEmbeddingProvider(" f"model={self._model!r}, " f"base_url={self._base_url!r})"
        )


__all__ = [
    "OllamaEmbeddingProvider",
    "OLLAMA_EMBEDDING_MODELS",
    "DEFAULT_OLLAMA_EMBEDDING_MODEL",
]
