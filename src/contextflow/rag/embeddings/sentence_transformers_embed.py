"""
Sentence Transformers Embedding Provider for ContextFlow.

Local embeddings without API costs. Supports:
- all-MiniLM-L6-v2 (384 dims, fast)
- all-mpnet-base-v2 (768 dims, balanced)
- multi-qa-mpnet-base-dot-v1 (768 dims, Q&A optimized)
- paraphrase-multilingual-MiniLM-L12-v2 (384 dims, multilingual)
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from contextflow.rag.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingCapabilities,
    EmbeddingProviderType,
    EmbeddingResult,
)
from contextflow.utils.errors import EmbeddingError
from contextflow.utils.logging import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

# =============================================================================
# Model Configurations
# =============================================================================

SENTENCE_TRANSFORMER_MODELS: dict[str, dict[str, Any]] = {
    "all-MiniLM-L6-v2": {
        "dimensions": 384,
        "max_tokens": 256,
        "speed": "fast",
        "quality": "good",
        "description": "Fast and lightweight, good for general use",
    },
    "all-mpnet-base-v2": {
        "dimensions": 768,
        "max_tokens": 384,
        "speed": "medium",
        "quality": "excellent",
        "description": "Best quality for English semantic search",
    },
    "multi-qa-mpnet-base-dot-v1": {
        "dimensions": 768,
        "max_tokens": 512,
        "speed": "medium",
        "quality": "excellent",
        "optimized_for": "qa",
        "description": "Optimized for question-answering tasks",
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "dimensions": 384,
        "max_tokens": 128,
        "speed": "fast",
        "quality": "good",
        "multilingual": True,
        "description": "Supports 50+ languages",
    },
    "all-distilroberta-v1": {
        "dimensions": 768,
        "max_tokens": 512,
        "speed": "medium",
        "quality": "good",
        "description": "Based on DistilRoBERTa, good balance",
    },
    "msmarco-distilbert-base-v4": {
        "dimensions": 768,
        "max_tokens": 512,
        "speed": "medium",
        "quality": "excellent",
        "optimized_for": "search",
        "description": "Trained on MS MARCO, great for search",
    },
}

# Default model for quick start
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Default batch size for embedding
DEFAULT_BATCH_SIZE = 32

# Thread pool for running sync operations
_thread_pool: ThreadPoolExecutor | None = None


def _get_thread_pool() -> ThreadPoolExecutor:
    """Get or create the shared thread pool."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="st_embed")
    return _thread_pool


@lru_cache(maxsize=4)
def _get_cached_model(
    model_name: str,
    device: str,
    cache_folder: str | None = None,
) -> SentenceTransformer:
    """
    Load and cache a SentenceTransformer model.

    Uses LRU cache to avoid reloading models.

    Args:
        model_name: HuggingFace model name
        device: Device to load model on
        cache_folder: Optional cache directory

    Returns:
        Loaded SentenceTransformer model
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for local embeddings. "
            "Install with: pip install sentence-transformers"
        ) from e

    logger.info(
        "Loading SentenceTransformer model",
        model=model_name,
        device=device,
    )

    model = SentenceTransformer(
        model_name,
        device=device,
        cache_folder=cache_folder,
    )

    logger.info(
        "Model loaded successfully",
        model=model_name,
        device=str(model.device),
    )

    return model


def _detect_device() -> str:
    """
    Auto-detect the best available device.

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        # Handle ImportError, DLL loading errors, etc.
        pass

    return "cpu"


# =============================================================================
# SentenceTransformersProvider
# =============================================================================


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """
    Sentence Transformers Provider for local embeddings.

    No API costs, runs locally on CPU or GPU. Supports a variety of
    pre-trained models optimized for different use cases.

    Features:
        - Lazy model loading (only loads when first used)
        - GPU support (CUDA, MPS) with auto-detection
        - Thread pool execution for async compatibility
        - LRU caching of loaded models
        - Configurable batch sizes

    Usage:
        # Basic usage with default model
        provider = SentenceTransformersProvider()

        # Use specific model
        provider = SentenceTransformersProvider(model="all-mpnet-base-v2")

        # Force CPU usage
        provider = SentenceTransformersProvider(device="cpu")

        # Embed documents
        result = await provider.embed(["doc1", "doc2"])
        print(result.vectors.shape)  # (2, 384)

        # Embed query
        query_vec = await provider.embed_query("search query")
        print(query_vec.shape)  # (384,)

    Available Models:
        - all-MiniLM-L6-v2: Fast, 384 dims, good general purpose
        - all-mpnet-base-v2: High quality, 768 dims, English
        - multi-qa-mpnet-base-dot-v1: Q&A optimized, 768 dims
        - paraphrase-multilingual-MiniLM-L12-v2: Multilingual, 384 dims
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        normalize: bool = True,
        cache_folder: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize Sentence Transformers Provider.

        Args:
            model: Model name from HuggingFace or local path
            device: Device to run on ("cpu", "cuda", "mps", or None for auto)
            normalize: Whether to L2-normalize embeddings
            cache_folder: Where to cache downloaded models
            batch_size: Default batch size for encoding
        """
        self._model_name = model
        self._device = device or _detect_device()
        self._normalize = normalize
        self._cache_folder = cache_folder
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

        # Get model config if known
        self._model_config = SENTENCE_TRANSFORMER_MODELS.get(model, {})

        logger.debug(
            "SentenceTransformersProvider initialized",
            model=model,
            device=self._device,
            normalize=normalize,
        )

    def _load_model(self) -> SentenceTransformer:
        """
        Lazy-load the model on first use.

        Returns:
            Loaded SentenceTransformer model

        Raises:
            EmbeddingError: If model loading fails
        """
        if self._model is None:
            try:
                self._model = _get_cached_model(
                    self._model_name,
                    self._device,
                    self._cache_folder,
                )
            except ImportError:
                raise
            except Exception as e:
                raise EmbeddingError(
                    provider="sentence_transformers",
                    message=f"Failed to load model '{self._model_name}': {e}",
                    details={"model": self._model_name, "device": self._device},
                ) from e

        return self._model

    @property
    def name(self) -> str:
        """Get the provider name."""
        return "sentence_transformers"

    @property
    def provider_type(self) -> EmbeddingProviderType:
        """Get the provider type enum."""
        return EmbeddingProviderType.SENTENCE_TRANSFORMERS

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        """Get provider capabilities based on model config."""
        dimensions = self._model_config.get("dimensions", 384)
        max_tokens = self._model_config.get("max_tokens", 256)

        return EmbeddingCapabilities(
            max_tokens_per_text=max_tokens,
            max_batch_size=512,
            dimensions=dimensions,
            supports_batching=True,
            supports_truncation=True,
            normalized=self._normalize,
        )

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model_name

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        batch_size: int | None = None,
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.

        Runs the actual encoding in a thread pool to avoid blocking
        the async event loop.

        Args:
            texts: List of texts to embed
            model: Optional model override (switches model if different)
            batch_size: Optional batch size override

        Returns:
            EmbeddingResult with embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If texts list is empty
        """
        # Validate texts
        if not texts:
            raise ValueError("texts list cannot be empty")

        cleaned_texts = self.validate_texts(texts)

        # Use specified model or default
        effective_model = model or self._model_name
        effective_batch_size = batch_size or self._batch_size

        # Switch model if different
        if effective_model != self._model_name:
            self._model_name = effective_model
            self._model_config = SENTENCE_TRANSFORMER_MODELS.get(effective_model, {})
            self._model = None  # Force reload

        start_time = time.perf_counter()

        try:
            # Run synchronous embedding in thread pool
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(
                _get_thread_pool(),
                self._run_sync_embed,
                cleaned_texts,
                effective_batch_size,
            )

            return self._create_result(
                vectors=vectors,
                model=effective_model,
                token_count=self._estimate_tokens(cleaned_texts),
                start_time=start_time,
                metadata={
                    "device": self._device,
                    "normalized": self._normalize,
                    "batch_size": effective_batch_size,
                },
            )

        except Exception as e:
            logger.error(
                "Embedding failed",
                error=str(e),
                model=effective_model,
                text_count=len(texts),
            )
            raise EmbeddingError(
                provider="sentence_transformers",
                message=f"Embedding failed: {e}",
                details={
                    "model": effective_model,
                    "text_count": len(texts),
                },
            ) from e

    async def embed_query(
        self,
        query: str,
        model: str | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Embed a single query string.

        This is a convenience method that calls embed() and returns
        just the embedding vector.

        Args:
            query: Query text to embed
            model: Optional model override

        Returns:
            1D numpy array with query embedding

        Raises:
            EmbeddingError: If embedding generation fails
        """
        result = await self.embed([query], model=model)
        return result.vectors[0]

    def get_dimensions(self, model: str | None = None) -> int:
        """
        Get the embedding dimensions for a model.

        Args:
            model: Model name (uses current model if None)

        Returns:
            Number of dimensions in embedding vectors
        """
        model_name = model or self._model_name
        config = SENTENCE_TRANSFORMER_MODELS.get(model_name, {})

        if "dimensions" in config:
            return config["dimensions"]

        # If unknown model and already loaded, get from model
        if self._model is not None and model_name == self._model_name:
            return self._model.get_sentence_embedding_dimension()

        # Default fallback
        return 384

    async def validate_connection(self) -> bool:
        """
        Validate that the model can be loaded.

        Returns:
            True if model loads successfully, False otherwise
        """
        try:
            # Try to load the model
            self._load_model()
            return True
        except Exception as e:
            logger.warning(
                "Model validation failed",
                model=self._model_name,
                error=str(e),
            )
            return False

    def _run_sync_embed(
        self,
        texts: list[str],
        batch_size: int,
    ) -> npt.NDArray[np.float32]:
        """
        Synchronous embedding (run in thread pool).

        This method does the actual work of encoding texts using
        the SentenceTransformer model.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding

        Returns:
            Numpy array of embeddings
        """
        model = self._load_model()

        # Encode with batching
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
        )

        # Ensure float32
        return np.asarray(embeddings, dtype=np.float32)

    def _estimate_tokens(self, texts: list[str]) -> int:
        """
        Estimate token count for texts.

        Uses a simple heuristic of ~4 characters per token.

        Args:
            texts: List of texts

        Returns:
            Estimated total token count
        """
        total_chars = sum(len(t) for t in texts)
        return total_chars // 4

    def encode_sync(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> npt.NDArray[np.float32]:
        """
        Synchronous embedding method for non-async contexts.

        Args:
            texts: List of texts to embed
            batch_size: Optional batch size override

        Returns:
            Numpy array of embeddings
        """
        effective_batch_size = batch_size or self._batch_size
        return self._run_sync_embed(texts, effective_batch_size)

    @classmethod
    def list_models(cls) -> dict[str, dict[str, Any]]:
        """
        List all supported models with their configurations.

        Returns:
            Dictionary of model names to their configurations
        """
        return SENTENCE_TRANSFORMER_MODELS.copy()

    @classmethod
    def get_model_info(cls, model_name: str) -> dict[str, Any] | None:
        """
        Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model configuration dict or None if unknown
        """
        return SENTENCE_TRANSFORMER_MODELS.get(model_name)


# =============================================================================
# Factory Function
# =============================================================================


def create_sentence_transformers_provider(
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    **kwargs: Any,
) -> SentenceTransformersProvider:
    """
    Factory function to create a SentenceTransformersProvider.

    Args:
        model: Model name
        device: Device to use
        **kwargs: Additional provider arguments

    Returns:
        Configured SentenceTransformersProvider instance
    """
    return SentenceTransformersProvider(
        model=model,
        device=device,
        **kwargs,
    )


__all__ = [
    "SentenceTransformersProvider",
    "SENTENCE_TRANSFORMER_MODELS",
    "DEFAULT_MODEL",
    "create_sentence_transformers_provider",
]
