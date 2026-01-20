"""
Embedding provider implementations for ContextFlow RAG.

Provides multiple embedding backends:
- openai: OpenAI text-embedding-3-small/large, ada-002
- sentence_transformers: Local embeddings, no API costs
- ollama: Local via Ollama server (nomic-embed-text, mxbai-embed-large, etc.)
"""

from contextflow.rag.embeddings.base import (
    BaseEmbeddingProvider,
    EmbeddingCapabilities,
    EmbeddingProviderType,
    EmbeddingResult,
)
from contextflow.rag.embeddings.ollama_embed import (
    DEFAULT_OLLAMA_EMBEDDING_MODEL,
    OLLAMA_EMBEDDING_MODELS,
    OllamaEmbeddingProvider,
)
from contextflow.rag.embeddings.openai_embed import (
    DEFAULT_MODEL as DEFAULT_OPENAI_MODEL,
)
from contextflow.rag.embeddings.openai_embed import (
    OPENAI_EMBEDDING_MODELS,
    OpenAIEmbeddingProvider,
    create_openai_embedding_provider,
)
from contextflow.rag.embeddings.sentence_transformers_embed import (
    DEFAULT_MODEL,
    SENTENCE_TRANSFORMER_MODELS,
    SentenceTransformersProvider,
    create_sentence_transformers_provider,
)

__all__ = [
    # Base classes
    "BaseEmbeddingProvider",
    "EmbeddingCapabilities",
    "EmbeddingProviderType",
    "EmbeddingResult",
    # OpenAI
    "OpenAIEmbeddingProvider",
    "OPENAI_EMBEDDING_MODELS",
    "DEFAULT_OPENAI_MODEL",
    "create_openai_embedding_provider",
    # Sentence Transformers
    "SentenceTransformersProvider",
    "SENTENCE_TRANSFORMER_MODELS",
    "DEFAULT_MODEL",
    "create_sentence_transformers_provider",
    # Ollama
    "OllamaEmbeddingProvider",
    "OLLAMA_EMBEDDING_MODELS",
    "DEFAULT_OLLAMA_EMBEDDING_MODEL",
]
