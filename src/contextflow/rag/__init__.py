"""RAG and embedding utilities for ContextFlow.

This module provides chunking, embedding, and vector search functionality:
- Text chunking with multiple strategies (fixed, semantic, sliding, smart)
- Embedding generation via multiple providers
- In-Memory FAISS-based RAG with 3-Layer Progressive Disclosure

3-Layer Search (Claude-Mem Pattern):
- Layer 1: search_compact() -> List[DocID] (~50-100 tokens)
- Layer 2: get_timeline() -> List[DocSummary] (~200-500 tokens)
- Layer 3: get_full_documents() -> List[Document] (~500-2000 tokens)
"""

from contextflow.rag.chunker import (
    Chunk,
    ChunkingResult,
    ChunkingStrategy,
    SmartChunker,
    chunk_for_embedding,
    chunk_text,
)
from contextflow.rag.embeddings import (
    BaseEmbeddingProvider,
    EmbeddingCapabilities,
    EmbeddingProviderType,
    EmbeddingResult,
)
from contextflow.rag.temp_rag import (
    DocSummary,
    IndexType,
    RAGDocument,
    SearchMode,
    SearchResult,
    TemporaryRAG,
    create_rag_from_texts,
    quick_search,
    select_index_type,
)

__all__: list[str] = [
    # Chunking
    "SmartChunker",
    "Chunk",
    "ChunkingResult",
    "ChunkingStrategy",
    "chunk_text",
    "chunk_for_embedding",
    # Embeddings
    "BaseEmbeddingProvider",
    "EmbeddingCapabilities",
    "EmbeddingProviderType",
    "EmbeddingResult",
    # TemporaryRAG
    "TemporaryRAG",
    "RAGDocument",
    "DocSummary",
    "SearchResult",
    "IndexType",
    "SearchMode",
    "create_rag_from_texts",
    "quick_search",
    "select_index_type",
]
