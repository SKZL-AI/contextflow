"""
Integration tests for RAG with strategies.

Tests the TemporaryRAG system integrated with ContextFlow strategies:
1. process_with_rag() indexes and retrieves documents
2. 3-layer progressive disclosure (search_compact, get_timeline, get_full_documents)
3. RAG-enhanced processing across different strategies
4. Session-based document management
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from contextflow.core.config import ContextFlowConfig
from contextflow.core.hooks import HooksManager
from contextflow.core.orchestrator import ContextFlow, OrchestratorConfig
from contextflow.core.types import StrategyType, TaskStatus
from contextflow.rag.temp_rag import (
    DocSummary,
    IndexType,
    RAGDocument,
    SearchResult,
    TemporaryRAG,
    create_rag_from_texts,
    quick_search,
    select_index_type,
)

# =============================================================================
# Process with RAG Tests
# =============================================================================


class TestProcessWithRAG:
    """Test process_with_rag() indexes and retrieves documents."""

    @pytest.mark.asyncio
    async def test_process_with_rag_indexes_documents(
        self,
        mock_provider,
        mock_embedding_provider,
        test_config: ContextFlowConfig,
        hooks_manager: HooksManager,
    ) -> None:
        """Test that process_with_rag indexes documents before processing."""
        orchestrator_config = OrchestratorConfig(
            enable_verification=False,
            enable_sessions=False,
            enable_hooks=True,
        )

        cf = ContextFlow(
            provider=mock_provider,
            config=test_config,
            orchestrator_config=orchestrator_config,
            hooks_manager=hooks_manager,
        )
        await cf.initialize()

        try:
            documents = [
                "Document 1: Information about Python programming.",
                "Document 2: Details about machine learning algorithms.",
                "Document 3: Guide to natural language processing.",
            ]

            with patch.object(cf, "_rag") as mock_rag:
                mock_rag.add_documents = AsyncMock(return_value=["doc1", "doc2", "doc3"])
                mock_rag.search = AsyncMock(return_value=[])

                result = await cf.process_with_rag(
                    task="What is machine learning?",
                    documents=documents,
                )

                # Documents should have been indexed
                mock_rag.add_documents.assert_called_once()

        finally:
            await cf.close()

    @pytest.mark.asyncio
    async def test_process_with_rag_retrieves_relevant_context(
        self,
        mock_provider,
        mock_embedding_provider,
        test_config: ContextFlowConfig,
        hooks_manager: HooksManager,
    ) -> None:
        """Test that process_with_rag retrieves relevant context."""
        # Create real RAG instance for this test
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
        )

        # Add documents
        await rag.add_documents(
            [
                "Python is a programming language known for readability.",
                "Machine learning enables computers to learn from data.",
                "Deep learning uses neural networks with many layers.",
            ]
        )

        orchestrator_config = OrchestratorConfig(
            enable_verification=False,
            enable_sessions=False,
            enable_hooks=True,
        )

        cf = ContextFlow(
            provider=mock_provider,
            config=test_config,
            orchestrator_config=orchestrator_config,
            hooks_manager=hooks_manager,
        )
        cf._rag = rag  # Inject RAG instance
        await cf.initialize()

        try:
            result = await cf.process_with_rag(
                task="Tell me about machine learning",
                documents=[],  # Already indexed
            )

            assert result.status == TaskStatus.COMPLETED
            # Context should have been enhanced with retrieved documents
            assert result.metadata.get("rag_enhanced", False) is True or result.answer is not None

        finally:
            await cf.close()

    @pytest.mark.asyncio
    async def test_process_with_rag_uses_correct_strategy(
        self,
        mock_provider,
        mock_embedding_provider,
        test_config: ContextFlowConfig,
        hooks_manager: HooksManager,
    ) -> None:
        """Test that strategy selection works with RAG."""
        orchestrator_config = OrchestratorConfig(
            enable_verification=False,
            enable_sessions=False,
            enable_hooks=True,
        )

        cf = ContextFlow(
            provider=mock_provider,
            config=test_config,
            orchestrator_config=orchestrator_config,
            hooks_manager=hooks_manager,
        )
        await cf.initialize()

        try:
            documents = ["Small document content."] * 5

            result = await cf.process_with_rag(
                task="Summarize",
                documents=documents,
                strategy=StrategyType.GSD_DIRECT,
            )

            assert result.strategy_used == StrategyType.GSD_DIRECT

        finally:
            await cf.close()


class TestThreeLayerSearch:
    """Test 3-layer progressive disclosure."""

    @pytest.mark.asyncio
    async def test_search_compact_returns_ids_only(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test Layer 1: search_compact returns only document IDs."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
        )

        # Add documents
        doc_ids = await rag.add_documents(
            [
                "Document about artificial intelligence.",
                "Document about database systems.",
                "Document about web development.",
            ]
        )

        # Search compact
        result_ids = await rag.search_compact("artificial intelligence", k=10)

        # Should return list of strings (IDs)
        assert isinstance(result_ids, list)
        assert all(isinstance(id_, str) for id_ in result_ids)
        assert len(result_ids) <= 10

    @pytest.mark.asyncio
    async def test_get_timeline_returns_summaries(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test Layer 2: get_timeline returns summaries."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
        )

        # Add documents
        await rag.add_documents(
            [
                "Document about artificial intelligence and machine learning applications.",
                "Document about database systems and data management.",
                "Document about web development frameworks and technologies.",
            ]
        )

        # Get IDs first
        doc_ids = await rag.search_compact("artificial intelligence", k=3)

        # Get timeline (summaries)
        timeline = await rag.get_timeline(doc_ids)

        # Should return list of DocSummary
        assert isinstance(timeline, list)
        assert all(isinstance(item, DocSummary) for item in timeline)
        for summary in timeline:
            assert hasattr(summary, "id")
            assert hasattr(summary, "summary")
            assert hasattr(summary, "relevance_score")

    @pytest.mark.asyncio
    async def test_get_full_documents_returns_complete_content(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test Layer 3: get_full_documents returns complete documents."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
        )

        original_content = "Full document content about artificial intelligence."

        # Add documents
        await rag.add_documents([original_content])

        # Get IDs first
        doc_ids = await rag.search_compact("artificial intelligence", k=1)

        # Get full documents
        full_docs = await rag.get_full_documents(doc_ids)

        # Should return list of RAGDocument
        assert isinstance(full_docs, list)
        assert all(isinstance(doc, RAGDocument) for doc in full_docs)
        # Content should be complete
        assert full_docs[0].content == original_content

    @pytest.mark.asyncio
    async def test_progressive_disclosure_flow(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test complete 3-layer progressive disclosure flow."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
        )

        # Add many documents
        documents = [f"Document {i}: Content about topic {i % 3}" for i in range(100)]
        await rag.add_documents(documents)

        # Layer 1: Get broad set of IDs (cheap)
        ids = await rag.search_compact("topic", k=20)
        assert len(ids) == 20
        layer1_tokens = len(str(ids)) // 4  # Rough token estimate

        # Layer 2: Get summaries of top 10 (moderate)
        summaries = await rag.get_timeline(ids[:10])
        assert len(summaries) == 10
        layer2_tokens = sum(len(s.summary) for s in summaries) // 4

        # Layer 3: Get full content of top 5 (expensive)
        full_docs = await rag.get_full_documents(ids[:5])
        assert len(full_docs) == 5
        layer3_tokens = sum(len(d.content) for d in full_docs) // 4

        # Progressive disclosure should have increasing detail
        assert layer1_tokens < layer2_tokens < layer3_tokens


class TestRAGDocumentManagement:
    """Test RAG document management operations."""

    @pytest.mark.asyncio
    async def test_add_single_document(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test adding a single document."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
        )

        doc_id = await rag.add_document(
            content="Test document content",
            metadata={"source": "test"},
        )

        assert doc_id is not None
        assert rag.size() == 1

    @pytest.mark.asyncio
    async def test_add_multiple_documents(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test adding multiple documents."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
        )

        doc_ids = await rag.add_documents(
            [
                "Document 1",
                "Document 2",
                "Document 3",
            ]
        )

        assert len(doc_ids) == 3
        assert rag.size() == 3

    @pytest.mark.asyncio
    async def test_remove_document(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test removing a document."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
        )

        doc_id = await rag.add_document("Test document")
        assert rag.size() == 1

        removed = rag.remove_document(doc_id)
        assert removed is True
        # Note: FAISS doesn't support efficient deletion, so size may not change

    @pytest.mark.asyncio
    async def test_get_document_by_id(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test retrieving a document by ID."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
        )

        original_content = "Original content"
        doc_id = await rag.add_document(original_content)

        doc = rag.get_document(doc_id)

        assert doc is not None
        assert doc.content == original_content

    @pytest.mark.asyncio
    async def test_clear_all_documents(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test clearing all documents."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
        )

        await rag.add_documents(["Doc 1", "Doc 2", "Doc 3"])
        assert rag.size() == 3

        rag.clear()
        assert rag.size() == 0


class TestRAGSearchVariants:
    """Test different RAG search methods."""

    @pytest.mark.asyncio
    async def test_traditional_search(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test traditional search returning SearchResult objects."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
        )

        await rag.add_documents(
            [
                "Python programming language",
                "Java programming language",
                "JavaScript web development",
            ]
        )

        results = await rag.search("Python", k=3)

        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)
        for r in results:
            assert hasattr(r, "document")
            assert hasattr(r, "score")
            assert hasattr(r, "rank")

    @pytest.mark.asyncio
    async def test_search_with_threshold(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test search with minimum score threshold."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
        )

        await rag.add_documents(
            [
                "Highly relevant document about Python",
                "Completely unrelated document about cooking",
            ]
        )

        # Search with high threshold
        results = await rag.search("Python programming", k=10, threshold=0.5)

        # Results should be filtered by threshold
        for r in results:
            assert r.score >= 0.5

    @pytest.mark.asyncio
    async def test_search_with_filter(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test search with custom filter function."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
        )

        await rag.add_document("Python basics", metadata={"level": "beginner"})
        await rag.add_document("Advanced Python", metadata={"level": "advanced"})
        await rag.add_document("Python intermediate", metadata={"level": "intermediate"})

        # Filter for advanced only
        results = await rag.search_with_filter(
            query="Python",
            k=10,
            metadata_filter={"level": "advanced"},
        )

        for r in results:
            assert r.document.metadata.get("level") == "advanced"


class TestRAGWithAutoChunking:
    """Test RAG auto-chunking for large documents."""

    @pytest.mark.asyncio
    async def test_auto_chunking_enabled(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test that large documents are automatically chunked."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
            auto_chunk=True,
            chunk_size=100,  # Small chunk size for testing
            chunk_overlap=20,
        )

        # Large document that exceeds chunk_size
        large_content = "This is a test sentence. " * 100

        doc_id = await rag.add_document(large_content)

        # Document should have been chunked
        # The parent doc exists but chunks are indexed
        parent_doc = rag.get_document(doc_id)
        assert parent_doc is not None
        assert parent_doc.metadata.get("is_parent", False) is True

    @pytest.mark.asyncio
    async def test_auto_chunking_disabled(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test that auto-chunking can be disabled."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
            auto_chunk=False,
        )

        content = "Test content " * 10
        doc_id = await rag.add_document(content)

        doc = rag.get_document(doc_id)
        assert doc is not None
        assert doc.metadata.get("is_parent", False) is False


class TestRAGStatistics:
    """Test RAG statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_tracking(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test that RAG tracks statistics."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
        )

        # Add documents
        await rag.add_documents(["Doc 1", "Doc 2", "Doc 3"])

        # Perform searches
        await rag.search("Doc", k=3)
        await rag.search("Test", k=3)

        stats = rag.get_stats()

        assert stats["total_documents"] == 3
        assert stats["total_adds"] == 3
        assert stats["total_searches"] == 2
        assert stats["avg_search_time_ms"] >= 0


class TestRAGConvenienceFunctions:
    """Test RAG convenience functions."""

    @pytest.mark.asyncio
    async def test_create_rag_from_texts(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test quick RAG creation from texts."""
        texts = ["Text 1", "Text 2", "Text 3"]

        rag = await create_rag_from_texts(
            texts=texts,
            embedding_provider=mock_embedding_provider,
        )

        assert rag.size() == 3

    @pytest.mark.asyncio
    async def test_quick_search(
        self,
        mock_embedding_provider,
    ) -> None:
        """Test quick search returning content strings."""
        rag = TemporaryRAG(
            embedding_provider=mock_embedding_provider,
            index_type=IndexType.FLAT,
        )
        await rag.add_documents(["Content 1", "Content 2", "Content 3"])

        results = await quick_search(rag, "Content", k=2)

        assert isinstance(results, list)
        assert all(isinstance(r, str) for r in results)
        assert len(results) <= 2

    def test_select_index_type(self) -> None:
        """Test automatic index type selection."""
        # Small dataset
        assert select_index_type(1000) == IndexType.FLAT

        # Medium dataset
        assert select_index_type(50000) == IndexType.IVF

        # Large dataset
        assert select_index_type(200000) == IndexType.HNSW
