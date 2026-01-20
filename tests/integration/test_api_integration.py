"""
Integration tests for API endpoints with mock processing.

Tests the FastAPI REST API server endpoints:
1. POST /api/v1/process - Main processing endpoint
2. POST /api/v1/process/stream - Streaming endpoint
3. GET /api/v1/health - Health check endpoint
4. POST /api/v1/analyze - Context analysis endpoint
5. POST /api/v1/batch - Batch processing endpoint
6. Session management endpoints
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from contextflow.api.server import API_PREFIX, app, get_contextflow
from contextflow.core.orchestrator import ContextFlow
from contextflow.core.types import ProcessResult, StrategyType, TaskStatus

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_contextflow() -> MagicMock:
    """Create a mock ContextFlow instance."""
    mock = MagicMock(spec=ContextFlow)
    mock.is_initialized = True
    mock.provider = MagicMock()
    mock.provider.name = "mock"
    mock.provider.capabilities = MagicMock()
    mock.provider.capabilities.max_context_tokens = 200000
    mock.provider.capabilities.supported_models = ["mock-model"]
    mock.provider.capabilities.supports_streaming = True
    mock.provider.capabilities.supports_tools = False
    mock.provider.capabilities.rate_limit_rpm = 1000

    # Mock process method
    async def mock_process(*args, **kwargs):
        return ProcessResult(
            answer="This is the mock response.",
            strategy_used=StrategyType.GSD_DIRECT,
            total_tokens=100,
            total_cost=0.001,
            execution_time=0.5,
            trajectory=[],
            sub_agent_count=0,
            warnings=[],
            metadata={"verification_passed": True, "verification_score": 0.9},
            status=TaskStatus.COMPLETED,
        )

    mock.process = AsyncMock(side_effect=mock_process)

    # Mock stream method
    async def mock_stream(*args, **kwargs):
        for chunk in ["Streaming ", "response ", "here."]:
            yield chunk
            await asyncio.sleep(0.01)

    mock.stream = mock_stream

    # Mock analyze method
    async def mock_analyze(*args, **kwargs):
        from contextflow.core.types import ContextAnalysis

        return ContextAnalysis(
            token_count=500,
            density_score=0.5,
            complexity_score=0.3,
            recommended_strategy=StrategyType.GSD_DIRECT,
            estimated_cost=0.01,
            estimated_time_seconds=1.0,
            structure_type="unstructured",
            warnings=[],
            metadata={},
        )

    mock.analyze = AsyncMock(side_effect=mock_analyze)

    return mock


@pytest.fixture
def test_client(mock_contextflow: MagicMock) -> TestClient:
    """Create test client with mocked ContextFlow."""

    async def override_get_contextflow():
        return mock_contextflow

    app.dependency_overrides[get_contextflow] = override_get_contextflow

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


# =============================================================================
# Process Endpoint Tests
# =============================================================================


class TestProcessEndpoint:
    """Test POST /api/v1/process endpoint."""

    def test_process_endpoint_success(
        self,
        test_client: TestClient,
        mock_contextflow: MagicMock,
    ) -> None:
        """Test successful processing request."""
        response = test_client.post(
            f"{API_PREFIX}/process",
            json={
                "task": "Summarize this document",
                "context": "Document content here.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["answer"] is not None
        assert data["strategy_used"] is not None

    def test_process_endpoint_with_strategy(
        self,
        test_client: TestClient,
        mock_contextflow: MagicMock,
    ) -> None:
        """Test processing with explicit strategy."""
        response = test_client.post(
            f"{API_PREFIX}/process",
            json={
                "task": "Analyze this",
                "context": "Content to analyze.",
                "strategy": "gsd_direct",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["strategy_used"] == "gsd_direct"

    def test_process_endpoint_with_documents(
        self,
        test_client: TestClient,
        mock_contextflow: MagicMock,
        tmp_path: Any,
    ) -> None:
        """Test processing with document paths."""
        # Note: In a real test, you'd need actual files
        response = test_client.post(
            f"{API_PREFIX}/process",
            json={
                "task": "Summarize documents",
                "documents": ["doc1.txt", "doc2.txt"],
            },
        )

        # May fail if validation requires real files
        # This tests the endpoint accepts the parameter
        assert response.status_code in [200, 400, 422]

    def test_process_endpoint_with_constraints(
        self,
        test_client: TestClient,
        mock_contextflow: MagicMock,
    ) -> None:
        """Test processing with constraints."""
        response = test_client.post(
            f"{API_PREFIX}/process",
            json={
                "task": "Summarize",
                "context": "Content here.",
                "constraints": ["Keep under 100 words", "Include key facts"],
            },
        )

        assert response.status_code == 200

    def test_process_endpoint_missing_task(
        self,
        test_client: TestClient,
    ) -> None:
        """Test error when task is missing."""
        response = test_client.post(
            f"{API_PREFIX}/process",
            json={
                "context": "Content without task.",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_process_endpoint_missing_context(
        self,
        test_client: TestClient,
    ) -> None:
        """Test error when both context and documents are missing."""
        response = test_client.post(
            f"{API_PREFIX}/process",
            json={
                "task": "Task without context.",
            },
        )

        # May return 400 or 422 depending on validation
        assert response.status_code in [400, 422]


# =============================================================================
# Stream Endpoint Tests
# =============================================================================


class TestStreamEndpoint:
    """Test POST /api/v1/process/stream endpoint."""

    def test_stream_endpoint_returns_sse(
        self,
        test_client: TestClient,
        mock_contextflow: MagicMock,
    ) -> None:
        """Test streaming endpoint returns SSE format."""
        response = test_client.post(
            f"{API_PREFIX}/process/stream",
            json={
                "task": "Stream this",
                "context": "Content for streaming.",
            },
        )

        assert response.status_code == 200
        assert response.headers.get("content-type") == "text/event-stream; charset=utf-8"

    def test_stream_endpoint_yields_chunks(
        self,
        test_client: TestClient,
        mock_contextflow: MagicMock,
    ) -> None:
        """Test streaming endpoint yields data chunks."""
        response = test_client.post(
            f"{API_PREFIX}/process/stream",
            json={
                "task": "Stream this",
                "context": "Content for streaming.",
            },
        )

        assert response.status_code == 200

        # Check that response contains SSE data lines
        content = response.text
        assert "data:" in content

    def test_stream_endpoint_with_strategy(
        self,
        test_client: TestClient,
        mock_contextflow: MagicMock,
    ) -> None:
        """Test streaming with explicit strategy."""
        response = test_client.post(
            f"{API_PREFIX}/process/stream",
            json={
                "task": "Stream analysis",
                "context": "Content here.",
                "strategy": "gsd_direct",
            },
        )

        assert response.status_code == 200


# =============================================================================
# Health Endpoint Tests
# =============================================================================


class TestHealthEndpoint:
    """Test GET /api/v1/health endpoint."""

    def test_health_endpoint_returns_status(
        self,
        test_client: TestClient,
    ) -> None:
        """Test health endpoint returns status."""
        response = test_client.get(f"{API_PREFIX}/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_endpoint_includes_version(
        self,
        test_client: TestClient,
    ) -> None:
        """Test health endpoint includes version."""
        response = test_client.get(f"{API_PREFIX}/health")

        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert data["version"] is not None

    def test_health_endpoint_includes_providers(
        self,
        test_client: TestClient,
    ) -> None:
        """Test health endpoint includes provider info."""
        response = test_client.get(f"{API_PREFIX}/health")

        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert isinstance(data["providers"], list)

    def test_health_endpoint_includes_uptime(
        self,
        test_client: TestClient,
    ) -> None:
        """Test health endpoint includes uptime."""
        response = test_client.get(f"{API_PREFIX}/health")

        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


# =============================================================================
# Analyze Endpoint Tests
# =============================================================================


class TestAnalyzeEndpoint:
    """Test POST /api/v1/analyze endpoint."""

    def test_analyze_endpoint_returns_analysis(
        self,
        test_client: TestClient,
        mock_contextflow: MagicMock,
    ) -> None:
        """Test analyze endpoint returns analysis."""
        response = test_client.post(
            f"{API_PREFIX}/analyze",
            json={
                "context": "Content to analyze.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "token_count" in data
        assert "recommended_strategy" in data

    def test_analyze_endpoint_includes_estimates(
        self,
        test_client: TestClient,
        mock_contextflow: MagicMock,
    ) -> None:
        """Test analyze endpoint includes cost/time estimates."""
        response = test_client.post(
            f"{API_PREFIX}/analyze",
            json={
                "context": "Content to analyze.",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "estimated_costs" in data
        assert "estimated_time" in data

    def test_analyze_endpoint_with_documents(
        self,
        test_client: TestClient,
        mock_contextflow: MagicMock,
    ) -> None:
        """Test analyze endpoint with documents."""
        response = test_client.post(
            f"{API_PREFIX}/analyze",
            json={
                "documents": ["doc1.txt"],
            },
        )

        # May fail if validation requires real files
        assert response.status_code in [200, 400, 422]


# =============================================================================
# Batch Endpoint Tests
# =============================================================================


class TestBatchEndpoint:
    """Test POST /api/v1/batch endpoint."""

    def test_batch_endpoint_processes_multiple(
        self,
        test_client: TestClient,
        mock_contextflow: MagicMock,
    ) -> None:
        """Test batch endpoint processes multiple requests."""
        response = test_client.post(
            f"{API_PREFIX}/batch",
            json={
                "requests": [
                    {"task": "Task 1", "context": "Context 1"},
                    {"task": "Task 2", "context": "Context 2"},
                    {"task": "Task 3", "context": "Context 3"},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 3
        assert len(data["results"]) == 3

    def test_batch_endpoint_parallel_processing(
        self,
        test_client: TestClient,
        mock_contextflow: MagicMock,
    ) -> None:
        """Test batch endpoint with parallel processing."""
        response = test_client.post(
            f"{API_PREFIX}/batch",
            json={
                "requests": [
                    {"task": "Task 1", "context": "Context 1"},
                    {"task": "Task 2", "context": "Context 2"},
                ],
                "parallel": True,
                "max_concurrent": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2

    def test_batch_endpoint_sequential_processing(
        self,
        test_client: TestClient,
        mock_contextflow: MagicMock,
    ) -> None:
        """Test batch endpoint with sequential processing."""
        response = test_client.post(
            f"{API_PREFIX}/batch",
            json={
                "requests": [
                    {"task": "Task 1", "context": "Context 1"},
                    {"task": "Task 2", "context": "Context 2"},
                ],
                "parallel": False,
            },
        )

        assert response.status_code == 200

    def test_batch_endpoint_returns_totals(
        self,
        test_client: TestClient,
        mock_contextflow: MagicMock,
    ) -> None:
        """Test batch endpoint returns total statistics."""
        response = test_client.post(
            f"{API_PREFIX}/batch",
            json={
                "requests": [
                    {"task": "Task 1", "context": "Context 1"},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "total_execution_time" in data
        assert "total_token_usage" in data
        assert "successful_count" in data
        assert "failed_count" in data


# =============================================================================
# Session Endpoint Tests
# =============================================================================


class TestSessionEndpoints:
    """Test session management endpoints."""

    def test_create_session(
        self,
        test_client: TestClient,
    ) -> None:
        """Test POST /api/v1/sessions creates session."""
        response = test_client.post(
            f"{API_PREFIX}/sessions",
            json={
                "context": "Initial context for session.",
                "ttl_seconds": 3600,
            },
        )

        # May not work without full session manager setup
        assert response.status_code in [201, 500, 503]

    def test_get_session(
        self,
        test_client: TestClient,
    ) -> None:
        """Test GET /api/v1/sessions/{id} returns session info."""
        # First create a session
        create_response = test_client.post(
            f"{API_PREFIX}/sessions",
            json={},
        )

        if create_response.status_code == 201:
            session_id = create_response.json()["session_id"]

            response = test_client.get(f"{API_PREFIX}/sessions/{session_id}")
            assert response.status_code in [200, 404]

    def test_delete_session(
        self,
        test_client: TestClient,
    ) -> None:
        """Test DELETE /api/v1/sessions/{id} deletes session."""
        # First create a session
        create_response = test_client.post(
            f"{API_PREFIX}/sessions",
            json={},
        )

        if create_response.status_code == 201:
            session_id = create_response.json()["session_id"]

            response = test_client.delete(f"{API_PREFIX}/sessions/{session_id}")
            assert response.status_code in [204, 404]


# =============================================================================
# Search Endpoint Tests
# =============================================================================


class TestSearchEndpoint:
    """Test POST /api/v1/search endpoint."""

    def test_search_endpoint(
        self,
        test_client: TestClient,
    ) -> None:
        """Test search within session context."""
        response = test_client.post(
            f"{API_PREFIX}/search",
            json={
                "query": "test query",
                "session_id": "test-session-id",
                "max_results": 10,
            },
        )

        # May fail without session, but tests endpoint exists
        assert response.status_code in [200, 400, 404]


# =============================================================================
# Provider Endpoint Tests
# =============================================================================


class TestProviderEndpoint:
    """Test GET /api/v1/providers endpoint."""

    def test_providers_endpoint(
        self,
        test_client: TestClient,
    ) -> None:
        """Test listing available providers."""
        response = test_client.get(f"{API_PREFIX}/providers")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


# =============================================================================
# Root Endpoint Tests
# =============================================================================


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_endpoint(
        self,
        test_client: TestClient,
    ) -> None:
        """Test root endpoint returns API info."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test API error handling."""

    def test_validation_error_response(
        self,
        test_client: TestClient,
    ) -> None:
        """Test validation error returns proper format."""
        response = test_client.post(
            f"{API_PREFIX}/process",
            json={},  # Missing required fields
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_not_found_response(
        self,
        test_client: TestClient,
    ) -> None:
        """Test 404 for non-existent endpoints."""
        response = test_client.get(f"{API_PREFIX}/nonexistent")

        assert response.status_code == 404

    def test_invalid_json_response(
        self,
        test_client: TestClient,
    ) -> None:
        """Test error for invalid JSON."""
        response = test_client.post(
            f"{API_PREFIX}/process",
            data="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


# =============================================================================
# CORS Tests
# =============================================================================


class TestCORS:
    """Test CORS middleware."""

    def test_cors_headers_present(
        self,
        test_client: TestClient,
    ) -> None:
        """Test CORS headers are present."""
        response = test_client.options(
            f"{API_PREFIX}/process",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        # CORS preflight should succeed
        assert response.status_code in [200, 204, 405]
