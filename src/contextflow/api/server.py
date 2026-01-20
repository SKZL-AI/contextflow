"""
FastAPI REST API Server for ContextFlow.

This module provides the main REST API server with full streaming support,
session management, batch processing, and health monitoring.

Features:
- Process endpoints with streaming (SSE) support
- Context analysis without execution
- Batch processing with parallel execution
- Session management for context persistence
- RAG search within sessions
- Provider management and health checks
- CORS middleware for web clients
- Global exception handling

Based on Boris' Best Practices for production-ready API design.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import psutil
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from contextflow.api.models import (
    # Response Models
    AnalysisResponse,
    # Request Models
    AnalyzeRequest,
    BatchProcessRequest,
    BatchProcessResponse,
    ChunkSuggestion,
    CreateSessionRequest,
    CreateSessionResponse,
    ErrorResponse,
    ErrorType,
    HealthResponse,
    HealthStatus,
    ProcessRequest,
    ProcessResponse,
    ProviderInfo,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SessionInfo,
    StreamChunk,
    StreamChunkType,
    TokenUsage,
    TrajectoryStepModel,
    VerificationResult,
    create_error_response,
)
from contextflow.core.config import get_config
from contextflow.core.orchestrator import ContextFlow, OrchestratorConfig
from contextflow.core.session import (
    Session,
    SessionManager,
    get_default_session_manager,
)
from contextflow.core.types import ProcessResult, StrategyType
from contextflow.providers.factory import get_available_providers, get_provider
from contextflow.utils.errors import (
    ContextFlowError,
    ProviderError,
    RateLimitError,
    TokenLimitError,
    ValidationError,
)
from contextflow.utils.logging import ProviderLogger, get_logger

# =============================================================================
# Module Constants
# =============================================================================

API_VERSION = "1.0.0"
API_PREFIX = "/api/v1"

logger = get_logger(__name__)
api_logger = ProviderLogger("api")


# =============================================================================
# Global State
# =============================================================================

# Singleton ContextFlow instance
_contextflow_instance: ContextFlow | None = None
_contextflow_lock = asyncio.Lock()

# Session storage (in-memory for API sessions, distinct from ContextFlow sessions)
_api_sessions: dict[str, dict[str, Any]] = {}
_api_sessions_lock = asyncio.Lock()

# Server start time for uptime tracking
_server_start_time: float = time.time()


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Handles startup and shutdown tasks:
    - Initialize ContextFlow on startup
    - Cleanup resources on shutdown
    """
    global _server_start_time
    _server_start_time = time.time()

    logger.info("Starting ContextFlow API server", version=API_VERSION)

    # Initialize ContextFlow lazily (on first request)
    yield

    # Cleanup on shutdown
    logger.info("Shutting down ContextFlow API server")

    global _contextflow_instance
    if _contextflow_instance is not None:
        await _contextflow_instance.close()
        _contextflow_instance = None

    logger.info("ContextFlow API server shutdown complete")


# =============================================================================
# FastAPI Application Setup
# =============================================================================

app = FastAPI(
    title="ContextFlow API",
    description="""
    Intelligent LLM Context Orchestration API.

    ContextFlow automatically selects the optimal strategy for processing
    large contexts with LLMs:
    - **GSD (Get Stuff Done)**: Direct processing for small contexts (<10K tokens)
    - **RALPH**: Structured iterative processing (10K-100K tokens)
    - **RLM**: Recursive processing with sub-agents (>100K tokens)

    Features:
    - Automatic strategy selection based on context analysis
    - Streaming responses via Server-Sent Events (SSE)
    - Session management for context persistence
    - RAG-based search within sessions
    - Batch processing with parallel execution
    - Verification loop for quality assurance

    All endpoints include comprehensive error handling and validation.
    """,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# =============================================================================
# CORS Middleware
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Execution-Time"],
)


# =============================================================================
# Dependency Injection
# =============================================================================


async def get_contextflow() -> ContextFlow:
    """
    Get or create the ContextFlow singleton instance.

    Uses lazy initialization with proper locking for thread safety.

    Returns:
        Configured ContextFlow instance

    Raises:
        HTTPException: If initialization fails
    """
    global _contextflow_instance

    async with _contextflow_lock:
        if _contextflow_instance is None:
            try:
                config = get_config()
                orchestrator_config = OrchestratorConfig(
                    enable_verification=True,
                    enable_sessions=True,
                    enable_hooks=True,
                    enable_cost_tracking=True,
                    enable_streaming=True,
                )

                _contextflow_instance = ContextFlow(
                    config=config,
                    orchestrator_config=orchestrator_config,
                )
                await _contextflow_instance.initialize()

                api_logger.info(
                    "ContextFlow initialized",
                    provider=_contextflow_instance.provider.name,
                )

            except Exception as e:
                api_logger.error("Failed to initialize ContextFlow", error=str(e))
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Service initialization failed: {str(e)}",
                )

    return _contextflow_instance


async def get_session_manager() -> SessionManager:
    """
    Get the session manager instance.

    Returns:
        SessionManager instance
    """
    return get_default_session_manager()


def get_request_id(request: Request) -> str:
    """
    Get or generate request ID for tracking.

    Args:
        request: FastAPI request

    Returns:
        Request ID string
    """
    return request.headers.get("X-Request-ID", f"req-{uuid.uuid4().hex[:12]}")


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle validation errors."""
    request_id = get_request_id(request)
    api_logger.warning(
        "Validation error",
        request_id=request_id,
        error=str(exc),
        field=getattr(exc, "field", None),
    )

    error_response = create_error_response(
        error=str(exc),
        error_type=ErrorType.VALIDATION_ERROR,
        error_code="VALIDATION_001",
        details={"field": getattr(exc, "field", None)},
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response.model_dump(mode="json"),
    )


@app.exception_handler(RateLimitError)
async def rate_limit_error_handler(request: Request, exc: RateLimitError) -> JSONResponse:
    """Handle rate limit errors."""
    request_id = get_request_id(request)
    api_logger.warning(
        "Rate limit exceeded",
        request_id=request_id,
        error=str(exc),
    )

    error_response = create_error_response(
        error=str(exc),
        error_type=ErrorType.RATE_LIMIT_ERROR,
        error_code="RATE_LIMIT_001",
        details={"retry_after_seconds": 60},
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=error_response.model_dump(mode="json"),
        headers={"Retry-After": "60"},
    )


@app.exception_handler(TokenLimitError)
async def token_limit_error_handler(request: Request, exc: TokenLimitError) -> JSONResponse:
    """Handle token limit errors."""
    request_id = get_request_id(request)
    api_logger.warning(
        "Token limit exceeded",
        request_id=request_id,
        error=str(exc),
    )

    error_response = create_error_response(
        error=str(exc),
        error_type=ErrorType.TOKEN_LIMIT_ERROR,
        error_code="TOKEN_LIMIT_001",
        details={"max_tokens": getattr(exc, "max_tokens", None)},
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        content=error_response.model_dump(mode="json"),
    )


@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
    """Handle provider errors."""
    request_id = get_request_id(request)
    api_logger.error(
        "Provider error",
        request_id=request_id,
        error=str(exc),
    )

    error_response = create_error_response(
        error=str(exc),
        error_type=ErrorType.PROVIDER_ERROR,
        error_code="PROVIDER_001",
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content=error_response.model_dump(mode="json"),
    )


@app.exception_handler(ContextFlowError)
async def contextflow_error_handler(request: Request, exc: ContextFlowError) -> JSONResponse:
    """Handle ContextFlow-specific errors."""
    request_id = get_request_id(request)
    api_logger.error(
        "ContextFlow error",
        request_id=request_id,
        error=str(exc),
    )

    error_response = create_error_response(
        error=str(exc),
        error_type=ErrorType.INTERNAL_ERROR,
        error_code="CONTEXTFLOW_001",
        details=getattr(exc, "details", None),
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(mode="json"),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    request_id = get_request_id(request)

    error_type = ErrorType.INTERNAL_ERROR
    if exc.status_code == 404:
        error_type = ErrorType.NOT_FOUND_ERROR
    elif exc.status_code == 401:
        error_type = ErrorType.AUTHENTICATION_ERROR
    elif exc.status_code == 400:
        error_type = ErrorType.VALIDATION_ERROR

    error_response = create_error_response(
        error=exc.detail,
        error_type=error_type,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions."""
    request_id = get_request_id(request)
    api_logger.error(
        "Unhandled exception",
        request_id=request_id,
        error=str(exc),
        exc_info=True,
    )

    error_response = create_error_response(
        error="An internal error occurred. Please try again later.",
        error_type=ErrorType.INTERNAL_ERROR,
        error_code="INTERNAL_001",
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(mode="json"),
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _convert_process_result_to_response(
    result: ProcessResult,
    request_id: str,
) -> ProcessResponse:
    """
    Convert internal ProcessResult to API ProcessResponse.

    Args:
        result: Internal process result
        request_id: Request ID for tracking

    Returns:
        API ProcessResponse model
    """
    # Convert trajectory steps
    trajectory = [
        TrajectoryStepModel(
            step_type=step.step_type,
            timestamp=step.timestamp,
            tokens_used=step.tokens_used,
            cost_usd=step.cost_usd,
            duration_ms=step.metadata.get("duration_ms", 0.0),
            metadata=step.metadata,
        )
        for step in result.trajectory
    ]

    # Build token usage
    token_usage = TokenUsage(
        input_tokens=result.total_tokens // 2,  # Estimate split
        output_tokens=result.total_tokens - (result.total_tokens // 2),
        total_tokens=result.total_tokens,
        cost_usd=result.total_cost,
    )

    # Extract verification info from metadata
    verification_passed = result.metadata.get("verification_passed", True)
    verification_score = result.metadata.get("verification_score", 1.0)

    # Build verification details if available
    verification_details = None
    if "verification_issues" in result.metadata:
        verification_details = VerificationResult(
            passed=verification_passed,
            score=verification_score,
            issues=result.metadata.get("verification_issues", []),
            suggestions=result.metadata.get("verification_suggestions", []),
        )

    return ProcessResponse(
        success=True,
        answer=result.answer,
        strategy_used=result.strategy_used.value,
        token_usage=token_usage,
        execution_time=result.execution_time,
        verification_passed=verification_passed,
        verification_score=verification_score,
        verification_details=verification_details,
        trajectory=trajectory,
        sub_agent_count=result.sub_agent_count,
        warnings=result.warnings,
        metadata=result.metadata,
        request_id=request_id,
        created_at=result.created_at,
    )


async def _get_or_create_api_session(
    session_id: str | None,
    session_manager: SessionManager,
) -> Session | None:
    """
    Get or create an API session.

    Args:
        session_id: Optional session ID
        session_manager: Session manager instance

    Returns:
        Session if session_id provided, None otherwise
    """
    if not session_id:
        return None

    async with _api_sessions_lock:
        if session_id in _api_sessions:
            return await session_manager.get_session(_api_sessions[session_id]["cf_session_id"])

    return None


# =============================================================================
# Process Endpoints
# =============================================================================


@app.post(
    f"{API_PREFIX}/process",
    response_model=ProcessResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation Error"},
        429: {"model": ErrorResponse, "description": "Rate Limit Exceeded"},
        500: {"model": ErrorResponse, "description": "Internal Error"},
        502: {"model": ErrorResponse, "description": "Provider Error"},
    },
    tags=["Process"],
    summary="Process task with context",
    description="""
    Main processing endpoint for ContextFlow.

    Automatically selects the optimal strategy based on context size:
    - GSD: <10K tokens (direct processing)
    - RALPH: 10K-100K tokens (structured iterative)
    - RLM: >100K tokens (recursive with sub-agents)

    Supports optional verification to ensure response quality.
    """,
)
async def process_task(
    request: ProcessRequest,
    cf: ContextFlow = Depends(get_contextflow),
    req: Request = None,
) -> ProcessResponse:
    """
    Process a task with provided context.

    Args:
        request: Process request with task and context
        cf: ContextFlow instance
        req: FastAPI request for tracking

    Returns:
        ProcessResponse with answer and metadata
    """
    request_id = get_request_id(req)
    start_time = time.time()

    api_logger.info(
        "Processing task",
        request_id=request_id,
        task_length=len(request.task),
        has_documents=bool(request.documents),
        has_context=bool(request.context),
        strategy=request.strategy,
    )

    try:
        # Parse strategy
        strategy = StrategyType(request.strategy or "auto")

        # Execute processing
        result = await cf.process(
            task=request.task,
            documents=request.documents,
            context=request.context,
            strategy=strategy,
            constraints=request.constraints,
        )

        execution_time = time.time() - start_time

        api_logger.info(
            "Processing complete",
            request_id=request_id,
            execution_time=execution_time,
            strategy_used=result.strategy_used.value,
            tokens=result.total_tokens,
        )

        return _convert_process_result_to_response(result, request_id)

    except Exception as e:
        api_logger.error(
            "Processing failed",
            request_id=request_id,
            error=str(e),
        )
        raise


@app.post(
    f"{API_PREFIX}/process/stream",
    responses={
        400: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Error"},
    },
    tags=["Process"],
    summary="Stream processing response",
    description="""
    Streaming processing endpoint using Server-Sent Events (SSE).

    Returns chunks as they are generated, allowing real-time display
    of the response. Each chunk is a JSON object with type and content.

    Chunk types:
    - content: Response text chunk
    - metadata: Processing metadata
    - progress: Progress update
    - error: Error occurred
    - done: Stream complete
    """,
)
async def process_stream(
    request: ProcessRequest,
    cf: ContextFlow = Depends(get_contextflow),
    req: Request = None,
) -> StreamingResponse:
    """
    Stream processing response via SSE.

    Args:
        request: Process request with task and context
        cf: ContextFlow instance
        req: FastAPI request for tracking

    Returns:
        StreamingResponse with SSE chunks
    """
    request_id = get_request_id(req)
    chunk_index = 0

    api_logger.info(
        "Starting stream processing",
        request_id=request_id,
        task_length=len(request.task),
    )

    async def generate() -> AsyncIterator[str]:
        """Generate SSE chunks from streaming response."""
        nonlocal chunk_index

        try:
            # Send initial metadata chunk
            metadata_chunk = StreamChunk(
                type=StreamChunkType.METADATA,
                metadata={
                    "request_id": request_id,
                    "strategy": request.strategy or "auto",
                    "started_at": datetime.utcnow().isoformat(),
                },
                chunk_index=chunk_index,
            )
            yield f"data: {metadata_chunk.model_dump_json()}\n\n"
            chunk_index += 1

            # Parse strategy
            strategy = StrategyType(request.strategy or "auto")

            # Stream from ContextFlow
            async for text_chunk in cf.stream(
                task=request.task,
                documents=request.documents,
                context=request.context,
                strategy=strategy,
                constraints=request.constraints,
            ):
                content_chunk = StreamChunk(
                    type=StreamChunkType.CONTENT,
                    content=text_chunk,
                    chunk_index=chunk_index,
                )
                yield f"data: {content_chunk.model_dump_json()}\n\n"
                chunk_index += 1

            # Send done chunk
            done_chunk = StreamChunk(
                type=StreamChunkType.DONE,
                metadata={
                    "total_chunks": chunk_index,
                    "completed_at": datetime.utcnow().isoformat(),
                },
                chunk_index=chunk_index,
            )
            yield f"data: {done_chunk.model_dump_json()}\n\n"

            api_logger.info(
                "Stream processing complete",
                request_id=request_id,
                total_chunks=chunk_index,
            )

        except Exception as e:
            api_logger.error(
                "Stream processing failed",
                request_id=request_id,
                error=str(e),
            )

            error_chunk = StreamChunk(
                type=StreamChunkType.ERROR,
                error=str(e),
                chunk_index=chunk_index,
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
        },
    )


@app.post(
    f"{API_PREFIX}/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Error"},
    },
    tags=["Process"],
    summary="Analyze context without execution",
    description="""
    Analyze context to determine optimal processing strategy
    without actually executing the task.

    Returns:
    - Token count and complexity analysis
    - Recommended strategy
    - Estimated costs per provider
    - Chunking recommendations
    """,
)
async def analyze_context(
    request: AnalyzeRequest,
    cf: ContextFlow = Depends(get_contextflow),
    req: Request = None,
) -> AnalysisResponse:
    """
    Analyze context without processing.

    Args:
        request: Analysis request with documents/context
        cf: ContextFlow instance
        req: FastAPI request for tracking

    Returns:
        AnalysisResponse with recommendations
    """
    request_id = get_request_id(req)

    api_logger.info(
        "Analyzing context",
        request_id=request_id,
        has_documents=bool(request.documents),
        has_context=bool(request.context),
    )

    try:
        # Perform analysis
        analysis = await cf.analyze(
            documents=request.documents,
            context=request.context,
        )

        # Build chunk suggestion if requested
        chunk_suggestion = None
        if request.include_chunk_suggestion and analysis.metadata:
            chunk_data = analysis.metadata.get("chunk_suggestion")
            if chunk_data:
                chunk_suggestion = ChunkSuggestion(
                    strategy=chunk_data.get("strategy", "semantic"),
                    chunk_size=chunk_data.get("chunk_size", 4000),
                    overlap=chunk_data.get("overlap", 200),
                    estimated_chunks=chunk_data.get("estimated_chunks", 1),
                    rationale=chunk_data.get("rationale", ""),
                )

        # Estimate costs for different providers
        estimated_costs = {}
        base_cost = analysis.estimated_cost
        estimated_costs["claude"] = base_cost
        estimated_costs["openai"] = base_cost * 1.2
        estimated_costs["gemini"] = base_cost * 0.8
        estimated_costs["ollama"] = 0.0  # Local

        # Determine complexity level
        if analysis.complexity_score < 0.3:
            complexity = "low"
        elif analysis.complexity_score < 0.6:
            complexity = "medium"
        elif analysis.complexity_score < 0.85:
            complexity = "high"
        else:
            complexity = "very_high"

        api_logger.info(
            "Analysis complete",
            request_id=request_id,
            token_count=analysis.token_count,
            complexity=complexity,
            recommended_strategy=analysis.recommended_strategy.value,
        )

        return AnalysisResponse(
            token_count=analysis.token_count,
            density=analysis.density_score,
            complexity=complexity,
            complexity_score=analysis.complexity_score,
            recommended_strategy=analysis.recommended_strategy.value,
            estimated_costs=estimated_costs,
            estimated_time=analysis.estimated_time_seconds,
            structure_type=analysis.structure_type,
            chunk_suggestion=chunk_suggestion,
            warnings=analysis.warnings,
            metadata=analysis.metadata,
        )

    except Exception as e:
        api_logger.error(
            "Analysis failed",
            request_id=request_id,
            error=str(e),
        )
        raise


@app.post(
    f"{API_PREFIX}/batch",
    response_model=BatchProcessResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Error"},
    },
    tags=["Process"],
    summary="Batch process multiple tasks",
    description="""
    Process multiple tasks in batch, optionally in parallel.

    Supports:
    - Parallel processing with configurable concurrency
    - Fail-fast mode to stop on first error
    - Aggregated token usage and timing
    """,
)
async def batch_process(
    request: BatchProcessRequest,
    cf: ContextFlow = Depends(get_contextflow),
    req: Request = None,
) -> BatchProcessResponse:
    """
    Process multiple tasks in batch.

    Args:
        request: Batch request with multiple tasks
        cf: ContextFlow instance
        req: FastAPI request for tracking

    Returns:
        BatchProcessResponse with all results
    """
    request_id = get_request_id(req)
    start_time = time.time()

    api_logger.info(
        "Starting batch processing",
        request_id=request_id,
        total_requests=len(request.requests),
        parallel=request.parallel,
        max_concurrent=request.max_concurrent,
    )

    results: list[ProcessResponse | ErrorResponse] = []
    total_tokens = 0
    total_cost = 0.0

    async def process_single(
        proc_request: ProcessRequest,
        index: int,
    ) -> ProcessResponse | ErrorResponse:
        """Process a single request from the batch."""
        try:
            strategy = StrategyType(proc_request.strategy or "auto")

            result = await cf.process(
                task=proc_request.task,
                documents=proc_request.documents,
                context=proc_request.context,
                strategy=strategy,
                constraints=proc_request.constraints,
            )

            return _convert_process_result_to_response(
                result,
                f"{request_id}-{index}",
            )

        except Exception as e:
            return create_error_response(
                error=str(e),
                error_type=ErrorType.INTERNAL_ERROR,
                request_id=f"{request_id}-{index}",
            )

    if request.parallel:
        # Process in parallel with concurrency limit
        semaphore = asyncio.Semaphore(request.max_concurrent)

        async def process_with_semaphore(
            proc_request: ProcessRequest,
            index: int,
        ) -> ProcessResponse | ErrorResponse:
            async with semaphore:
                return await process_single(proc_request, index)

        tasks = [
            process_with_semaphore(proc_request, i)
            for i, proc_request in enumerate(request.requests)
        ]

        if request.fail_fast:
            # Use gather with return_exceptions to catch first error
            try:
                results = await asyncio.gather(*tasks, return_exceptions=False)
            except Exception:
                # Add error for remaining tasks
                completed = len(results)
                for i in range(completed, len(request.requests)):
                    results.append(
                        create_error_response(
                            error="Batch cancelled due to earlier failure",
                            error_type=ErrorType.INTERNAL_ERROR,
                            request_id=f"{request_id}-{i}",
                        )
                    )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=False)

    else:
        # Process sequentially
        for i, proc_request in enumerate(request.requests):
            result = await process_single(proc_request, i)
            results.append(result)

            if request.fail_fast and isinstance(result, ErrorResponse):
                # Add cancelled results for remaining
                for j in range(i + 1, len(request.requests)):
                    results.append(
                        create_error_response(
                            error="Batch cancelled due to earlier failure",
                            error_type=ErrorType.INTERNAL_ERROR,
                            request_id=f"{request_id}-{j}",
                        )
                    )
                break

    # Calculate totals
    successful_count = 0
    for result in results:
        if isinstance(result, ProcessResponse) and result.success:
            successful_count += 1
            total_tokens += result.token_usage.total_tokens
            total_cost += result.token_usage.cost_usd

    failed_count = len(results) - successful_count
    total_execution_time = time.time() - start_time

    api_logger.info(
        "Batch processing complete",
        request_id=request_id,
        successful=successful_count,
        failed=failed_count,
        total_time=total_execution_time,
    )

    return BatchProcessResponse(
        success=failed_count == 0,
        results=results,
        total_requests=len(request.requests),
        successful_count=successful_count,
        failed_count=failed_count,
        total_execution_time=total_execution_time,
        total_token_usage=TokenUsage(
            input_tokens=total_tokens // 2,
            output_tokens=total_tokens - (total_tokens // 2),
            total_tokens=total_tokens,
            cost_usd=total_cost,
        ),
    )


# =============================================================================
# Search Endpoints
# =============================================================================


@app.post(
    f"{API_PREFIX}/search",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation Error"},
        404: {"model": ErrorResponse, "description": "Session Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Error"},
    },
    tags=["Search"],
    summary="Search within session context",
    description="""
    Perform RAG-based search within a session's indexed context.

    Returns relevant content chunks with similarity scores.
    Requires an active session with indexed documents.
    """,
)
async def search_context(
    request: SearchRequest,
    session_manager: SessionManager = Depends(get_session_manager),
    req: Request = None,
) -> SearchResponse:
    """
    Search for relevant content in session context.

    Args:
        request: Search request with query
        session_manager: Session manager instance
        req: FastAPI request for tracking

    Returns:
        SearchResponse with matching results
    """
    request_id = get_request_id(req)
    start_time = time.time()

    api_logger.info(
        "Searching context",
        request_id=request_id,
        query_length=len(request.query),
        session_id=request.session_id,
        max_results=request.max_results,
    )

    try:
        # Get relevant context from session manager
        context = await session_manager.get_relevant_context(
            query=request.query,
            max_tokens=request.max_results * 500,  # Estimate tokens per result
        )

        # Convert observations to search results
        results = []
        for obs in context.observations:
            if request.threshold > 0 and obs.relevance_score < request.threshold:
                continue

            result = SearchResult(
                content=obs.content,
                score=obs.relevance_score if request.include_scores else 0.0,
                chunk_id=obs.id,
                metadata={
                    "session_id": obs.session_id,
                    "type": obs.type.value,
                    "timestamp": obs.timestamp.isoformat(),
                },
            )
            results.append(result)

            if len(results) >= request.max_results:
                break

        search_time_ms = (time.time() - start_time) * 1000

        api_logger.info(
            "Search complete",
            request_id=request_id,
            results_found=len(results),
            search_time_ms=search_time_ms,
        )

        return SearchResponse(
            results=results,
            total_results=len(results),
            query=request.query,
            search_time_ms=search_time_ms,
        )

    except Exception as e:
        api_logger.error(
            "Search failed",
            request_id=request_id,
            error=str(e),
        )
        raise


# =============================================================================
# Session Endpoints
# =============================================================================


@app.post(
    f"{API_PREFIX}/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Error"},
    },
    tags=["Sessions"],
    summary="Create new session",
    description="""
    Create a new session for context persistence.

    Sessions allow:
    - Persistent context across multiple requests
    - RAG-based search within session documents
    - Observation tracking and retrieval
    """,
)
async def create_session(
    request: CreateSessionRequest,
    session_manager: SessionManager = Depends(get_session_manager),
    req: Request = None,
) -> CreateSessionResponse:
    """
    Create a new API session.

    Args:
        request: Session creation request
        session_manager: Session manager instance
        req: FastAPI request for tracking

    Returns:
        CreateSessionResponse with session details
    """
    request_id = get_request_id(req)

    api_logger.info(
        "Creating session",
        request_id=request_id,
        has_documents=bool(request.documents),
        ttl_seconds=request.ttl_seconds,
    )

    try:
        # Create session in session manager
        session = await session_manager.start_session(
            metadata={
                "request_id": request_id,
                "ttl_seconds": request.ttl_seconds,
                **(request.metadata or {}),
            }
        )

        # Store in API sessions map
        async with _api_sessions_lock:
            _api_sessions[session.id] = {
                "cf_session_id": session.id,
                "created_at": datetime.utcnow(),
                "ttl_seconds": request.ttl_seconds,
                "document_count": len(request.documents or []),
                "context_length": len(request.context or ""),
            }

        # Calculate tokens if context provided
        total_tokens = 0
        if request.context:
            total_tokens = len(request.context) // 4

        session_info = SessionInfo(
            session_id=session.id,
            created_at=session.started_at,
            last_accessed=session.started_at,
            document_count=len(request.documents or []),
            total_tokens=total_tokens,
            chunk_count=0,
            metadata=session.metadata,
        )

        api_logger.info(
            "Session created",
            request_id=request_id,
            session_id=session.id,
        )

        return CreateSessionResponse(
            session_id=session.id,
            session_info=session_info,
        )

    except Exception as e:
        api_logger.error(
            "Session creation failed",
            request_id=request_id,
            error=str(e),
        )
        raise


@app.get(
    f"{API_PREFIX}/sessions/{{session_id}}",
    response_model=SessionInfo,
    responses={
        404: {"model": ErrorResponse, "description": "Session Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Error"},
    },
    tags=["Sessions"],
    summary="Get session info",
    description="Retrieve information about an existing session.",
)
async def get_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
    req: Request = None,
) -> SessionInfo:
    """
    Get session information.

    Args:
        session_id: Session ID to retrieve
        session_manager: Session manager instance
        req: FastAPI request for tracking

    Returns:
        SessionInfo with session details
    """
    request_id = get_request_id(req)

    api_logger.debug(
        "Getting session info",
        request_id=request_id,
        session_id=session_id,
    )

    session = await session_manager.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    return SessionInfo(
        session_id=session.id,
        created_at=session.started_at,
        last_accessed=datetime.utcnow(),
        document_count=session.metadata.get("document_count", 0),
        total_tokens=session.total_tokens,
        chunk_count=len(session.observations),
        metadata=session.metadata,
    )


@app.delete(
    f"{API_PREFIX}/sessions/{{session_id}}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    tags=["Sessions"],
    summary="Delete session",
    description="End and delete a session, freeing associated resources.",
)
async def delete_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
    req: Request = None,
):
    """
    Delete a session.

    Args:
        session_id: Session ID to delete
        session_manager: Session manager instance
        req: FastAPI request for tracking
    """
    request_id = get_request_id(req)

    api_logger.info(
        "Deleting session",
        request_id=request_id,
        session_id=session_id,
    )

    # Check if session exists
    session = await session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    # End and delete session
    if session.is_active:
        await session_manager.end_session(session_id)

    deleted = await session_manager.delete_session(session_id)

    # Remove from API sessions
    async with _api_sessions_lock:
        _api_sessions.pop(session_id, None)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    api_logger.info(
        "Session deleted",
        request_id=request_id,
        session_id=session_id,
    )


# =============================================================================
# Health Endpoints
# =============================================================================


@app.get(
    f"{API_PREFIX}/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="""
    Check API health status and get system metrics.

    Returns:
    - Overall health status
    - Provider availability
    - Active sessions count
    - Memory usage
    - Uptime
    """,
)
async def health_check(
    req: Request = None,
) -> HealthResponse:
    """
    Check API health status.

    Returns:
        HealthResponse with health metrics
    """
    global _contextflow_instance

    # Calculate uptime
    uptime_seconds = time.time() - _server_start_time

    # Get memory usage
    try:
        process = psutil.Process(os.getpid())
        memory_usage_mb = process.memory_info().rss / 1024 / 1024
    except Exception:
        memory_usage_mb = None

    # Get provider info
    providers: list[ProviderInfo] = []
    overall_status = HealthStatus.HEALTHY

    try:
        available_providers = get_available_providers()

        for provider_name in available_providers:
            try:
                provider = get_provider(provider_name)
                capabilities = provider.capabilities

                provider_info = ProviderInfo(
                    name=provider_name,
                    available=True,
                    models=capabilities.supported_models,
                    max_context=capabilities.max_context_tokens,
                    supports_streaming=capabilities.supports_streaming,
                    supports_tools=capabilities.supports_tools,
                    rate_limit_rpm=capabilities.rate_limit_rpm,
                )
                providers.append(provider_info)

            except Exception as e:
                api_logger.warning(
                    f"Provider {provider_name} unavailable: {e}",
                )
                providers.append(
                    ProviderInfo(
                        name=provider_name,
                        available=False,
                        models=[],
                        max_context=0,
                    )
                )
                overall_status = HealthStatus.DEGRADED

    except Exception as e:
        api_logger.error(f"Failed to get providers: {e}")
        overall_status = HealthStatus.UNHEALTHY

    # Count active sessions
    async with _api_sessions_lock:
        active_sessions = len(_api_sessions)

    return HealthResponse(
        status=overall_status,
        version=API_VERSION,
        providers=providers,
        uptime_seconds=uptime_seconds,
        active_sessions=active_sessions,
        memory_usage_mb=memory_usage_mb,
        timestamp=datetime.utcnow(),
    )


@app.get(
    f"{API_PREFIX}/providers",
    response_model=list[ProviderInfo],
    tags=["Health"],
    summary="List available providers",
    description="Get information about all configured LLM providers.",
)
async def list_providers(
    req: Request = None,
) -> list[ProviderInfo]:
    """
    List all available LLM providers.

    Returns:
        List of ProviderInfo with provider details
    """
    providers: list[ProviderInfo] = []

    try:
        available_providers = get_available_providers()

        for provider_name in available_providers:
            try:
                provider = get_provider(provider_name)
                capabilities = provider.capabilities

                provider_info = ProviderInfo(
                    name=provider_name,
                    available=True,
                    models=capabilities.supported_models,
                    max_context=capabilities.max_context_tokens,
                    supports_streaming=capabilities.supports_streaming,
                    supports_tools=capabilities.supports_tools,
                    rate_limit_rpm=capabilities.rate_limit_rpm,
                )
                providers.append(provider_info)

            except Exception as e:
                api_logger.warning(
                    f"Provider {provider_name} unavailable: {e}",
                )
                providers.append(
                    ProviderInfo(
                        name=provider_name,
                        available=False,
                        models=[],
                        max_context=0,
                    )
                )

    except Exception as e:
        api_logger.error(f"Failed to get providers: {e}")

    return providers


# =============================================================================
# Root Endpoint
# =============================================================================


@app.get(
    "/",
    tags=["Root"],
    summary="API root",
    description="Welcome endpoint with API information.",
)
async def root() -> dict[str, Any]:
    """
    Root endpoint with API information.

    Returns:
        API welcome message and links
    """
    return {
        "name": "ContextFlow API",
        "version": API_VERSION,
        "description": "Intelligent LLM Context Orchestration",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "health": f"{API_PREFIX}/health",
    }


# =============================================================================
# Server Entry Point
# =============================================================================


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app instance
    """
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
) -> None:
    """
    Run the API server with uvicorn.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
        workers: Number of worker processes
    """
    import uvicorn

    uvicorn.run(
        "contextflow.api.server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info",
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "app",
    "create_app",
    "run_server",
    "get_contextflow",
    "get_session_manager",
    "API_VERSION",
    "API_PREFIX",
]
