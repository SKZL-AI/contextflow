"""
Pydantic models for ContextFlow REST API, CLI, and MCP Server.

This module defines all request/response models with full validation,
OpenAPI documentation, and JSON serialization support.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# =============================================================================
# Enums for API
# =============================================================================


class StreamChunkType(str, Enum):
    """Types of streaming chunks."""

    CONTENT = "content"
    METADATA = "metadata"
    ERROR = "error"
    DONE = "done"
    PROGRESS = "progress"


class ComplexityLevel(str, Enum):
    """Context complexity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class HealthStatus(str, Enum):
    """Service health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ErrorType(str, Enum):
    """API error types for categorization."""

    VALIDATION_ERROR = "validation_error"
    PROVIDER_ERROR = "provider_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TOKEN_LIMIT_ERROR = "token_limit_error"
    CONFIGURATION_ERROR = "configuration_error"
    INTERNAL_ERROR = "internal_error"
    NOT_FOUND_ERROR = "not_found_error"
    AUTHENTICATION_ERROR = "authentication_error"


# =============================================================================
# Supporting Models
# =============================================================================


class TokenUsage(BaseModel):
    """Token usage statistics for a request."""

    input_tokens: int = Field(
        ...,
        ge=0,
        description="Number of input tokens consumed",
    )
    output_tokens: int = Field(
        ...,
        ge=0,
        description="Number of output tokens generated",
    )
    total_tokens: int = Field(
        ...,
        ge=0,
        description="Total tokens (input + output)",
    )
    cost_usd: float = Field(
        ...,
        ge=0.0,
        description="Estimated cost in USD",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "input_tokens": 1500,
                "output_tokens": 500,
                "total_tokens": 2000,
                "cost_usd": 0.025,
            }
        }
    )

    @model_validator(mode="after")
    def validate_total(self) -> TokenUsage:
        """Ensure total equals input + output."""
        expected_total = self.input_tokens + self.output_tokens
        if self.total_tokens != expected_total:
            self.total_tokens = expected_total
        return self


class ProviderInfo(BaseModel):
    """Information about an LLM provider."""

    name: str = Field(
        ...,
        min_length=1,
        description="Provider name (e.g., 'claude', 'openai')",
    )
    available: bool = Field(
        ...,
        description="Whether the provider is currently available",
    )
    models: list[str] = Field(
        default_factory=list,
        description="List of available model names",
    )
    max_context: int = Field(
        ...,
        gt=0,
        description="Maximum context window size in tokens",
    )
    supports_streaming: bool = Field(
        default=True,
        description="Whether the provider supports streaming responses",
    )
    supports_tools: bool = Field(
        default=False,
        description="Whether the provider supports tool/function calling",
    )
    rate_limit_rpm: int | None = Field(
        default=None,
        description="Rate limit in requests per minute",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "claude",
                "available": True,
                "models": [
                    "claude-sonnet-4-20250514",
                    "claude-opus-4-20250514",
                ],
                "max_context": 200000,
                "supports_streaming": True,
                "supports_tools": True,
                "rate_limit_rpm": 50,
            }
        }
    )


class ChunkSuggestion(BaseModel):
    """Suggestion for context chunking."""

    strategy: str = Field(
        ...,
        description="Recommended chunking strategy",
    )
    chunk_size: int = Field(
        ...,
        gt=0,
        description="Recommended chunk size in tokens",
    )
    overlap: int = Field(
        default=0,
        ge=0,
        description="Recommended overlap between chunks",
    )
    estimated_chunks: int = Field(
        ...,
        gt=0,
        description="Estimated number of chunks",
    )
    rationale: str = Field(
        default="",
        description="Explanation for the recommendation",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "strategy": "semantic",
                "chunk_size": 4000,
                "overlap": 200,
                "estimated_chunks": 12,
                "rationale": "Document has clear section boundaries",
            }
        }
    )


class VerificationResult(BaseModel):
    """Result of answer verification."""

    passed: bool = Field(
        ...,
        description="Whether verification passed",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Verification confidence score (0.0-1.0)",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of identified issues",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for improvement",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "passed": True,
                "score": 0.92,
                "issues": [],
                "suggestions": ["Consider adding specific examples"],
            }
        }
    )


class TrajectoryStepModel(BaseModel):
    """Single step in execution trajectory for API response."""

    step_type: str = Field(
        ...,
        description="Type of step (analysis, strategy, sub_agent, aggregation)",
    )
    timestamp: datetime = Field(
        ...,
        description="When the step occurred",
    )
    tokens_used: int = Field(
        default=0,
        ge=0,
        description="Tokens consumed in this step",
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost of this step in USD",
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Duration of this step in milliseconds",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional step metadata",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "step_type": "sub_agent",
                "timestamp": "2026-01-20T12:00:00Z",
                "tokens_used": 500,
                "cost_usd": 0.005,
                "duration_ms": 1250.5,
                "metadata": {"chunk_id": "chunk_3", "confidence": 0.85},
            }
        }
    )


# =============================================================================
# Request Models
# =============================================================================


class ProcessRequest(BaseModel):
    """Request to process a task with context."""

    task: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The task or question to process",
    )
    documents: list[str] | None = Field(
        default=None,
        description="List of file paths or URLs to process as context",
    )
    context: str | None = Field(
        default=None,
        max_length=10_000_000,
        description="Direct context string (alternative to documents)",
    )
    strategy: str | None = Field(
        default="auto",
        description="Strategy override: auto, gsd_direct, ralph_structured, rlm_full",
    )
    provider: str | None = Field(
        default=None,
        description="LLM provider override (e.g., 'claude', 'openai')",
    )
    model: str | None = Field(
        default=None,
        description="Specific model override (e.g., 'claude-sonnet-4-20250514')",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response",
    )
    constraints: list[str] | None = Field(
        default=None,
        description="Additional constraints or requirements",
    )
    config: dict[str, Any] | None = Field(
        default=None,
        description="Additional configuration options",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for context persistence",
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Maximum tokens for response",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation",
    )
    verify: bool = Field(
        default=True,
        description="Whether to verify the response",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "task": "Summarize the key findings from these research papers",
                "documents": ["/path/to/paper1.pdf", "/path/to/paper2.pdf"],
                "strategy": "auto",
                "stream": False,
                "verify": True,
            }
        }
    )

    @model_validator(mode="after")
    def validate_context_source(self) -> ProcessRequest:
        """Ensure at least one context source is provided."""
        if not self.documents and not self.context:
            # Allow tasks without context for simple queries
            pass
        return self

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str | None) -> str | None:
        """Validate strategy name."""
        if v is None:
            return "auto"
        valid_strategies = {
            "auto",
            "gsd_direct",
            "ralph_structured",
            "rlm_full",
            "rlm_dense",
        }
        if v.lower() not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{v}'. Must be one of: {valid_strategies}"
            )
        return v.lower()


class AnalyzeRequest(BaseModel):
    """Request to analyze context without processing."""

    documents: list[str] | None = Field(
        default=None,
        description="List of file paths or URLs to analyze",
    )
    context: str | None = Field(
        default=None,
        max_length=10_000_000,
        description="Direct context string to analyze",
    )
    provider: str | None = Field(
        default=None,
        description="Provider to use for cost estimation",
    )
    include_chunk_suggestion: bool = Field(
        default=True,
        description="Whether to include chunking recommendations",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "documents": ["/path/to/document.txt"],
                "include_chunk_suggestion": True,
            }
        }
    )

    @model_validator(mode="after")
    def validate_has_content(self) -> AnalyzeRequest:
        """Ensure at least one content source is provided."""
        if not self.documents and not self.context:
            raise ValueError("Either 'documents' or 'context' must be provided")
        return self


class SearchRequest(BaseModel):
    """Request to search within session context."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Search query",
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID to search within",
    )
    include_scores: bool = Field(
        default=True,
        description="Whether to include similarity scores",
    )
    threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "machine learning algorithms",
                "max_results": 5,
                "include_scores": True,
                "threshold": 0.5,
            }
        }
    )


class BatchProcessRequest(BaseModel):
    """Request to process multiple tasks in batch."""

    requests: list[ProcessRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of process requests",
    )
    parallel: bool = Field(
        default=True,
        description="Whether to process requests in parallel",
    )
    max_concurrent: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent requests",
    )
    fail_fast: bool = Field(
        default=False,
        description="Stop on first error",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "requests": [
                    {"task": "Summarize document 1", "documents": ["/doc1.txt"]},
                    {"task": "Summarize document 2", "documents": ["/doc2.txt"]},
                ],
                "parallel": True,
                "max_concurrent": 5,
            }
        }
    )


# =============================================================================
# Response Models
# =============================================================================


class ProcessResponse(BaseModel):
    """Response from task processing."""

    success: bool = Field(
        ...,
        description="Whether processing completed successfully",
    )
    answer: str = Field(
        ...,
        description="The generated answer",
    )
    strategy_used: str = Field(
        ...,
        description="Strategy that was used for processing",
    )
    token_usage: TokenUsage = Field(
        ...,
        description="Token consumption details",
    )
    execution_time: float = Field(
        ...,
        ge=0.0,
        description="Total execution time in seconds",
    )
    verification_passed: bool = Field(
        ...,
        description="Whether verification checks passed",
    )
    verification_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Verification confidence score",
    )
    verification_details: VerificationResult | None = Field(
        default=None,
        description="Detailed verification results",
    )
    trajectory: list[TrajectoryStepModel] = Field(
        default_factory=list,
        description="Execution trajectory steps",
    )
    sub_agent_count: int = Field(
        default=0,
        ge=0,
        description="Number of sub-agents used",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings generated during processing",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional response metadata",
    )
    request_id: str | None = Field(
        default=None,
        description="Unique request identifier",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response creation timestamp",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "answer": "The key findings from the research papers are...",
                "strategy_used": "ralph_structured",
                "token_usage": {
                    "input_tokens": 15000,
                    "output_tokens": 2000,
                    "total_tokens": 17000,
                    "cost_usd": 0.25,
                },
                "execution_time": 12.5,
                "verification_passed": True,
                "verification_score": 0.95,
                "sub_agent_count": 3,
                "warnings": [],
            }
        }
    )


class AnalysisResponse(BaseModel):
    """Response from context analysis."""

    token_count: int = Field(
        ...,
        ge=0,
        description="Total token count of the context",
    )
    density: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Information density score (0.0-1.0)",
    )
    complexity: str = Field(
        ...,
        description="Complexity level: low, medium, high, very_high",
    )
    complexity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Numeric complexity score (0.0-1.0)",
    )
    recommended_strategy: str = Field(
        ...,
        description="Recommended processing strategy",
    )
    estimated_costs: dict[str, float] = Field(
        ...,
        description="Estimated costs by provider in USD",
    )
    estimated_time: float = Field(
        ...,
        ge=0.0,
        description="Estimated processing time in seconds",
    )
    structure_type: str = Field(
        ...,
        description="Detected content structure type",
    )
    chunk_suggestion: ChunkSuggestion | None = Field(
        default=None,
        description="Recommended chunking configuration",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Analysis warnings",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional analysis metadata",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "token_count": 45000,
                "density": 0.72,
                "complexity": "high",
                "complexity_score": 0.78,
                "recommended_strategy": "ralph_structured",
                "estimated_costs": {
                    "claude": 0.15,
                    "openai": 0.18,
                    "gemini": 0.10,
                },
                "estimated_time": 25.0,
                "structure_type": "markdown",
                "chunk_suggestion": {
                    "strategy": "semantic",
                    "chunk_size": 4000,
                    "overlap": 200,
                    "estimated_chunks": 12,
                    "rationale": "Document has clear section boundaries",
                },
                "warnings": [],
            }
        }
    )


class SearchResult(BaseModel):
    """Single search result."""

    content: str = Field(
        ...,
        description="Matched content snippet",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score",
    )
    chunk_id: str = Field(
        ...,
        description="Source chunk identifier",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional result metadata",
    )


class SearchResponse(BaseModel):
    """Response from context search."""

    results: list[SearchResult] = Field(
        ...,
        description="List of search results",
    )
    total_results: int = Field(
        ...,
        ge=0,
        description="Total number of matching results",
    )
    query: str = Field(
        ...,
        description="Original search query",
    )
    search_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Search execution time in milliseconds",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "content": "Machine learning is a subset of AI...",
                        "score": 0.92,
                        "chunk_id": "chunk_5",
                        "metadata": {"source": "doc1.pdf", "page": 3},
                    }
                ],
                "total_results": 1,
                "query": "machine learning",
                "search_time_ms": 15.2,
            }
        }
    )


class StreamChunk(BaseModel):
    """Single chunk from streaming response."""

    type: StreamChunkType = Field(
        ...,
        description="Type of stream chunk",
    )
    content: str | None = Field(
        default=None,
        description="Content for content-type chunks",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Metadata for metadata-type chunks",
    )
    error: str | None = Field(
        default=None,
        description="Error message for error-type chunks",
    )
    chunk_index: int = Field(
        default=0,
        ge=0,
        description="Sequential chunk index",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Chunk timestamp",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "content",
                "content": "The analysis shows that",
                "chunk_index": 5,
                "timestamp": "2026-01-20T12:00:00Z",
            }
        }
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    success: bool = Field(
        default=False,
        description="Always false for errors",
    )
    error: str = Field(
        ...,
        description="Human-readable error message",
    )
    error_type: ErrorType = Field(
        ...,
        description="Error classification",
    )
    error_code: str | None = Field(
        default=None,
        description="Machine-readable error code",
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error details",
    )
    request_id: str | None = Field(
        default=None,
        description="Request identifier for support",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "error": "Rate limit exceeded for provider 'claude'",
                "error_type": "rate_limit_error",
                "error_code": "RATE_LIMIT_001",
                "details": {"retry_after_seconds": 60, "provider": "claude"},
                "request_id": "req_abc123",
            }
        }
    )


class HealthResponse(BaseModel):
    """API health check response."""

    status: HealthStatus = Field(
        ...,
        description="Overall service health status",
    )
    version: str = Field(
        ...,
        description="API version string",
    )
    providers: list[ProviderInfo] = Field(
        ...,
        description="Status of each configured provider",
    )
    uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="Service uptime in seconds",
    )
    active_sessions: int = Field(
        default=0,
        ge=0,
        description="Number of active sessions",
    )
    memory_usage_mb: float | None = Field(
        default=None,
        description="Current memory usage in MB",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "providers": [
                    {
                        "name": "claude",
                        "available": True,
                        "models": ["claude-sonnet-4-20250514"],
                        "max_context": 200000,
                        "supports_streaming": True,
                        "supports_tools": True,
                    }
                ],
                "uptime_seconds": 3600.0,
                "active_sessions": 5,
                "memory_usage_mb": 256.5,
            }
        }
    )


class BatchProcessResponse(BaseModel):
    """Response from batch processing."""

    success: bool = Field(
        ...,
        description="Whether all requests succeeded",
    )
    results: list[ProcessResponse | ErrorResponse] = Field(
        ...,
        description="Results for each request",
    )
    total_requests: int = Field(
        ...,
        ge=1,
        description="Total number of requests",
    )
    successful_count: int = Field(
        ...,
        ge=0,
        description="Number of successful requests",
    )
    failed_count: int = Field(
        ...,
        ge=0,
        description="Number of failed requests",
    )
    total_execution_time: float = Field(
        ...,
        ge=0.0,
        description="Total execution time in seconds",
    )
    total_token_usage: TokenUsage = Field(
        ...,
        description="Aggregated token usage",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "results": [],
                "total_requests": 5,
                "successful_count": 5,
                "failed_count": 0,
                "total_execution_time": 45.2,
                "total_token_usage": {
                    "input_tokens": 50000,
                    "output_tokens": 10000,
                    "total_tokens": 60000,
                    "cost_usd": 0.85,
                },
            }
        }
    )


# =============================================================================
# Session Models
# =============================================================================


class SessionInfo(BaseModel):
    """Information about a session."""

    session_id: str = Field(
        ...,
        description="Unique session identifier",
    )
    created_at: datetime = Field(
        ...,
        description="Session creation timestamp",
    )
    last_accessed: datetime = Field(
        ...,
        description="Last access timestamp",
    )
    document_count: int = Field(
        default=0,
        ge=0,
        description="Number of documents in session",
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total tokens in session context",
    )
    chunk_count: int = Field(
        default=0,
        ge=0,
        description="Number of chunks in RAG index",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Session metadata",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "sess_abc123",
                "created_at": "2026-01-20T10:00:00Z",
                "last_accessed": "2026-01-20T12:00:00Z",
                "document_count": 3,
                "total_tokens": 25000,
                "chunk_count": 8,
            }
        }
    )


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""

    documents: list[str] | None = Field(
        default=None,
        description="Initial documents to load",
    )
    context: str | None = Field(
        default=None,
        description="Initial context string",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Session metadata",
    )
    ttl_seconds: int | None = Field(
        default=3600,
        gt=0,
        description="Session time-to-live in seconds",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "documents": ["/path/to/doc1.txt", "/path/to/doc2.pdf"],
                "ttl_seconds": 7200,
            }
        }
    )


class CreateSessionResponse(BaseModel):
    """Response from session creation."""

    session_id: str = Field(
        ...,
        description="Created session identifier",
    )
    session_info: SessionInfo = Field(
        ...,
        description="Session details",
    )


# =============================================================================
# MCP Server Models
# =============================================================================


class MCPToolRequest(BaseModel):
    """Request for MCP tool execution."""

    tool_name: str = Field(
        ...,
        description="Name of the MCP tool to execute",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments",
    )
    session_id: str | None = Field(
        default=None,
        description="Session context",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tool_name": "process_context",
                "arguments": {
                    "task": "Summarize this document",
                    "context": "Document content here...",
                },
            }
        }
    )


class MCPToolResponse(BaseModel):
    """Response from MCP tool execution."""

    success: bool = Field(
        ...,
        description="Whether tool execution succeeded",
    )
    result: Any = Field(
        default=None,
        description="Tool execution result",
    )
    error: str | None = Field(
        default=None,
        description="Error message if failed",
    )
    execution_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Execution time in milliseconds",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "result": {"answer": "The document discusses...", "confidence": 0.95},
                "execution_time_ms": 2500.0,
            }
        }
    )


class MCPResourceInfo(BaseModel):
    """Information about an MCP resource."""

    uri: str = Field(
        ...,
        description="Resource URI",
    )
    name: str = Field(
        ...,
        description="Resource display name",
    )
    description: str | None = Field(
        default=None,
        description="Resource description",
    )
    mime_type: str | None = Field(
        default=None,
        description="Resource MIME type",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "uri": "contextflow://session/sess_abc123",
                "name": "Session Context",
                "description": "Current session context and documents",
                "mime_type": "application/json",
            }
        }
    )


# =============================================================================
# CLI Models
# =============================================================================


class CLIProcessArgs(BaseModel):
    """CLI process command arguments."""

    task: str = Field(
        ...,
        description="Task to process",
    )
    files: list[str] = Field(
        default_factory=list,
        description="Input files",
    )
    context: str | None = Field(
        default=None,
        description="Direct context",
    )
    strategy: str = Field(
        default="auto",
        description="Processing strategy",
    )
    provider: str | None = Field(
        default=None,
        description="LLM provider",
    )
    output: str | None = Field(
        default=None,
        description="Output file path",
    )
    format: str = Field(
        default="text",
        description="Output format: text, json, markdown",
    )
    verbose: bool = Field(
        default=False,
        description="Verbose output",
    )
    stream: bool = Field(
        default=False,
        description="Stream output",
    )


class CLIAnalyzeArgs(BaseModel):
    """CLI analyze command arguments."""

    files: list[str] = Field(
        default_factory=list,
        description="Files to analyze",
    )
    context: str | None = Field(
        default=None,
        description="Direct context",
    )
    format: str = Field(
        default="text",
        description="Output format: text, json",
    )
    verbose: bool = Field(
        default=False,
        description="Verbose output",
    )


# =============================================================================
# Utility Functions
# =============================================================================


def create_error_response(
    error: str,
    error_type: ErrorType,
    error_code: str | None = None,
    details: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> ErrorResponse:
    """Factory function to create error responses."""
    return ErrorResponse(
        success=False,
        error=error,
        error_type=error_type,
        error_code=error_code,
        details=details,
        request_id=request_id,
    )


def create_success_response(
    answer: str,
    strategy_used: str,
    token_usage: TokenUsage,
    execution_time: float,
    verification_passed: bool = True,
    verification_score: float = 1.0,
    **kwargs: Any,
) -> ProcessResponse:
    """Factory function to create success responses."""
    return ProcessResponse(
        success=True,
        answer=answer,
        strategy_used=strategy_used,
        token_usage=token_usage,
        execution_time=execution_time,
        verification_passed=verification_passed,
        verification_score=verification_score,
        **kwargs,
    )


__all__ = [
    # Enums
    "StreamChunkType",
    "ComplexityLevel",
    "HealthStatus",
    "ErrorType",
    # Supporting Models
    "TokenUsage",
    "ProviderInfo",
    "ChunkSuggestion",
    "VerificationResult",
    "TrajectoryStepModel",
    # Request Models
    "ProcessRequest",
    "AnalyzeRequest",
    "SearchRequest",
    "BatchProcessRequest",
    # Response Models
    "ProcessResponse",
    "AnalysisResponse",
    "SearchResult",
    "SearchResponse",
    "StreamChunk",
    "ErrorResponse",
    "HealthResponse",
    "BatchProcessResponse",
    # Session Models
    "SessionInfo",
    "CreateSessionRequest",
    "CreateSessionResponse",
    # MCP Models
    "MCPToolRequest",
    "MCPToolResponse",
    "MCPResourceInfo",
    # CLI Models
    "CLIProcessArgs",
    "CLIAnalyzeArgs",
    # Utility Functions
    "create_error_response",
    "create_success_response",
]
