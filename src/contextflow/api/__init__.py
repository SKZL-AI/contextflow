"""
FastAPI REST API for ContextFlow.

This package provides the REST API server, models, and utilities
for the ContextFlow framework.

Usage:
    # Run the server
    from contextflow.api import run_server
    run_server(host="0.0.0.0", port=8000)

    # Or use the app directly with uvicorn
    uvicorn contextflow.api.server:app --reload
"""

from contextflow.api.models import (
    AnalysisResponse,
    AnalyzeRequest,
    BatchProcessRequest,
    BatchProcessResponse,
    ChunkSuggestion,
    CLIAnalyzeArgs,
    # CLI Models
    CLIProcessArgs,
    ComplexityLevel,
    CreateSessionRequest,
    CreateSessionResponse,
    ErrorResponse,
    ErrorType,
    HealthResponse,
    HealthStatus,
    MCPResourceInfo,
    # MCP Models
    MCPToolRequest,
    MCPToolResponse,
    # Request Models
    ProcessRequest,
    # Response Models
    ProcessResponse,
    ProviderInfo,
    SearchRequest,
    SearchResponse,
    SearchResult,
    # Session Models
    SessionInfo,
    StreamChunk,
    # Enums
    StreamChunkType,
    # Supporting Models
    TokenUsage,
    TrajectoryStepModel,
    VerificationResult,
    # Utility Functions
    create_error_response,
    create_success_response,
)
from contextflow.api.server import (
    API_PREFIX,
    API_VERSION,
    app,
    create_app,
    get_contextflow,
    get_session_manager,
    run_server,
)
from contextflow.api.websocket import (
    WebSocketManager,
    WSAnalyzeMessage,
    WSMessageType,
    WSProcessMessage,
    WSResponse,
    WSResponseType,
)
from contextflow.api.websocket import (
    manager as websocket_manager,
)
from contextflow.api.websocket import (
    router as websocket_router,
)

__all__ = [
    # Server
    "app",
    "create_app",
    "run_server",
    "get_contextflow",
    "get_session_manager",
    "API_VERSION",
    "API_PREFIX",
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
    # WebSocket
    "WebSocketManager",
    "WSMessageType",
    "WSResponseType",
    "WSProcessMessage",
    "WSAnalyzeMessage",
    "WSResponse",
    "websocket_router",
    "websocket_manager",
]
