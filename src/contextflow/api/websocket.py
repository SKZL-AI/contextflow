"""
WebSocket Support for ContextFlow API.

Real-time bidirectional communication for streaming processing with:
- Connection management with unique IDs
- Heartbeat/ping-pong for keepalive
- Graceful error handling
- Message validation
- Request cancellation support

Based on Boris' Best Practices for production-ready WebSocket design.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

from contextflow.core.orchestrator import ContextFlow
from contextflow.core.types import StrategyType
from contextflow.utils.errors import ContextFlowError, ValidationError
from contextflow.utils.logging import ProviderLogger, get_logger

# =============================================================================
# Module Constants
# =============================================================================

logger = get_logger(__name__)
ws_logger = ProviderLogger("websocket")

# WebSocket close codes
WS_CLOSE_NORMAL = 1000
WS_CLOSE_GOING_AWAY = 1001
WS_CLOSE_PROTOCOL_ERROR = 1002
WS_CLOSE_UNSUPPORTED_DATA = 1003
WS_CLOSE_INVALID_PAYLOAD = 1007
WS_CLOSE_POLICY_VIOLATION = 1008
WS_CLOSE_MESSAGE_TOO_BIG = 1009
WS_CLOSE_INTERNAL_ERROR = 1011


# =============================================================================
# Message Type Enums
# =============================================================================


class WSMessageType(str, Enum):
    """Incoming WebSocket message types from client."""

    PROCESS = "process"
    ANALYZE = "analyze"
    CANCEL = "cancel"
    PING = "ping"


class WSResponseType(str, Enum):
    """Outgoing WebSocket response types to client."""

    CHUNK = "chunk"
    METADATA = "metadata"
    PROGRESS = "progress"
    DONE = "done"
    ERROR = "error"
    PONG = "pong"
    CANCELLED = "cancelled"
    CONNECTED = "connected"


# =============================================================================
# Message Models
# =============================================================================


class WSProcessMessage(BaseModel):
    """Client message for processing request."""

    type: WSMessageType = Field(
        ...,
        description="Message type (must be 'process')",
    )
    task: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The task or question to process",
    )
    context: str | None = Field(
        default=None,
        max_length=10_000_000,
        description="Context string to process",
    )
    documents: list[str] | None = Field(
        default=None,
        description="List of document paths",
    )
    strategy: str | None = Field(
        default="auto",
        description="Processing strategy",
    )
    constraints: list[str] | None = Field(
        default=None,
        description="Verification constraints",
    )
    request_id: str | None = Field(
        default=None,
        description="Client-provided request ID for tracking",
    )


class WSAnalyzeMessage(BaseModel):
    """Client message for analysis request."""

    type: WSMessageType = Field(
        ...,
        description="Message type (must be 'analyze')",
    )
    context: str | None = Field(
        default=None,
        max_length=10_000_000,
        description="Context string to analyze",
    )
    documents: list[str] | None = Field(
        default=None,
        description="List of document paths",
    )
    request_id: str | None = Field(
        default=None,
        description="Client-provided request ID for tracking",
    )


class WSCancelMessage(BaseModel):
    """Client message for cancellation request."""

    type: WSMessageType = Field(
        ...,
        description="Message type (must be 'cancel')",
    )
    request_id: str = Field(
        ...,
        description="Request ID to cancel",
    )


class WSPingMessage(BaseModel):
    """Client ping message for keepalive."""

    type: WSMessageType = Field(
        ...,
        description="Message type (must be 'ping')",
    )
    timestamp: float | None = Field(
        default=None,
        description="Client timestamp for latency measurement",
    )


class WSResponse(BaseModel):
    """Standard WebSocket response format."""

    type: WSResponseType = Field(
        ...,
        description="Response type",
    )
    content: str | None = Field(
        default=None,
        description="Response content",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Response metadata",
    )
    error: str | None = Field(
        default=None,
        description="Error message if applicable",
    )
    request_id: str | None = Field(
        default=None,
        description="Associated request ID",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp",
    )


# =============================================================================
# WebSocket Manager
# =============================================================================


class WebSocketManager:
    """
    Manage WebSocket connections.

    Handles connection lifecycle, message routing, and broadcast operations.

    Features:
    - Connection tracking by unique ID
    - Active request tracking for cancellation
    - Graceful disconnection handling
    - Broadcast to all connections

    Example:
        manager = WebSocketManager()
        await manager.connect("conn-123", websocket)
        await manager.send("conn-123", {"type": "chunk", "content": "Hello"})
        await manager.disconnect("conn-123")
    """

    def __init__(self) -> None:
        """Initialize WebSocket manager."""
        self.active_connections: dict[str, WebSocket] = {}
        self.active_requests: dict[str, set[str]] = {}  # connection_id -> request_ids
        self.cancelled_requests: set[str] = set()
        self._lock = asyncio.Lock()

    async def connect(
        self,
        connection_id: str,
        websocket: WebSocket,
    ) -> None:
        """
        Accept and register a WebSocket connection.

        Args:
            connection_id: Unique connection identifier
            websocket: FastAPI WebSocket instance
        """
        await websocket.accept()

        async with self._lock:
            self.active_connections[connection_id] = websocket
            self.active_requests[connection_id] = set()

        ws_logger.info(
            "WebSocket connected",
            connection_id=connection_id,
            total_connections=len(self.active_connections),
        )

        # Send connection confirmation
        await self.send(
            connection_id,
            WSResponse(
                type=WSResponseType.CONNECTED,
                metadata={
                    "connection_id": connection_id,
                    "server_time": datetime.utcnow().isoformat(),
                },
            ).model_dump(mode="json"),
        )

    async def disconnect(self, connection_id: str) -> None:
        """
        Unregister a WebSocket connection.

        Args:
            connection_id: Connection identifier to disconnect
        """
        async with self._lock:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

            # Cancel any active requests for this connection
            if connection_id in self.active_requests:
                for request_id in self.active_requests[connection_id]:
                    self.cancelled_requests.add(request_id)
                del self.active_requests[connection_id]

        ws_logger.info(
            "WebSocket disconnected",
            connection_id=connection_id,
            total_connections=len(self.active_connections),
        )

    async def send(
        self,
        connection_id: str,
        message: dict[str, Any],
    ) -> bool:
        """
        Send a message to a specific connection.

        Args:
            connection_id: Target connection ID
            message: Message dict to send

        Returns:
            True if sent successfully, False otherwise
        """
        if connection_id not in self.active_connections:
            ws_logger.warning(
                "Attempted to send to unknown connection",
                connection_id=connection_id,
            )
            return False

        try:
            await self.active_connections[connection_id].send_json(message)
            return True
        except Exception as e:
            ws_logger.error(
                "Failed to send message",
                connection_id=connection_id,
                error=str(e),
            )
            return False

    async def send_response(
        self,
        connection_id: str,
        response: WSResponse,
    ) -> bool:
        """
        Send a WSResponse to a specific connection.

        Args:
            connection_id: Target connection ID
            response: WSResponse to send

        Returns:
            True if sent successfully
        """
        return await self.send(connection_id, response.model_dump(mode="json"))

    async def broadcast(self, message: dict[str, Any]) -> int:
        """
        Broadcast a message to all connected clients.

        Args:
            message: Message dict to broadcast

        Returns:
            Number of clients that received the message
        """
        success_count = 0

        for connection_id in list(self.active_connections.keys()):
            if await self.send(connection_id, message):
                success_count += 1

        return success_count

    def register_request(
        self,
        connection_id: str,
        request_id: str,
    ) -> None:
        """
        Register an active request for a connection.

        Args:
            connection_id: Connection ID
            request_id: Request ID to register
        """
        if connection_id in self.active_requests:
            self.active_requests[connection_id].add(request_id)

    def unregister_request(
        self,
        connection_id: str,
        request_id: str,
    ) -> None:
        """
        Unregister an active request.

        Args:
            connection_id: Connection ID
            request_id: Request ID to unregister
        """
        if connection_id in self.active_requests:
            self.active_requests[connection_id].discard(request_id)

        # Clean up from cancelled set
        self.cancelled_requests.discard(request_id)

    def cancel_request(self, request_id: str) -> bool:
        """
        Mark a request as cancelled.

        Args:
            request_id: Request ID to cancel

        Returns:
            True if request was found and cancelled
        """
        # Check if request exists
        for requests in self.active_requests.values():
            if request_id in requests:
                self.cancelled_requests.add(request_id)
                ws_logger.info("Request cancelled", request_id=request_id)
                return True

        return False

    def is_cancelled(self, request_id: str) -> bool:
        """
        Check if a request has been cancelled.

        Args:
            request_id: Request ID to check

        Returns:
            True if request is cancelled
        """
        return request_id in self.cancelled_requests

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)

    def get_connection_ids(self) -> list[str]:
        """Get list of all active connection IDs."""
        return list(self.active_connections.keys())


# =============================================================================
# Global Manager Instance
# =============================================================================

manager = WebSocketManager()


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance."""
    return manager


# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter(tags=["WebSocket"])


# =============================================================================
# Dependency for ContextFlow
# =============================================================================


async def get_contextflow_for_ws() -> ContextFlow:
    """
    Get ContextFlow instance for WebSocket handlers.

    This mirrors the server.py dependency but is separate to avoid
    circular imports.
    """
    from contextflow.api.server import get_contextflow

    return await get_contextflow()


# =============================================================================
# WebSocket Endpoints
# =============================================================================


@router.websocket("/ws/process")
async def websocket_process(
    websocket: WebSocket,
    cf: ContextFlow = Depends(get_contextflow_for_ws),
) -> None:
    """
    WebSocket endpoint for real-time processing.

    Message Format (Client -> Server):
    ```json
    {
        "type": "process",
        "task": "Summarize this document",
        "context": "...",
        "strategy": "auto",
        "request_id": "optional-client-id"
    }
    ```

    Message Format (Server -> Client):
    ```json
    {
        "type": "chunk" | "metadata" | "progress" | "done" | "error",
        "content": "...",
        "metadata": {...},
        "request_id": "..."
    }
    ```

    Supported message types:
    - process: Process task with context
    - analyze: Analyze context without execution
    - cancel: Cancel an in-progress request
    - ping: Keepalive ping (server responds with pong)
    """
    connection_id = f"ws-{uuid.uuid4().hex[:12]}"

    await manager.connect(connection_id, websocket)

    try:
        while True:
            # Receive message
            try:
                data = await websocket.receive_json()
            except Exception as e:
                ws_logger.error(
                    "Failed to receive message",
                    connection_id=connection_id,
                    error=str(e),
                )
                break

            # Validate message type
            message_type = data.get("type")

            if message_type is None:
                await manager.send_response(
                    connection_id,
                    WSResponse(
                        type=WSResponseType.ERROR,
                        error="Missing 'type' field in message",
                    ),
                )
                continue

            try:
                msg_type = WSMessageType(message_type)
            except ValueError:
                await manager.send_response(
                    connection_id,
                    WSResponse(
                        type=WSResponseType.ERROR,
                        error=f"Unknown message type: {message_type}",
                        metadata={
                            "valid_types": [t.value for t in WSMessageType],
                        },
                    ),
                )
                continue

            # Handle message based on type
            if msg_type == WSMessageType.PING:
                await _handle_ping(connection_id, data)

            elif msg_type == WSMessageType.CANCEL:
                await _handle_cancel(connection_id, data)

            elif msg_type == WSMessageType.PROCESS:
                await _handle_process(connection_id, data, cf)

            elif msg_type == WSMessageType.ANALYZE:
                await _handle_analyze(connection_id, data, cf)

    except WebSocketDisconnect:
        ws_logger.info(
            "WebSocket client disconnected",
            connection_id=connection_id,
        )

    except Exception as e:
        ws_logger.error(
            "WebSocket error",
            connection_id=connection_id,
            error=str(e),
            exc_info=True,
        )

        # Try to send error before disconnecting
        try:
            await manager.send_response(
                connection_id,
                WSResponse(
                    type=WSResponseType.ERROR,
                    error=f"Internal server error: {str(e)}",
                ),
            )
        except Exception:
            pass

    finally:
        await manager.disconnect(connection_id)


@router.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    cf: ContextFlow = Depends(get_contextflow_for_ws),
) -> None:
    """
    Simplified WebSocket endpoint for streaming only.

    This endpoint only accepts process requests and streams responses.
    Use /ws/process for full bidirectional communication.
    """
    connection_id = f"stream-{uuid.uuid4().hex[:12]}"

    await manager.connect(connection_id, websocket)

    try:
        while True:
            data = await websocket.receive_json()

            # Only handle process messages
            if data.get("type") != "process":
                await manager.send_response(
                    connection_id,
                    WSResponse(
                        type=WSResponseType.ERROR,
                        error="This endpoint only accepts 'process' messages",
                    ),
                )
                continue

            await _handle_process(connection_id, data, cf)

    except WebSocketDisconnect:
        ws_logger.info(
            "Stream client disconnected",
            connection_id=connection_id,
        )

    finally:
        await manager.disconnect(connection_id)


# =============================================================================
# Message Handlers
# =============================================================================


async def _handle_ping(
    connection_id: str,
    data: dict[str, Any],
) -> None:
    """
    Handle ping message for keepalive.

    Args:
        connection_id: Connection ID
        data: Ping message data
    """
    client_timestamp = data.get("timestamp")

    await manager.send_response(
        connection_id,
        WSResponse(
            type=WSResponseType.PONG,
            metadata={
                "client_timestamp": client_timestamp,
                "server_timestamp": datetime.utcnow().timestamp(),
            },
        ),
    )


async def _handle_cancel(
    connection_id: str,
    data: dict[str, Any],
) -> None:
    """
    Handle cancel request.

    Args:
        connection_id: Connection ID
        data: Cancel message data
    """
    try:
        msg = WSCancelMessage(**data)
    except PydanticValidationError as e:
        await manager.send_response(
            connection_id,
            WSResponse(
                type=WSResponseType.ERROR,
                error=f"Invalid cancel message: {str(e)}",
            ),
        )
        return

    cancelled = manager.cancel_request(msg.request_id)

    await manager.send_response(
        connection_id,
        WSResponse(
            type=WSResponseType.CANCELLED if cancelled else WSResponseType.ERROR,
            request_id=msg.request_id,
            metadata={"cancelled": cancelled},
            error=None if cancelled else f"Request not found: {msg.request_id}",
        ),
    )


async def _handle_process(
    connection_id: str,
    data: dict[str, Any],
    cf: ContextFlow,
) -> None:
    """
    Handle process request with streaming response.

    Args:
        connection_id: Connection ID
        data: Process message data
        cf: ContextFlow instance
    """
    # Validate message
    try:
        msg = WSProcessMessage(**data)
    except PydanticValidationError as e:
        await manager.send_response(
            connection_id,
            WSResponse(
                type=WSResponseType.ERROR,
                error=f"Invalid process message: {str(e)}",
            ),
        )
        return

    # Generate request ID if not provided
    request_id = msg.request_id or f"req-{uuid.uuid4().hex[:12]}"

    # Register request
    manager.register_request(connection_id, request_id)

    ws_logger.info(
        "Processing request",
        connection_id=connection_id,
        request_id=request_id,
        task_length=len(msg.task),
        has_context=bool(msg.context),
    )

    try:
        # Send metadata
        await manager.send_response(
            connection_id,
            WSResponse(
                type=WSResponseType.METADATA,
                request_id=request_id,
                metadata={
                    "task": msg.task[:100] + "..." if len(msg.task) > 100 else msg.task,
                    "strategy": msg.strategy or "auto",
                    "started_at": datetime.utcnow().isoformat(),
                },
            ),
        )

        # Parse strategy
        strategy = StrategyType(msg.strategy.lower() if msg.strategy else "auto")

        # Stream processing
        chunk_index = 0

        async for chunk in cf.stream(
            task=msg.task,
            documents=msg.documents,
            context=msg.context,
            strategy=strategy,
            constraints=msg.constraints,
        ):
            # Check for cancellation
            if manager.is_cancelled(request_id):
                ws_logger.info(
                    "Request cancelled mid-stream",
                    request_id=request_id,
                )
                await manager.send_response(
                    connection_id,
                    WSResponse(
                        type=WSResponseType.CANCELLED,
                        request_id=request_id,
                        metadata={"chunks_sent": chunk_index},
                    ),
                )
                return

            # Send chunk
            await manager.send_response(
                connection_id,
                WSResponse(
                    type=WSResponseType.CHUNK,
                    content=chunk,
                    request_id=request_id,
                    metadata={"chunk_index": chunk_index},
                ),
            )
            chunk_index += 1

        # Send completion
        await manager.send_response(
            connection_id,
            WSResponse(
                type=WSResponseType.DONE,
                request_id=request_id,
                metadata={
                    "total_chunks": chunk_index,
                    "completed_at": datetime.utcnow().isoformat(),
                },
            ),
        )

        ws_logger.info(
            "Request completed",
            connection_id=connection_id,
            request_id=request_id,
            total_chunks=chunk_index,
        )

    except ValidationError as e:
        await manager.send_response(
            connection_id,
            WSResponse(
                type=WSResponseType.ERROR,
                error=f"Validation error: {str(e)}",
                request_id=request_id,
            ),
        )

    except ContextFlowError as e:
        await manager.send_response(
            connection_id,
            WSResponse(
                type=WSResponseType.ERROR,
                error=f"Processing error: {str(e)}",
                request_id=request_id,
            ),
        )

    except Exception as e:
        ws_logger.error(
            "Processing failed",
            connection_id=connection_id,
            request_id=request_id,
            error=str(e),
            exc_info=True,
        )
        await manager.send_response(
            connection_id,
            WSResponse(
                type=WSResponseType.ERROR,
                error=f"Internal error: {str(e)}",
                request_id=request_id,
            ),
        )

    finally:
        manager.unregister_request(connection_id, request_id)


async def _handle_analyze(
    connection_id: str,
    data: dict[str, Any],
    cf: ContextFlow,
) -> None:
    """
    Handle analyze request.

    Args:
        connection_id: Connection ID
        data: Analyze message data
        cf: ContextFlow instance
    """
    # Validate message
    try:
        msg = WSAnalyzeMessage(**data)
    except PydanticValidationError as e:
        await manager.send_response(
            connection_id,
            WSResponse(
                type=WSResponseType.ERROR,
                error=f"Invalid analyze message: {str(e)}",
            ),
        )
        return

    # Generate request ID if not provided
    request_id = msg.request_id or f"analyze-{uuid.uuid4().hex[:12]}"

    # Register request
    manager.register_request(connection_id, request_id)

    ws_logger.info(
        "Analyzing context",
        connection_id=connection_id,
        request_id=request_id,
        has_context=bool(msg.context),
        has_documents=bool(msg.documents),
    )

    try:
        # Perform analysis
        analysis = await cf.analyze(
            documents=msg.documents,
            context=msg.context,
        )

        # Send result
        await manager.send_response(
            connection_id,
            WSResponse(
                type=WSResponseType.DONE,
                request_id=request_id,
                metadata={
                    "token_count": analysis.token_count,
                    "complexity_score": analysis.complexity_score,
                    "density_score": analysis.density_score,
                    "structure_type": analysis.structure_type,
                    "recommended_strategy": analysis.recommended_strategy.value,
                    "estimated_cost": analysis.estimated_cost,
                    "estimated_time_seconds": analysis.estimated_time_seconds,
                    "warnings": analysis.warnings,
                },
            ),
        )

        ws_logger.info(
            "Analysis completed",
            connection_id=connection_id,
            request_id=request_id,
            token_count=analysis.token_count,
        )

    except ValidationError as e:
        await manager.send_response(
            connection_id,
            WSResponse(
                type=WSResponseType.ERROR,
                error=f"Validation error: {str(e)}",
                request_id=request_id,
            ),
        )

    except Exception as e:
        ws_logger.error(
            "Analysis failed",
            connection_id=connection_id,
            request_id=request_id,
            error=str(e),
            exc_info=True,
        )
        await manager.send_response(
            connection_id,
            WSResponse(
                type=WSResponseType.ERROR,
                error=f"Analysis error: {str(e)}",
                request_id=request_id,
            ),
        )

    finally:
        manager.unregister_request(connection_id, request_id)


# =============================================================================
# Health Check Endpoint
# =============================================================================


@router.get("/ws/status")
async def websocket_status() -> dict[str, Any]:
    """
    Get WebSocket service status.

    Returns information about active connections and requests.
    """
    return {
        "active_connections": manager.get_connection_count(),
        "connection_ids": manager.get_connection_ids(),
        "cancelled_requests": len(manager.cancelled_requests),
        "timestamp": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "WSMessageType",
    "WSResponseType",
    # Models
    "WSProcessMessage",
    "WSAnalyzeMessage",
    "WSCancelMessage",
    "WSPingMessage",
    "WSResponse",
    # Manager
    "WebSocketManager",
    "manager",
    "get_websocket_manager",
    # Router
    "router",
]
