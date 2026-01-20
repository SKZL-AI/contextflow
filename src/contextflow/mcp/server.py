"""
MCP Server Implementation for ContextFlow.

This module implements the Model Context Protocol (MCP) server that exposes
ContextFlow capabilities to MCP-compatible clients like Claude Code.

The server supports:
- Tool calls for processing, analysis, and search
- Resource exposure for session context
- stdio and HTTP transports

Based on MCP specification: https://spec.modelcontextprotocol.io/
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from contextflow.mcp.tools import (
    CONTEXTFLOW_TOOLS,
    AnalysisResult,
    ProcessResult,
    SearchResultItem,
    get_tool_handler,
    list_tool_names,
)
from contextflow.utils.errors import ContextFlowError, ValidationError
from contextflow.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# MCP Protocol Types (Compatible Interface)
# =============================================================================


@dataclass
class MCPTool:
    """MCP Tool definition."""

    name: str
    description: str
    inputSchema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }


@dataclass
class MCPResource:
    """MCP Resource definition."""

    uri: str
    name: str
    description: str | None = None
    mimeType: str = "application/json"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "uri": self.uri,
            "name": self.name,
            "mimeType": self.mimeType,
        }
        if self.description:
            result["description"] = self.description
        return result


@dataclass
class MCPTextContent:
    """MCP Text content block."""

    type: str = "text"
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"type": self.type, "text": self.text}


@dataclass
class MCPToolResult:
    """MCP Tool execution result."""

    content: list[MCPTextContent]
    isError: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": [c.to_dict() for c in self.content],
            "isError": self.isError,
        }


@dataclass
class MCPError:
    """MCP Error response."""

    code: int
    message: str
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {"code": self.code, "message": self.message}
        if self.data:
            result["data"] = self.data
        return result


# MCP Error Codes
class MCPErrorCodes:
    """Standard MCP error codes."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


# =============================================================================
# Server Configuration
# =============================================================================


@dataclass
class MCPServerConfig:
    """Configuration for MCP server."""

    name: str = "contextflow"
    version: str = "1.0.0"
    description: str = "ContextFlow AI - Intelligent LLM Context Orchestration"
    enable_resources: bool = True
    enable_prompts: bool = False  # Not implemented yet
    default_provider: str = "claude"
    max_request_size: int = 10_000_000  # 10MB
    request_timeout: float = 300.0  # 5 minutes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "enable_resources": self.enable_resources,
            "enable_prompts": self.enable_prompts,
            "default_provider": self.default_provider,
        }


# =============================================================================
# Main MCP Server
# =============================================================================


class ContextFlowMCPServer:
    """
    MCP Server for ContextFlow integration with Claude Code.

    This server implements the Model Context Protocol to expose
    ContextFlow capabilities as MCP tools. It supports both stdio
    and HTTP transports.

    Features:
    - Tool execution for process, analyze, and search operations
    - Resource exposure for session context
    - Session management for persistent context
    - Structured logging and error handling

    Example:
        # Run with stdio (for Claude Code)
        server = ContextFlowMCPServer()
        await server.run()

        # Run with HTTP
        server = ContextFlowMCPServer()
        await server.run(transport="http", port=8765)

        # Programmatic tool call
        result = await server.call_tool(
            "contextflow_process",
            {"task": "Summarize", "context": "..."}
        )
    """

    def __init__(
        self,
        config: MCPServerConfig | None = None,
    ) -> None:
        """
        Initialize the MCP server.

        Args:
            config: Server configuration (uses defaults if None)
        """
        self.config = config or MCPServerConfig()
        self._contextflow: Any | None = None
        self._sessions: dict[str, dict[str, Any]] = {}
        self._initialized = False
        self._start_time = datetime.utcnow()
        self._request_count = 0
        self._tool_calls: dict[str, int] = {}

        logger.info(
            "MCP Server initialized",
            name=self.config.name,
            version=self.config.version,
        )

    # =========================================================================
    # Initialization
    # =========================================================================

    async def initialize(self) -> None:
        """
        Initialize ContextFlow instance and components.

        Called lazily on first request or explicitly before running.
        """
        if self._initialized:
            return

        logger.info("Initializing ContextFlow for MCP server")

        try:
            from contextflow.core.orchestrator import ContextFlow

            self._contextflow = ContextFlow(
                provider=self.config.default_provider,
            )
            await self._contextflow.initialize()
            self._initialized = True

            logger.info("ContextFlow initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize ContextFlow", error=str(e))
            raise ContextFlowError(
                f"Failed to initialize ContextFlow: {str(e)}",
                details={"error_type": type(e).__name__},
            )

    async def _ensure_initialized(self) -> None:
        """Ensure server is initialized before handling requests."""
        if not self._initialized:
            await self.initialize()

    # =========================================================================
    # MCP Protocol Handlers
    # =========================================================================

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Handle an incoming MCP JSON-RPC request.

        Args:
            request: JSON-RPC request object

        Returns:
            JSON-RPC response object
        """
        self._request_count += 1
        request_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        logger.debug(
            "Handling MCP request",
            method=method,
            request_id=request_id,
        )

        try:
            # Route to appropriate handler
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "tools/list":
                result = await self._handle_list_tools()
            elif method == "tools/call":
                result = await self._handle_call_tool(params)
            elif method == "resources/list":
                result = await self._handle_list_resources()
            elif method == "resources/read":
                result = await self._handle_read_resource(params)
            elif method == "ping":
                result = {"pong": True}
            else:
                return self._error_response(
                    request_id,
                    MCPErrorCodes.METHOD_NOT_FOUND,
                    f"Unknown method: {method}",
                )

            return self._success_response(request_id, result)

        except ValidationError as e:
            logger.warning("Validation error", error=str(e))
            return self._error_response(
                request_id,
                MCPErrorCodes.INVALID_PARAMS,
                str(e),
                e.details,
            )

        except ContextFlowError as e:
            logger.error("ContextFlow error", error=str(e))
            return self._error_response(
                request_id,
                MCPErrorCodes.INTERNAL_ERROR,
                str(e),
                e.details,
            )

        except Exception as e:
            logger.error("Unexpected error", error=str(e), exc_info=True)
            return self._error_response(
                request_id,
                MCPErrorCodes.INTERNAL_ERROR,
                f"Internal error: {str(e)}",
            )

    async def _handle_initialize(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle initialize request."""
        await self._ensure_initialized()

        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": (
                    {"subscribe": False, "listChanged": False}
                    if self.config.enable_resources
                    else None
                ),
            },
            "serverInfo": {
                "name": self.config.name,
                "version": self.config.version,
            },
        }

    async def _handle_list_tools(self) -> dict[str, Any]:
        """Handle tools/list request."""
        tools = [
            MCPTool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.input_schema,
            ).to_dict()
            for tool in CONTEXTFLOW_TOOLS
        ]

        return {"tools": tools}

    async def _handle_call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call request."""
        await self._ensure_initialized()

        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        # Track tool calls
        self._tool_calls[tool_name] = self._tool_calls.get(tool_name, 0) + 1

        logger.info(
            "Calling tool",
            tool_name=tool_name,
            arguments_keys=list(arguments.keys()),
        )

        # Execute the tool
        result = await self.call_tool(tool_name, arguments)
        return result.to_dict()

    async def _handle_list_resources(self) -> dict[str, Any]:
        """Handle resources/list request."""
        if not self.config.enable_resources:
            return {"resources": []}

        resources = []

        # Current session resource
        resources.append(
            MCPResource(
                uri="contextflow://session/current",
                name="Current Session",
                description="Current ContextFlow session context and state",
                mimeType="application/json",
            ).to_dict()
        )

        # Server stats resource
        resources.append(
            MCPResource(
                uri="contextflow://server/stats",
                name="Server Statistics",
                description="MCP server statistics and metrics",
                mimeType="application/json",
            ).to_dict()
        )

        return {"resources": resources}

    async def _handle_read_resource(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle resources/read request."""
        uri = params.get("uri", "")

        if uri == "contextflow://session/current":
            content = self._get_session_resource()
        elif uri == "contextflow://server/stats":
            content = self._get_stats_resource()
        else:
            raise ValidationError(f"Unknown resource URI: {uri}")

        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(content, indent=2, default=str),
                }
            ]
        }

    # =========================================================================
    # Tool Execution
    # =========================================================================

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPToolResult:
        """
        Execute an MCP tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            MCPToolResult with execution result

        Raises:
            ValidationError: If tool not found or arguments invalid
        """
        start_time = time.time()

        # Get tool handler
        handler = get_tool_handler(name)
        if handler is None:
            available = list_tool_names()
            return MCPToolResult(
                content=[
                    MCPTextContent(
                        text=f"Unknown tool: {name}. Available: {available}"
                    )
                ],
                isError=True,
            )

        try:
            # Inject ContextFlow instance
            arguments["contextflow_instance"] = self._contextflow

            # Execute tool
            result = await handler(**arguments)

            execution_time = time.time() - start_time

            # Format result based on type
            if isinstance(result, ProcessResult):
                return self._format_process_result(result, execution_time)
            elif isinstance(result, AnalysisResult):
                return self._format_analysis_result(result, execution_time)
            elif isinstance(result, list):  # Search results
                return self._format_search_results(result, execution_time)
            else:
                # Generic result
                return MCPToolResult(
                    content=[
                        MCPTextContent(
                            text=json.dumps(result, indent=2, default=str)
                        )
                    ],
                    isError=False,
                )

        except ValidationError as e:
            logger.warning("Tool validation error", tool=name, error=str(e))
            return MCPToolResult(
                content=[MCPTextContent(text=f"Validation error: {e.message}")],
                isError=True,
            )

        except ContextFlowError as e:
            logger.error("Tool execution error", tool=name, error=str(e))
            return MCPToolResult(
                content=[MCPTextContent(text=f"Error: {e.message}")],
                isError=True,
            )

        except Exception as e:
            logger.error(
                "Unexpected tool error",
                tool=name,
                error=str(e),
                exc_info=True,
            )
            return MCPToolResult(
                content=[MCPTextContent(text=f"Unexpected error: {str(e)}")],
                isError=True,
            )

    def _format_process_result(
        self,
        result: ProcessResult,
        execution_time: float,
    ) -> MCPToolResult:
        """Format ProcessResult for MCP response."""
        if result.success:
            # Build formatted response
            text_parts = [
                f"## Result\n\n{result.answer}",
                f"\n\n---\n\n**Strategy:** {result.strategy_used}",
                f"**Execution Time:** {execution_time:.2f}s",
                f"**Tokens Used:** {result.token_usage.get('total_tokens', 0):,}",
                f"**Verification:** {'Passed' if result.verification_passed else 'Failed'} ({result.verification_score:.0%})",
            ]

            if result.warnings:
                text_parts.append(f"**Warnings:** {', '.join(result.warnings)}")

            return MCPToolResult(
                content=[MCPTextContent(text="\n".join(text_parts))],
                isError=False,
            )
        else:
            return MCPToolResult(
                content=[MCPTextContent(text=f"Processing failed: {result.answer}")],
                isError=True,
            )

    def _format_analysis_result(
        self,
        result: AnalysisResult,
        execution_time: float,
    ) -> MCPToolResult:
        """Format AnalysisResult for MCP response."""
        text_parts = [
            "## Context Analysis\n",
            f"**Token Count:** {result.token_count:,}",
            f"**Complexity:** {result.complexity} ({result.complexity_score:.0%})",
            f"**Information Density:** {result.density:.0%}",
            f"**Structure Type:** {result.structure_type}",
            f"**Recommended Strategy:** {result.recommended_strategy}",
            f"**Estimated Time:** {result.estimated_time:.1f}s",
            "\n### Estimated Costs by Provider",
        ]

        for provider, cost in result.estimated_costs.items():
            text_parts.append(f"- {provider}: ${cost:.4f}")

        if result.chunk_suggestion:
            cs = result.chunk_suggestion
            text_parts.extend([
                "\n### Chunking Recommendation",
                f"- Strategy: {cs.get('strategy', 'N/A')}",
                f"- Chunk Size: {cs.get('chunk_size', 'N/A')} tokens",
                f"- Overlap: {cs.get('overlap', 'N/A')} tokens",
                f"- Estimated Chunks: {cs.get('estimated_chunks', 'N/A')}",
            ])
            if cs.get("rationale"):
                text_parts.append(f"- Rationale: {cs['rationale']}")

        if result.warnings:
            text_parts.append(f"\n**Warnings:** {', '.join(result.warnings)}")

        return MCPToolResult(
            content=[MCPTextContent(text="\n".join(text_parts))],
            isError=False,
        )

    def _format_search_results(
        self,
        results: list[SearchResultItem],
        execution_time: float,
    ) -> MCPToolResult:
        """Format search results for MCP response."""
        if not results:
            return MCPToolResult(
                content=[MCPTextContent(text="No results found.")],
                isError=False,
            )

        text_parts = [
            f"## Search Results ({len(results)} found)\n",
        ]

        for i, result in enumerate(results, 1):
            text_parts.extend([
                f"### Result {i} (Score: {result.score:.2f})",
                f"**Chunk ID:** {result.chunk_id}",
                f"```\n{result.content[:500]}{'...' if len(result.content) > 500 else ''}\n```",
                "",
            ])

        text_parts.append(f"\n*Search completed in {execution_time:.2f}s*")

        return MCPToolResult(
            content=[MCPTextContent(text="\n".join(text_parts))],
            isError=False,
        )

    # =========================================================================
    # Resource Handlers
    # =========================================================================

    def _get_session_resource(self) -> dict[str, Any]:
        """Get current session resource data."""
        if self._contextflow is None:
            return {"status": "not_initialized", "sessions": []}

        return {
            "status": "active" if self._initialized else "not_initialized",
            "current_session": (
                self._contextflow._current_session.id
                if hasattr(self._contextflow, "_current_session")
                and self._contextflow._current_session
                else None
            ),
            "stats": self._contextflow.stats if self._initialized else {},
        }

    def _get_stats_resource(self) -> dict[str, Any]:
        """Get server statistics resource data."""
        uptime = (datetime.utcnow() - self._start_time).total_seconds()

        return {
            "server": {
                "name": self.config.name,
                "version": self.config.version,
                "uptime_seconds": uptime,
                "initialized": self._initialized,
            },
            "requests": {
                "total": self._request_count,
                "tool_calls": self._tool_calls,
            },
            "contextflow": (
                self._contextflow.stats if self._contextflow else None
            ),
        }

    # =========================================================================
    # Response Helpers
    # =========================================================================

    def _success_response(
        self,
        request_id: str | int | None,
        result: Any,
    ) -> dict[str, Any]:
        """Build a successful JSON-RPC response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    def _error_response(
        self,
        request_id: str | int | None,
        code: int,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build an error JSON-RPC response."""
        error = MCPError(code=code, message=message, data=data)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error.to_dict(),
        }

    # =========================================================================
    # Transport Handlers
    # =========================================================================

    async def run(
        self,
        transport: str = "stdio",
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        """
        Run the MCP server.

        Args:
            transport: Transport type ("stdio" or "http")
            host: Host for HTTP transport
            port: Port for HTTP transport

        Example:
            # stdio for Claude Code
            await server.run()

            # HTTP for testing
            await server.run(transport="http", port=8765)
        """
        logger.info(
            "Starting MCP server",
            transport=transport,
            host=host if transport == "http" else None,
            port=port if transport == "http" else None,
        )

        if transport == "stdio":
            await self._run_stdio()
        elif transport == "http":
            await self._run_http(host, port)
        else:
            raise ValueError(f"Unknown transport: {transport}")

    async def _run_stdio(self) -> None:
        """Run server using stdio transport."""
        logger.info("MCP server running on stdio")

        # Read from stdin, write to stdout
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)

        await asyncio.get_event_loop().connect_read_pipe(
            lambda: protocol, sys.stdin
        )

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(
            writer_transport, writer_protocol, reader, asyncio.get_event_loop()
        )

        try:
            while True:
                # Read line from stdin
                line = await reader.readline()
                if not line:
                    break

                try:
                    # Parse JSON-RPC request
                    request = json.loads(line.decode("utf-8").strip())

                    # Handle request
                    response = await self.handle_request(request)

                    # Write response
                    response_json = json.dumps(response) + "\n"
                    writer.write(response_json.encode("utf-8"))
                    await writer.drain()

                except json.JSONDecodeError as e:
                    # Invalid JSON
                    error_response = self._error_response(
                        None,
                        MCPErrorCodes.PARSE_ERROR,
                        f"Invalid JSON: {str(e)}",
                    )
                    writer.write(
                        (json.dumps(error_response) + "\n").encode("utf-8")
                    )
                    await writer.drain()

        except asyncio.CancelledError:
            logger.info("MCP server stopped")
            raise

        finally:
            writer.close()

    async def _run_http(self, host: str, port: int) -> None:
        """Run server using HTTP transport."""
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError(
                "aiohttp is required for HTTP transport. "
                "Install with: pip install aiohttp"
            )

        async def handle_post(request: web.Request) -> web.Response:
            """Handle POST request."""
            try:
                body = await request.json()
                response = await self.handle_request(body)
                return web.json_response(response)
            except json.JSONDecodeError:
                error_response = self._error_response(
                    None,
                    MCPErrorCodes.PARSE_ERROR,
                    "Invalid JSON",
                )
                return web.json_response(error_response, status=400)

        app = web.Application()
        app.router.add_post("/", handle_post)
        app.router.add_post("/mcp", handle_post)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        logger.info(f"MCP server running on http://{host}:{port}")

        # Keep running until cancelled
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await runner.cleanup()
            raise

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def close(self) -> None:
        """Close the server and release resources."""
        logger.info("Closing MCP server")

        if self._contextflow is not None:
            await self._contextflow.close()
            self._contextflow = None

        self._initialized = False
        self._sessions.clear()

    async def __aenter__(self) -> ContextFlowMCPServer:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()


# =============================================================================
# CLI Entry Point
# =============================================================================


async def main() -> None:
    """Main entry point for running MCP server from command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ContextFlow MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with stdio (for Claude Code)
  python -m contextflow.mcp

  # Run with HTTP transport
  python -m contextflow.mcp --transport http --port 8765

  # Run with custom provider
  python -m contextflow.mcp --provider openai
        """,
    )

    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for HTTP transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for HTTP transport (default: 8765)",
    )
    parser.add_argument(
        "--provider",
        default="claude",
        help="Default LLM provider (default: claude)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        from contextflow.utils.logging import setup_logging
        setup_logging(level="DEBUG")

    # Create and run server
    config = MCPServerConfig(default_provider=args.provider)
    server = ContextFlowMCPServer(config=config)

    try:
        await server.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
        )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    finally:
        await server.close()


def run_server() -> None:
    """Synchronous entry point for running MCP server."""
    asyncio.run(main())


if __name__ == "__main__":
    run_server()


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Main Server
    "ContextFlowMCPServer",
    # Configuration
    "MCPServerConfig",
    # Protocol Types
    "MCPTool",
    "MCPResource",
    "MCPTextContent",
    "MCPToolResult",
    "MCPError",
    "MCPErrorCodes",
    # Entry Points
    "main",
    "run_server",
]
