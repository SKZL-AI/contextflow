"""
MCP (Model Context Protocol) Server for ContextFlow.

This module provides MCP server implementation for integrating ContextFlow
with Claude Code and other MCP-compatible clients.

The MCP server exposes ContextFlow capabilities as tools:
- contextflow_process: Process documents with intelligent strategy selection
- contextflow_analyze: Analyze context without execution
- contextflow_search: Search in session memory

Example usage with Claude Code:
    Add to claude_desktop_config.json:
    {
        "mcpServers": {
            "contextflow": {
                "command": "python",
                "args": ["-m", "contextflow.mcp"]
            }
        }
    }

Example programmatic usage:
    from contextflow.mcp import ContextFlowMCPServer

    server = ContextFlowMCPServer()
    await server.run()
"""

from __future__ import annotations

from contextflow.mcp.server import ContextFlowMCPServer
from contextflow.mcp.tools import (
    CONTEXTFLOW_TOOLS,
    contextflow_analyze,
    contextflow_process,
    contextflow_search,
)

__all__ = [
    # Main Server
    "ContextFlowMCPServer",
    # Tool Functions
    "contextflow_process",
    "contextflow_analyze",
    "contextflow_search",
    # Tool Definitions
    "CONTEXTFLOW_TOOLS",
]
