"""
Entry point for running ContextFlow MCP server as a module.

Usage:
    python -m contextflow.mcp
    python -m contextflow.mcp --transport http --port 8765
    python -m contextflow.mcp --help
"""

from contextflow.mcp.server import run_server

if __name__ == "__main__":
    run_server()
