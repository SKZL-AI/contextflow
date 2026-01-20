"""
ContextFlow Command-Line Interface.

Provides the CLI entry point for interacting with ContextFlow
from the command line.

Commands:
    process  - Process documents with intelligent strategy selection
    analyze  - Analyze context and get recommendations
    serve    - Start the REST API server
    info     - Display system information
    providers - List available LLM providers
    strategies - Display strategy information
    config   - Show current configuration

Example:
    $ contextflow process "Summarize this document" doc.txt
    $ contextflow analyze large_file.txt --output json
    $ contextflow serve --port 8080
"""

from __future__ import annotations

from contextflow.cli.main import app, cli, main

__all__ = [
    "app",
    "cli",
    "main",
]
