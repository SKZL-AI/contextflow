"""
Serve command for ContextFlow CLI.

Starts the ContextFlow REST API server.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
error_console = Console(stderr=True)


def serve(
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Host to bind to"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to listen on"),
    ] = 8000,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Number of worker processes"),
    ] = 1,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Enable auto-reload for development"),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", "-l", help="Log level: debug, info, warning, error"),
    ] = "info",
    cors_origins: Annotated[
        str | None,
        typer.Option("--cors-origins", help="Comma-separated CORS origins"),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", envvar="CONTEXTFLOW_API_KEY", help="API key for authentication"),
    ] = None,
    ssl_keyfile: Annotated[
        str | None,
        typer.Option("--ssl-keyfile", help="SSL key file path"),
    ] = None,
    ssl_certfile: Annotated[
        str | None,
        typer.Option("--ssl-certfile", help="SSL certificate file path"),
    ] = None,
) -> None:
    """
    Start the ContextFlow REST API server.

    Launches a FastAPI server that provides REST endpoints for
    processing, analysis, and session management.

    Examples:
        contextflow serve
        contextflow serve --host 0.0.0.0 --port 8080
        contextflow serve --reload --log-level debug
        contextflow serve --workers 4 --host 0.0.0.0
    """
    # Display startup info
    table = Table(title="ContextFlow Server", show_header=False, box=None)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Host", host)
    table.add_row("Port", str(port))
    table.add_row("Workers", str(workers))
    table.add_row("Reload", "[green]Enabled[/green]" if reload else "[dim]Disabled[/dim]")
    table.add_row("Log Level", log_level.upper())

    if cors_origins:
        table.add_row("CORS Origins", cors_origins)
    if api_key:
        table.add_row("API Key", "[dim]****[/dim]")
    if ssl_keyfile and ssl_certfile:
        table.add_row("SSL", "[green]Enabled[/green]")

    protocol = "https" if ssl_keyfile and ssl_certfile else "http"
    url = f"{protocol}://{host}:{port}"

    console.print()
    console.print(table)
    console.print()
    console.print(
        Panel(
            f"[bold green]Server starting at:[/bold green] {url}\n"
            f"[dim]API Docs:[/dim] {url}/docs\n"
            f"[dim]Health:[/dim] {url}/health",
            title="[bold blue]ContextFlow[/bold blue]",
            expand=False,
        )
    )
    console.print()

    try:
        import uvicorn

        # Configure uvicorn
        uvicorn_config = {
            "app": "contextflow.api.server:app",
            "host": host,
            "port": port,
            "workers": workers if not reload else 1,  # Reload requires single worker
            "reload": reload,
            "log_level": log_level.lower(),
        }

        # Add SSL if configured
        if ssl_keyfile and ssl_certfile:
            uvicorn_config["ssl_keyfile"] = ssl_keyfile
            uvicorn_config["ssl_certfile"] = ssl_certfile

        # Run server
        uvicorn.run(**uvicorn_config)

    except ImportError:
        error_console.print(
            "[red]Error:[/red] uvicorn is required for the server. "
            "Install with: pip install uvicorn"
        )
        raise typer.Exit(code=1)

    except Exception as e:
        error_console.print(f"[red]Error:[/red] Failed to start server: {e}")
        raise typer.Exit(code=1)


__all__ = ["serve"]
