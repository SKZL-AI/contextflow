"""
Main ContextFlow CLI application.

Provides the entry point for the contextflow command-line interface
with subcommands for processing, analysis, and serving.
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from contextflow.cli.commands import analyze, process, serve

# =============================================================================
# CLI Application Setup
# =============================================================================

app = typer.Typer(
    name="contextflow",
    help="ContextFlow CLI - Intelligent LLM Context Orchestration",
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=True,
    pretty_exceptions_show_locals=False,
)

console = Console()
error_console = Console(stderr=True)

# Add subcommand typers
app.add_typer(process.app, name="process")
app.add_typer(analyze.app, name="analyze")

# Add serve as a direct command
app.command()(serve.serve)


# =============================================================================
# Version and Info Commands
# =============================================================================


def _get_version() -> str:
    """Get package version."""
    try:
        from contextflow import __version__

        return __version__
    except ImportError:
        return "0.1.0"


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]ContextFlow[/bold blue] version [green]{_get_version()}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """
    ContextFlow - Intelligent LLM Context Orchestration.

    Process large documents with automatic strategy selection,
    verification, and multi-provider support.

    Strategies:
      - GSD: Get Stuff Done - Direct processing for small contexts (<10K tokens)
      - RALPH: Relevance-Anchored Linear Processing Heuristic - Structured for medium (10K-100K)
      - RLM: Reasoning through Large Memories - Full RAG pipeline for large (>100K)

    Examples:
        contextflow process "Summarize this" doc.txt
        contextflow analyze doc.txt --output json
        contextflow serve --host 0.0.0.0 --port 8000
    """
    pass


# =============================================================================
# Additional Commands
# =============================================================================


@app.command()
def info() -> None:
    """Display system and configuration information."""
    import platform

    table = Table(title="ContextFlow System Info", show_header=False, box=None)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", _get_version())
    table.add_row("Python", platform.python_version())
    table.add_row("Platform", platform.platform())

    # Check available providers
    providers = []
    try:
        from contextflow.providers.factory import list_providers

        providers = list_providers()
    except ImportError:
        providers = ["(unable to detect)"]

    table.add_row("Providers", ", ".join(providers) if providers else "None detected")

    # Check configuration
    try:
        from contextflow.core.config import get_config

        config = get_config()
        table.add_row("Default Provider", config.default_provider)
        table.add_row("Config Loaded", "[green]Yes[/green]")
    except Exception as e:
        table.add_row("Config Loaded", f"[red]No[/red] ({str(e)[:50]})")

    console.print()
    console.print(table)
    console.print()


@app.command()
def providers() -> None:
    """List available LLM providers and their status."""
    table = Table(title="Available Providers", show_header=True, header_style="bold cyan")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Models")
    table.add_column("Max Context")

    try:
        from contextflow.providers.factory import is_provider_available, list_providers

        for provider_name in list_providers():
            try:
                available = is_provider_available(provider_name)
                status = "[green]Available[/green]" if available else "[yellow]Configured[/yellow]"
                table.add_row(provider_name, status, "-", "-")
            except Exception:
                table.add_row(provider_name, "[yellow]Unknown[/yellow]", "-", "-")

    except ImportError:
        # Fallback with known providers
        known_providers = [
            ("claude", "Anthropic Claude"),
            ("openai", "OpenAI GPT"),
            ("ollama", "Ollama Local"),
            ("vllm", "vLLM Server"),
            ("groq", "Groq API"),
            ("gemini", "Google Gemini"),
            ("mistral", "Mistral AI"),
        ]
        for name, desc in known_providers:
            table.add_row(name, "[dim]Not loaded[/dim]", desc, "-")

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Use --provider/-p with process/analyze to specify a provider.[/dim]")


@app.command()
def strategies() -> None:
    """Display information about processing strategies."""
    console.print()

    # GSD
    console.print(
        Panel(
            "[bold]GSD (Get Stuff Done)[/bold]\n\n"
            "Direct, single-pass processing for small contexts.\n\n"
            "[cyan]Token Range:[/cyan] < 10,000 tokens\n"
            "[cyan]Use Case:[/cyan] Quick summaries, simple Q&A, short documents\n"
            "[cyan]Speed:[/cyan] [green]Fast[/green] (single LLM call)",
            title="[blue]GSD Strategy[/blue]",
            expand=False,
        )
    )
    console.print()

    # RALPH
    console.print(
        Panel(
            "[bold]RALPH (Relevance-Anchored Linear Processing Heuristic)[/bold]\n\n"
            "Structured, multi-pass processing for medium contexts.\n\n"
            "[cyan]Token Range:[/cyan] 10,000 - 100,000 tokens\n"
            "[cyan]Use Case:[/cyan] Document analysis, code review, detailed summaries\n"
            "[cyan]Speed:[/cyan] [yellow]Moderate[/yellow] (multiple coordinated calls)",
            title="[blue]RALPH Strategy[/blue]",
            expand=False,
        )
    )
    console.print()

    # RLM
    console.print(
        Panel(
            "[bold]RLM (Reasoning through Large Memories)[/bold]\n\n"
            "Full RAG pipeline with sub-agents for large contexts.\n\n"
            "[cyan]Token Range:[/cyan] > 100,000 tokens\n"
            "[cyan]Use Case:[/cyan] Large codebases, multi-document analysis, research\n"
            "[cyan]Speed:[/cyan] [red]Slower[/red] (RAG indexing + parallel processing)",
            title="[blue]RLM Strategy[/blue]",
            expand=False,
        )
    )
    console.print()

    console.print("[dim]Use --strategy/-s auto to let ContextFlow choose automatically.[/dim]")


@app.command()
def config(
    show_all: Annotated[
        bool,
        typer.Option("--all", "-a", help="Show all configuration values"),
    ] = False,
) -> None:
    """Display current configuration."""
    try:
        from contextflow.core.config import get_config

        config = get_config()

        table = Table(title="ContextFlow Configuration", show_header=True, header_style="bold cyan")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        # Core settings
        table.add_row("Default Provider", config.default_provider)

        # Strategy settings
        table.add_row("GSD Max Tokens", f"{config.strategy.gsd_max_tokens:,}")
        table.add_row("RALPH Max Tokens", f"{config.strategy.ralph_max_tokens:,}")
        table.add_row("RLM Min Tokens", f"{config.strategy.rlm_min_tokens:,}")

        if show_all:
            # RAG settings
            table.add_row("RAG Chunk Size", str(config.rag.chunk_size))
            table.add_row("RAG Chunk Overlap", str(config.rag.chunk_overlap))
            table.add_row("RAG Top K", str(config.rag.top_k))

            # Provider configs (without sensitive data)
            if hasattr(config, "claude") and config.claude:
                table.add_row("Claude Model", config.claude.model or "default")
            if hasattr(config, "openai") and config.openai:
                table.add_row("OpenAI Model", config.openai.model or "default")

        console.print()
        console.print(table)
        console.print()

        if not show_all:
            console.print("[dim]Use --all to see full configuration.[/dim]")

    except Exception as e:
        error_console.print(f"[red]Error loading configuration:[/red] {e}")
        raise typer.Exit(code=1)


# =============================================================================
# Entry Point
# =============================================================================


def cli() -> None:
    """Entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        error_console.print(f"[red]Unexpected error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()


__all__ = ["app", "cli", "main"]
