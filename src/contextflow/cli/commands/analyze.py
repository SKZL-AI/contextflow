"""
Analyze command for ContextFlow CLI.

Analyzes context without processing to provide recommendations.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import directly from models to avoid triggering server import
try:
    from contextflow.api.models import CLIAnalyzeArgs
except ImportError:
    # Fallback: define locally if models not available

    from pydantic import BaseModel, Field

    class CLIAnalyzeArgs(BaseModel):
        """CLI analyze command arguments."""

        files: list[str] = Field(default_factory=list, description="Files to analyze")
        context: str | None = Field(default=None, description="Direct context")
        format: str = Field(default="text", description="Output format")
        verbose: bool = Field(default=False, description="Verbose output")


app = typer.Typer(help="Analyze context and get recommendations")
console = Console()
error_console = Console(stderr=True)


def _format_analysis_text(analysis: dict, verbose: bool = False) -> str:
    """Format analysis as plain text."""
    lines = [
        f"Token Count: {analysis.get('token_count', 0):,}",
        f"Complexity: {analysis.get('complexity_score', 0):.2f}",
        f"Density: {analysis.get('density_score', 0):.2f}",
        f"Structure: {analysis.get('structure_type', 'unknown')}",
        f"Recommended Strategy: {analysis.get('recommended_strategy', 'auto')}",
        f"Estimated Cost: ${analysis.get('estimated_cost', 0):.4f}",
        f"Estimated Time: {analysis.get('estimated_time_seconds', 0):.1f}s",
    ]

    if verbose and analysis.get("metadata"):
        meta = analysis["metadata"]
        if meta.get("reasoning"):
            lines.append(f"\nReasoning: {meta['reasoning']}")
        if meta.get("alternatives"):
            lines.append(f"Alternatives: {', '.join(meta['alternatives'])}")

    warnings = analysis.get("warnings", [])
    if warnings:
        lines.append("\nWarnings:")
        for w in warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def _print_analysis_panel(analysis: dict, verbose: bool = False) -> None:
    """Print analysis in a formatted panel."""
    # Main metrics table
    table = Table(title="Context Analysis", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    token_count = analysis.get("token_count", 0)
    table.add_row("Token Count", f"{token_count:,}")

    complexity = analysis.get("complexity_score", 0)
    complexity_color = "green" if complexity < 0.4 else "yellow" if complexity < 0.7 else "red"
    table.add_row("Complexity", f"[{complexity_color}]{complexity:.2f}[/{complexity_color}]")

    density = analysis.get("density_score", 0)
    table.add_row("Density", f"{density:.2f}")

    table.add_row("Structure", analysis.get("structure_type", "unknown"))

    strategy = analysis.get("recommended_strategy", "auto")
    if hasattr(strategy, "value"):
        strategy = strategy.value
    table.add_row("Recommended Strategy", f"[bold]{strategy}[/bold]")

    table.add_row("Estimated Cost", f"${analysis.get('estimated_cost', 0):.4f}")
    table.add_row("Estimated Time", f"{analysis.get('estimated_time_seconds', 0):.1f}s")

    console.print(table)

    # Chunk suggestion if available
    metadata = analysis.get("metadata", {})
    chunk_suggestion = metadata.get("chunk_suggestion")
    if chunk_suggestion and verbose:
        console.print()
        chunk_table = Table(title="Chunking Suggestion", show_header=False, box=None)
        chunk_table.add_column("Key", style="dim")
        chunk_table.add_column("Value")

        chunk_table.add_row("Strategy", chunk_suggestion.get("strategy", "N/A"))
        chunk_table.add_row("Chunk Size", str(chunk_suggestion.get("chunk_size", "N/A")))
        chunk_table.add_row("Overlap", str(chunk_suggestion.get("overlap", 0)))
        chunk_table.add_row("Est. Chunks", str(chunk_suggestion.get("estimated_chunks", "N/A")))
        if chunk_suggestion.get("rationale"):
            chunk_table.add_row("Rationale", chunk_suggestion["rationale"])

        console.print(chunk_table)

    # Reasoning
    if verbose and metadata.get("reasoning"):
        console.print()
        console.print(
            Panel(
                metadata["reasoning"],
                title="[bold]Reasoning[/bold]",
                expand=False,
            )
        )

    # Alternatives
    if verbose and metadata.get("alternatives"):
        console.print()
        console.print("[bold]Alternative Strategies:[/bold]")
        for alt in metadata["alternatives"]:
            console.print(f"  [dim]-[/dim] {alt}")

    # Warnings
    warnings = analysis.get("warnings", [])
    if warnings:
        console.print()
        console.print("[yellow bold]Warnings:[/yellow bold]")
        for warning in warnings:
            console.print(f"  [yellow]! {warning}[/yellow]")


async def _analyze_async(args: CLIAnalyzeArgs) -> dict:
    """Execute analysis asynchronously."""
    from contextflow.core.orchestrator import ContextFlow

    cf = ContextFlow()

    try:
        await cf.initialize()

        # Build inputs
        documents = args.files if args.files else None
        context = args.context

        # Analyze
        analysis = await cf.analyze(
            documents=documents,
            context=context,
            use_llm=args.verbose,  # Use LLM for verbose analysis
        )

        # Convert to dict
        strategy_value = analysis.recommended_strategy
        if hasattr(strategy_value, "value"):
            strategy_value = strategy_value.value

        return {
            "token_count": analysis.token_count,
            "complexity_score": analysis.complexity_score,
            "density_score": analysis.density_score,
            "structure_type": analysis.structure_type,
            "recommended_strategy": strategy_value,
            "estimated_cost": analysis.estimated_cost,
            "estimated_time_seconds": analysis.estimated_time_seconds,
            "warnings": analysis.warnings,
            "metadata": analysis.metadata or {},
        }

    finally:
        await cf.close()


@app.callback(invoke_without_command=True)
def analyze(
    documents: Annotated[
        list[Path] | None,
        typer.Argument(help="Document files to analyze"),
    ] = None,
    context: Annotated[
        str | None,
        typer.Option("--context", "-c", help="Direct context string to analyze"),
    ] = None,
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output format: text, json"),
    ] = "text",
    output_file: Annotated[
        Path | None,
        typer.Option("--output-file", "-O", help="Write output to file"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output with LLM analysis"),
    ] = False,
) -> None:
    """
    Analyze context and get processing recommendations.

    Provides token count, complexity analysis, and strategy recommendations
    without actually processing the content.

    Examples:
        contextflow analyze doc.txt
        contextflow analyze --context "some text" --output json
        contextflow analyze *.py --verbose
    """
    # Validate inputs
    if documents is None and context is None:
        error_console.print("[red]Error:[/red] Either documents or --context must be provided")
        raise typer.Exit(code=1)

    # Build args
    file_paths = [str(p) for p in documents] if documents else []
    args = CLIAnalyzeArgs(
        files=file_paths,
        context=context,
        format=output,
        verbose=verbose,
    )

    if verbose:
        console.print("[dim]Analyzing context...[/dim]")
        if documents:
            console.print(f"[dim]Files:[/dim] {len(documents)}")
        console.print()

    try:
        with console.status("[bold green]Analyzing..."):
            result = asyncio.run(_analyze_async(args))

        # Format and output result
        if output == "json":
            formatted = json.dumps(result, indent=2, default=str)
            if output_file:
                output_file.write_text(formatted, encoding="utf-8")
                console.print(f"[green]Output written to:[/green] {output_file}")
            else:
                console.print(formatted)
        else:
            if output_file:
                formatted = _format_analysis_text(result, verbose=verbose)
                output_file.write_text(formatted, encoding="utf-8")
                console.print(f"[green]Output written to:[/green] {output_file}")
            else:
                _print_analysis_panel(result, verbose=verbose)

    except Exception as e:
        error_console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback

            error_console.print(traceback.format_exc())
        raise typer.Exit(code=1)


__all__ = ["app", "analyze"]
