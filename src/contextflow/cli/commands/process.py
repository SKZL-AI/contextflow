"""
Process command for ContextFlow CLI.

Handles document processing with intelligent strategy selection.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Import directly from models to avoid triggering server import
try:
    from contextflow.api.models import CLIProcessArgs
except ImportError:
    # Fallback: define locally if models not available

    from pydantic import BaseModel, Field

    class CLIProcessArgs(BaseModel):
        """CLI process command arguments."""
        task: str = Field(..., description="Task to process")
        files: List[str] = Field(default_factory=list, description="Input files")
        context: Optional[str] = Field(default=None, description="Direct context")
        strategy: str = Field(default="auto", description="Processing strategy")
        provider: Optional[str] = Field(default=None, description="LLM provider")
        output: Optional[str] = Field(default=None, description="Output file path")
        format: str = Field(default="text", description="Output format")
        verbose: bool = Field(default=False, description="Verbose output")
        stream: bool = Field(default=False, description="Stream output")

app = typer.Typer(help="Process documents with ContextFlow")
console = Console()
error_console = Console(stderr=True)


def _format_output(
    result: dict,
    output_format: str,
    verbose: bool = False,
) -> str:
    """Format output based on format type."""
    if output_format == "json":
        return json.dumps(result, indent=2, default=str)
    elif output_format == "markdown":
        parts = [
            f"# Result\n\n{result.get('answer', '')}",
            "",
            "## Metadata",
            f"- **Strategy**: {result.get('strategy_used', 'N/A')}",
            f"- **Tokens**: {result.get('total_tokens', 0)}",
            f"- **Cost**: ${result.get('total_cost', 0):.4f}",
            f"- **Time**: {result.get('execution_time', 0):.2f}s",
        ]
        if verbose and result.get("trajectory"):
            parts.append("\n## Trajectory")
            for step in result.get("trajectory", []):
                parts.append(f"- {step.get('step_type', 'N/A')}")
        return "\n".join(parts)
    else:
        # Plain text
        return result.get("answer", str(result))


def _print_result_panel(
    result: dict,
    verbose: bool = False,
) -> None:
    """Print result in a formatted panel."""
    answer = result.get("answer", "No answer generated")

    # Create metadata table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Strategy", result.get("strategy_used", "N/A"))
    table.add_row("Tokens", str(result.get("total_tokens", 0)))
    table.add_row("Cost", f"${result.get('total_cost', 0):.4f}")
    table.add_row("Time", f"{result.get('execution_time', 0):.2f}s")

    verification_passed = result.get("metadata", {}).get("verification_passed", True)
    verification_score = result.get("metadata", {}).get("verification_score", 1.0)
    status = "[green]Passed[/green]" if verification_passed else "[red]Failed[/red]"
    table.add_row("Verification", f"{status} ({verification_score:.2f})")

    # Print answer
    console.print(Panel(answer, title="[bold blue]Answer[/bold blue]", expand=False))
    console.print()
    console.print(table)

    # Print trajectory if verbose
    if verbose and result.get("trajectory"):
        console.print()
        console.print("[bold]Execution Trajectory:[/bold]")
        for step in result.get("trajectory", []):
            step_type = step.get("step_type", "unknown")
            tokens = step.get("tokens_used", 0)
            console.print(f"  [dim]->[/dim] {step_type} [dim](tokens: {tokens})[/dim]")

    # Print warnings
    warnings = result.get("warnings", [])
    if warnings:
        console.print()
        console.print("[yellow bold]Warnings:[/yellow bold]")
        for warning in warnings:
            console.print(f"  [yellow]! {warning}[/yellow]")


async def _process_async(
    args: CLIProcessArgs,
    no_verify: bool = False,
) -> dict:
    """Execute processing asynchronously."""
    from contextflow.core.orchestrator import ContextFlow
    from contextflow.core.types import StrategyType

    # Create ContextFlow instance
    cf = ContextFlow(provider=args.provider)

    try:
        await cf.initialize()

        # Load context from files or direct context
        context = args.context
        documents = args.files if args.files else None

        # Parse strategy
        strategy_map = {
            "auto": StrategyType.AUTO,
            "gsd": StrategyType.GSD_DIRECT,
            "gsd_direct": StrategyType.GSD_DIRECT,
            "ralph": StrategyType.RALPH_STRUCTURED,
            "ralph_structured": StrategyType.RALPH_STRUCTURED,
            "rlm": StrategyType.RLM_FULL,
            "rlm_full": StrategyType.RLM_FULL,
        }
        strategy = strategy_map.get(args.strategy.lower(), StrategyType.AUTO)

        # Process
        result = await cf.process(
            task=args.task,
            documents=documents,
            context=context,
            strategy=strategy,
        )

        # Convert result to dict
        return {
            "answer": result.answer,
            "strategy_used": result.strategy_used.value,
            "total_tokens": result.total_tokens,
            "total_cost": result.total_cost,
            "execution_time": result.execution_time,
            "trajectory": [
                {
                    "step_type": s.step_type,
                    "tokens_used": s.tokens_used,
                    "cost_usd": s.cost_usd,
                }
                for s in result.trajectory
            ],
            "warnings": result.warnings,
            "metadata": result.metadata or {},
        }

    finally:
        await cf.close()


async def _stream_async(
    args: CLIProcessArgs,
) -> None:
    """Execute streaming processing."""
    from contextflow.core.orchestrator import ContextFlow
    from contextflow.core.types import StrategyType

    cf = ContextFlow(provider=args.provider)

    try:
        await cf.initialize()

        context = args.context
        documents = args.files if args.files else None

        strategy_map = {
            "auto": StrategyType.AUTO,
            "gsd": StrategyType.GSD_DIRECT,
            "ralph": StrategyType.RALPH_STRUCTURED,
            "rlm": StrategyType.RLM_FULL,
        }
        strategy = strategy_map.get(args.strategy.lower(), StrategyType.AUTO)

        # Stream output
        async for chunk in cf.stream(
            task=args.task,
            documents=documents,
            context=context,
            strategy=strategy,
        ):
            console.print(chunk, end="")

        console.print()  # Final newline

    finally:
        await cf.close()


@app.callback(invoke_without_command=True)
def process(
    task: Annotated[
        str,
        typer.Argument(help="Task or question to process"),
    ],
    documents: Annotated[
        Optional[List[Path]],
        typer.Argument(help="Document files to process"),
    ] = None,
    context: Annotated[
        Optional[str],
        typer.Option("--context", "-c", help="Direct context string"),
    ] = None,
    strategy: Annotated[
        str,
        typer.Option("--strategy", "-s", help="Strategy: auto, gsd, ralph, rlm"),
    ] = "auto",
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-p", help="LLM provider to use"),
    ] = None,
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output format: text, json, markdown"),
    ] = "text",
    output_file: Annotated[
        Optional[Path],
        typer.Option("--output-file", "-O", help="Write output to file"),
    ] = None,
    stream: Annotated[
        bool,
        typer.Option("--stream", help="Enable streaming output"),
    ] = False,
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify", help="Disable verification"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """
    Process documents with ContextFlow.

    Examples:
        contextflow process "Summarize this document" doc.txt
        contextflow process "Extract key points" --context "inline context"
        contextflow process "Analyze code" src/*.py --strategy ralph
    """
    # Validate inputs
    if documents is None and context is None:
        error_console.print(
            "[red]Error:[/red] Either documents or --context must be provided"
        )
        raise typer.Exit(code=1)

    # Validate strategy
    valid_strategies = {"auto", "gsd", "gsd_direct", "ralph", "ralph_structured", "rlm", "rlm_full"}
    if strategy.lower() not in valid_strategies:
        error_console.print(
            f"[red]Error:[/red] Invalid strategy '{strategy}'. "
            f"Valid: {', '.join(sorted(valid_strategies))}"
        )
        raise typer.Exit(code=1)

    # Build args
    file_paths = [str(p) for p in documents] if documents else []
    args = CLIProcessArgs(
        task=task,
        files=file_paths,
        context=context,
        strategy=strategy,
        provider=provider,
        format=output,
        verbose=verbose,
        stream=stream,
    )

    if verbose:
        console.print(f"[dim]Task:[/dim] {task}")
        if documents:
            console.print(f"[dim]Documents:[/dim] {len(documents)} file(s)")
        console.print(f"[dim]Strategy:[/dim] {strategy}")
        console.print(f"[dim]Provider:[/dim] {provider or 'default'}")
        console.print()

    try:
        if stream:
            # Streaming mode
            with console.status("[bold green]Processing with streaming...") as status:
                asyncio.run(_stream_async(args))
        else:
            # Standard mode with progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task_id = progress.add_task("Processing...", total=None)
                result = asyncio.run(_process_async(args, no_verify=no_verify))
                progress.update(task_id, completed=True)

            # Format and output result
            if output == "text" and not output_file:
                _print_result_panel(result, verbose=verbose)
            else:
                formatted = _format_output(result, output, verbose=verbose)
                if output_file:
                    output_file.write_text(formatted, encoding="utf-8")
                    console.print(f"[green]Output written to:[/green] {output_file}")
                else:
                    console.print(formatted)

    except Exception as e:
        error_console.print(f"[red]Error:[/red] {e}")
        if verbose:
            import traceback
            error_console.print(traceback.format_exc())
        raise typer.Exit(code=1)


__all__ = ["app", "process"]
