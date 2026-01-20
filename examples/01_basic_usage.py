#!/usr/bin/env python
"""
Basic ContextFlow Usage Example.

This example demonstrates the fundamental usage patterns of ContextFlow:
- Creating a ContextFlow instance
- Processing text with automatic strategy selection
- Using the context manager pattern
- Analyzing context before processing
- Accessing result metadata

Prerequisites:
    - Set ANTHROPIC_API_KEY environment variable (or other provider key)
    - Install contextflow: pip install -e .

Run:
    python examples/01_basic_usage.py
"""

import asyncio
import os
from pathlib import Path

# Add parent to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextflow import ContextFlow, StrategyType, ProcessResult


# =============================================================================
# Sample Data
# =============================================================================

SAMPLE_DOCUMENT = """
# ContextFlow Framework Overview

ContextFlow is an intelligent LLM context orchestration framework designed
to handle large-scale document processing efficiently.

## Key Features

1. **Automatic Strategy Selection**: Chooses optimal processing strategy
   based on context size (GSD, RALPH, or RLM).

2. **Multi-Provider Support**: Works with Claude, OpenAI, Ollama, and more.

3. **Verification Protocol**: Built-in output verification for quality assurance.

4. **RAG Integration**: In-memory FAISS-based retrieval for large documents.

## Architecture

The framework consists of several core components:

- **Orchestrator**: Main entry point that coordinates all operations
- **StrategyRouter**: Selects and executes the appropriate strategy
- **VerificationProtocol**: Ensures output quality meets requirements
- **SessionManager**: Tracks processing sessions and observations

## Use Cases

ContextFlow excels at:
- Document summarization
- Code analysis and review
- Research paper synthesis
- Multi-document question answering
"""


# =============================================================================
# Example Functions
# =============================================================================


async def basic_processing_example() -> None:
    """
    Basic processing with ContextFlow.

    Shows the simplest way to use ContextFlow to process text.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Processing")
    print("=" * 60)

    # Create ContextFlow with default provider (claude)
    # Uses ANTHROPIC_API_KEY from environment
    async with ContextFlow() as cf:
        # Process with automatic strategy selection
        result: ProcessResult = await cf.process(
            task="Summarize the key features and components of this framework",
            context=SAMPLE_DOCUMENT,
            strategy=StrategyType.AUTO,  # Let ContextFlow choose
        )

        print(f"\nTask: Summarize the document")
        print(f"Strategy Used: {result.strategy_used.value}")
        print(f"Tokens Used: {result.total_tokens}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        print(f"\nSummary:\n{result.answer}")

    # Expected output:
    # Strategy Used: gsd_direct (for small context)
    # Summary: [AI-generated summary of the document]


async def context_analysis_example() -> None:
    """
    Analyze context before processing.

    Shows how to analyze context to understand complexity and
    get strategy recommendations.
    """
    print("\n" + "=" * 60)
    print("Example 2: Context Analysis")
    print("=" * 60)

    async with ContextFlow() as cf:
        # Analyze context without processing
        analysis = await cf.analyze(
            context=SAMPLE_DOCUMENT,
            task="Summarize this document",
        )

        print(f"\nContext Analysis Results:")
        print(f"  Token Count: {analysis.token_count}")
        print(f"  Complexity Score: {analysis.complexity_score:.2f}")
        print(f"  Density Score: {analysis.density_score:.2f}")
        print(f"  Structure Type: {analysis.structure_type}")
        print(f"  Recommended Strategy: {analysis.recommended_strategy.value}")
        print(f"  Estimated Cost: ${analysis.estimated_cost:.4f}")
        print(f"  Estimated Time: {analysis.estimated_time_seconds:.1f}s")

        if analysis.warnings:
            print(f"  Warnings: {analysis.warnings}")

    # Expected output:
    # Token Count: ~400 tokens
    # Recommended Strategy: gsd_direct (small context)


async def process_with_constraints_example() -> None:
    """
    Process with verification constraints.

    Shows how to add constraints that the output must satisfy.
    """
    print("\n" + "=" * 60)
    print("Example 3: Processing with Constraints")
    print("=" * 60)

    async with ContextFlow() as cf:
        result = await cf.process(
            task="List the main components of the framework",
            context=SAMPLE_DOCUMENT,
            constraints=[
                "Include at least 4 components",
                "Format as a numbered list",
                "Keep response under 200 words",
            ],
        )

        print(f"\nTask: List main components with constraints")
        print(f"Strategy: {result.strategy_used.value}")
        print(f"Verification Passed: {result.metadata.get('verification_passed', 'N/A')}")
        print(f"\nResult:\n{result.answer}")

    # Expected output:
    # A numbered list of framework components


async def manual_lifecycle_example() -> None:
    """
    Manual initialization and cleanup.

    Shows how to use ContextFlow without the context manager,
    giving more control over the lifecycle.
    """
    print("\n" + "=" * 60)
    print("Example 4: Manual Lifecycle Management")
    print("=" * 60)

    # Create instance
    cf = ContextFlow(provider="claude")

    try:
        # Explicitly initialize
        await cf.initialize()
        print(f"Initialized: {cf.is_initialized}")
        print(f"Provider: {cf.provider.name}")

        # Process
        result = await cf.process(
            task="What is ContextFlow used for?",
            context=SAMPLE_DOCUMENT,
        )

        print(f"\nResult: {result.answer[:200]}...")

        # Check stats
        stats = cf.stats
        print(f"\nStats:")
        print(f"  Provider: {stats['provider']}")
        print(f"  Executions: {stats['execution_count']}")
        print(f"  Total Tokens: {stats['total_tokens_used']}")
        print(f"  Total Cost: ${stats['total_cost_usd']:.6f}")

    finally:
        # Always cleanup
        await cf.close()
        print("\nClosed ContextFlow instance")


async def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ContextFlow Basic Usage Examples")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nWarning: ANTHROPIC_API_KEY not set.")
        print("Set the environment variable to run these examples:")
        print("  export ANTHROPIC_API_KEY=your-key-here")
        print("\nRunning in demo mode with simulated output...\n")
        return

    try:
        await basic_processing_example()
        await context_analysis_example()
        await process_with_constraints_example()
        await manual_lifecycle_example()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
