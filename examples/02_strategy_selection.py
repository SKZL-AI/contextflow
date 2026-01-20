#!/usr/bin/env python
"""
Strategy Selection Example.

This example demonstrates how ContextFlow selects and executes
different strategies based on context size and complexity:
- AUTO: Automatic strategy selection
- GSD_DIRECT: For small contexts (<10K tokens)
- RALPH_STRUCTURED: For medium contexts (10K-100K tokens)
- RLM_FULL: For large contexts (>100K tokens)

Prerequisites:
    - Set ANTHROPIC_API_KEY environment variable
    - Install contextflow: pip install -e .

Run:
    python examples/02_strategy_selection.py
"""

import asyncio
import os
from pathlib import Path

# Add parent to path for development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextflow import ContextFlow, StrategyType
from contextflow.core.router import StrategyRouter, RouterConfig


# =============================================================================
# Sample Contexts of Different Sizes
# =============================================================================


def generate_small_context() -> str:
    """Generate small context (~500 tokens) - GSD territory."""
    return """
    # Project README

    This is a simple Python utility for processing text files.

    ## Installation
    pip install text-processor

    ## Usage
    from text_processor import process
    result = process("input.txt")

    ## Features
    - Fast processing
    - Multiple file formats
    - CLI support
    """


def generate_medium_context() -> str:
    """Generate medium context (~15K tokens) - RALPH territory."""
    # Simulate a medium-sized document by repeating content
    base_text = """
    ## Chapter {n}: System Architecture

    This chapter discusses the architectural patterns used in modern
    software systems. We explore microservices, event-driven design,
    and domain-driven development.

    ### Microservices

    Microservices architecture decomposes applications into small,
    independent services that communicate over well-defined APIs.
    Each service is responsible for a specific business capability.

    Key principles include:
    - Single responsibility
    - Independence and isolation
    - Decentralized data management
    - Infrastructure automation

    ### Event-Driven Architecture

    Events represent state changes in the system. Services publish
    events when something significant happens, and other services
    subscribe to events they care about.

    Benefits include:
    - Loose coupling between services
    - Better scalability
    - Audit trail of all changes
    - Easier integration

    ### Domain-Driven Design

    DDD focuses on the core domain and domain logic. It emphasizes
    collaboration between technical and domain experts to create
    a shared understanding of the problem space.

    Key concepts:
    - Bounded contexts
    - Ubiquitous language
    - Aggregates and entities
    - Domain events

    """

    # Generate ~15K tokens (roughly 60KB of text)
    chapters = [base_text.format(n=i) for i in range(1, 40)]
    return "\n\n".join(chapters)


def generate_large_context() -> str:
    """Generate large context (~120K tokens) - RLM territory."""
    # Simulate a very large document
    medium = generate_medium_context()
    # Repeat to create ~120K tokens
    return "\n\n---\n\n".join([medium for _ in range(8)])


# =============================================================================
# Example Functions
# =============================================================================


async def automatic_strategy_selection() -> None:
    """
    Demonstrate automatic strategy selection.

    ContextFlow analyzes the context and chooses the best strategy.
    """
    print("\n" + "=" * 60)
    print("Example 1: Automatic Strategy Selection (AUTO)")
    print("=" * 60)

    contexts = [
        ("Small (~500 tokens)", generate_small_context()),
        ("Medium (~15K tokens)", generate_medium_context()),
        # Note: Large context would require significant API calls
    ]

    async with ContextFlow() as cf:
        for name, context in contexts:
            print(f"\n--- Processing {name} ---")

            # Analyze first to see recommendation
            analysis = await cf.analyze(context=context)
            print(f"  Token Count: {analysis.token_count:,}")
            print(f"  Recommended Strategy: {analysis.recommended_strategy.value}")

            # Process with AUTO (let ContextFlow decide)
            result = await cf.process(
                task="Provide a brief summary of the main topics covered",
                context=context,
                strategy=StrategyType.AUTO,
            )

            print(f"  Strategy Used: {result.strategy_used.value}")
            print(f"  Execution Time: {result.execution_time:.2f}s")
            print(f"  Summary: {result.answer[:150]}...")


async def manual_strategy_selection() -> None:
    """
    Demonstrate manual strategy selection.

    Override automatic selection to force a specific strategy.
    """
    print("\n" + "=" * 60)
    print("Example 2: Manual Strategy Selection")
    print("=" * 60)

    context = generate_small_context()

    async with ContextFlow() as cf:
        # Force GSD strategy
        print("\n--- Forcing GSD_DIRECT ---")
        result_gsd = await cf.process(
            task="Summarize this document",
            context=context,
            strategy=StrategyType.GSD_DIRECT,
        )
        print(f"  Strategy: {result_gsd.strategy_used.value}")
        print(f"  Tokens: {result_gsd.total_tokens}")
        print(f"  Time: {result_gsd.execution_time:.2f}s")

        # Force RALPH strategy (even on small context)
        print("\n--- Forcing RALPH_STRUCTURED (on small context) ---")
        result_ralph = await cf.process(
            task="Summarize this document",
            context=context,
            strategy=StrategyType.RALPH_STRUCTURED,
        )
        print(f"  Strategy: {result_ralph.strategy_used.value}")
        print(f"  Tokens: {result_ralph.total_tokens}")
        print(f"  Time: {result_ralph.execution_time:.2f}s")

        # Compare results
        print("\n--- Comparison ---")
        print(f"GSD used {result_gsd.total_tokens} tokens in {result_gsd.execution_time:.2f}s")
        print(f"RALPH used {result_ralph.total_tokens} tokens in {result_ralph.execution_time:.2f}s")
        print("(GSD is typically more efficient for small contexts)")


async def strategy_router_example() -> None:
    """
    Demonstrate direct use of StrategyRouter.

    Shows how to use the router independently for analysis.
    """
    print("\n" + "=" * 60)
    print("Example 3: Using StrategyRouter Directly")
    print("=" * 60)

    from contextflow.providers.factory import get_provider

    # Create provider
    provider = get_provider("claude")

    # Create router with custom thresholds
    router = StrategyRouter(
        provider=provider,
        config=RouterConfig(
            gsd_max_tokens=10000,      # Default: 10K
            ralph_max_tokens=100000,   # Default: 100K
            rlm_min_tokens=100000,     # Default: 100K
        ),
    )

    contexts = [
        ("Tiny", "Hello world"),
        ("Small", generate_small_context()),
        ("Medium", generate_medium_context()[:10000]),  # Truncate for demo
    ]

    for name, context in contexts:
        # Analyze without executing
        analysis = router.analyze(
            task="Summarize this content",
            context=context,
            constraints=["Be concise"],
        )

        print(f"\n--- {name} Context ---")
        print(f"  Tokens: {analysis.token_count:,}")
        print(f"  Complexity: {analysis.complexity.value}")
        print(f"  Estimated Density: {analysis.estimated_density:.2f}")
        print(f"  Recommended: {analysis.recommended_strategy.value}")
        print(f"  Alternatives: {[s.value for s in analysis.alternative_strategies]}")
        print(f"  Reasoning: {analysis.reasoning}")


async def complexity_based_selection() -> None:
    """
    Demonstrate complexity-based strategy selection.

    Shows how content complexity affects strategy choice.
    """
    print("\n" + "=" * 60)
    print("Example 4: Complexity-Based Selection")
    print("=" * 60)

    # Simple content (low complexity)
    simple_content = """
    The cat sat on the mat.
    The dog ran in the park.
    The bird flew over the tree.
    """ * 100

    # Complex content (high complexity) - technical document
    complex_content = """
    The implementation of the Byzantine fault-tolerant consensus algorithm
    requires careful consideration of the asynchronous network model and
    the partial synchrony assumptions. The protocol must handle malicious
    nodes that can behave arbitrarily, including sending conflicting messages
    to different nodes.

    Key invariants that must be maintained:
    1. Safety: No two honest nodes decide on different values
    2. Liveness: All honest nodes eventually decide
    3. Validity: If all honest nodes propose the same value, they decide it

    The protocol proceeds in rounds, where each round consists of:
    - PRE-PREPARE phase: Leader proposes a value
    - PREPARE phase: Nodes vote on the proposal
    - COMMIT phase: Nodes commit if 2f+1 PREPARE messages received
    """ * 50

    async with ContextFlow() as cf:
        # Analyze simple content
        simple_analysis = await cf.analyze(context=simple_content)
        print(f"\n--- Simple Content ---")
        print(f"  Tokens: {simple_analysis.token_count:,}")
        print(f"  Complexity: {simple_analysis.complexity_score:.2f}")
        print(f"  Density: {simple_analysis.density_score:.2f}")
        print(f"  Recommended: {simple_analysis.recommended_strategy.value}")

        # Analyze complex content
        complex_analysis = await cf.analyze(context=complex_content)
        print(f"\n--- Complex Content ---")
        print(f"  Tokens: {complex_analysis.token_count:,}")
        print(f"  Complexity: {complex_analysis.complexity_score:.2f}")
        print(f"  Density: {complex_analysis.density_score:.2f}")
        print(f"  Recommended: {complex_analysis.recommended_strategy.value}")


async def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ContextFlow Strategy Selection Examples")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nWarning: ANTHROPIC_API_KEY not set.")
        print("Set the environment variable to run these examples.")
        return

    try:
        await automatic_strategy_selection()
        await manual_strategy_selection()
        await strategy_router_example()
        await complexity_based_selection()

        print("\n" + "=" * 60)
        print("All strategy selection examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
