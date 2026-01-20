#!/usr/bin/env python
"""
Streaming Output Example.

This example demonstrates ContextFlow's streaming capabilities:
- Real-time token streaming
- Async iteration over stream chunks
- Progress tracking during streaming
- Combining streaming with verification

Prerequisites:
    - Set ANTHROPIC_API_KEY environment variable
    - Install contextflow: pip install -e .

Run:
    python examples/03_streaming.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextflow import ContextFlow, StrategyType


# =============================================================================
# Sample Data
# =============================================================================

ANALYSIS_DOCUMENT = """
# Quarterly Performance Report Q4 2024

## Executive Summary

This quarter marked significant growth across all business units.
Revenue increased by 23% year-over-year, driven primarily by our
cloud services division and international expansion.

## Financial Highlights

- Total Revenue: $45.2M (up 23% YoY)
- Operating Income: $12.1M (up 18% YoY)
- Net Profit Margin: 26.8%
- Cash Position: $89.4M

## Product Performance

### Cloud Services
- 340,000 active subscribers (up 45%)
- Average revenue per user: $89/month
- Churn rate reduced to 2.1%

### Enterprise Solutions
- 156 new enterprise clients
- Contract value up 34%
- Implementation time reduced by 20%

### Consumer Products
- 2.3M units shipped
- Customer satisfaction: 4.6/5
- Return rate: 1.8%

## Strategic Initiatives

1. AI Integration: Launched AI-powered analytics
2. Global Expansion: Entered 12 new markets
3. Partnership Program: 45 new partners onboarded
4. Sustainability: 50% carbon reduction achieved

## Outlook

Q1 2025 projections indicate continued momentum with expected
revenue growth of 18-22%. Key focus areas include expanding
our AI capabilities and deepening market penetration in APAC.
"""


# =============================================================================
# Example Functions
# =============================================================================


async def basic_streaming_example() -> None:
    """
    Basic streaming with async iteration.

    Demonstrates real-time output as the LLM generates tokens.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Streaming")
    print("=" * 60)
    print("\nStreaming response (watch tokens appear):\n")
    print("-" * 40)

    async with ContextFlow() as cf:
        # Use stream() method for streaming output
        full_response = ""
        token_count = 0

        async for chunk in cf.stream(
            task="Provide a detailed analysis of the financial performance",
            context=ANALYSIS_DOCUMENT,
            strategy=StrategyType.AUTO,
        ):
            # Print each chunk as it arrives
            print(chunk, end="", flush=True)
            full_response += chunk
            token_count += 1

        print("\n" + "-" * 40)
        print(f"\nStreaming complete!")
        print(f"Total chunks received: {token_count}")
        print(f"Total characters: {len(full_response)}")


async def streaming_with_progress() -> None:
    """
    Streaming with progress tracking.

    Shows how to track progress during streaming operations.
    """
    print("\n" + "=" * 60)
    print("Example 2: Streaming with Progress Tracking")
    print("=" * 60)

    async with ContextFlow() as cf:
        start_time = time.time()
        chunks_received = 0
        bytes_received = 0

        print("\nProgress:")

        async for chunk in cf.stream(
            task="List the top 5 strategic priorities based on this report",
            context=ANALYSIS_DOCUMENT,
        ):
            chunks_received += 1
            bytes_received += len(chunk.encode('utf-8'))

            # Update progress every 10 chunks
            if chunks_received % 10 == 0:
                elapsed = time.time() - start_time
                rate = bytes_received / elapsed if elapsed > 0 else 0
                print(f"\r  Chunks: {chunks_received} | "
                      f"Bytes: {bytes_received} | "
                      f"Rate: {rate:.0f} B/s", end="", flush=True)

        elapsed = time.time() - start_time
        print(f"\n\nCompleted in {elapsed:.2f}s")
        print(f"Final: {chunks_received} chunks, {bytes_received} bytes")


async def streaming_to_buffer() -> None:
    """
    Streaming to a buffer with post-processing.

    Demonstrates collecting streamed output for further processing.
    """
    print("\n" + "=" * 60)
    print("Example 3: Streaming to Buffer")
    print("=" * 60)

    async with ContextFlow() as cf:
        # Collect chunks into a buffer
        buffer: list[str] = []
        word_count = 0

        print("\nCollecting stream into buffer...")

        async for chunk in cf.stream(
            task="Summarize the key metrics from each division",
            context=ANALYSIS_DOCUMENT,
        ):
            buffer.append(chunk)
            # Count words as they come in
            word_count += len(chunk.split())

        # Join buffer to get complete response
        complete_response = "".join(buffer)

        print(f"\nBuffer Statistics:")
        print(f"  Chunks collected: {len(buffer)}")
        print(f"  Total characters: {len(complete_response)}")
        print(f"  Estimated words: {word_count}")

        # Post-process the complete response
        print(f"\n--- Complete Response ---")
        print(complete_response)
        print(f"--- End Response ---")


async def streaming_with_timeout() -> None:
    """
    Streaming with timeout handling.

    Shows how to handle timeouts during streaming.
    """
    print("\n" + "=" * 60)
    print("Example 4: Streaming with Timeout")
    print("=" * 60)

    async with ContextFlow() as cf:
        chunks_collected: list[str] = []
        timeout_seconds = 30  # Max wait time

        print(f"\nStreaming with {timeout_seconds}s timeout...")

        try:
            async with asyncio.timeout(timeout_seconds):
                async for chunk in cf.stream(
                    task="Provide executive talking points for the board meeting",
                    context=ANALYSIS_DOCUMENT,
                ):
                    chunks_collected.append(chunk)
                    print(chunk, end="", flush=True)

            print(f"\n\nCompleted successfully!")

        except asyncio.TimeoutError:
            print(f"\n\nTimeout after {timeout_seconds}s")
            partial_response = "".join(chunks_collected)
            print(f"Partial response ({len(partial_response)} chars):")
            print(partial_response[:500] + "..." if len(partial_response) > 500 else partial_response)


async def streaming_callback_example() -> None:
    """
    Streaming with callback-style processing.

    Demonstrates processing each chunk with a callback.
    """
    print("\n" + "=" * 60)
    print("Example 5: Streaming with Callbacks")
    print("=" * 60)

    # Statistics tracker
    stats = {
        "chunks": 0,
        "chars": 0,
        "sentences": 0,
    }

    def process_chunk(chunk: str) -> None:
        """Callback to process each chunk."""
        stats["chunks"] += 1
        stats["chars"] += len(chunk)
        stats["sentences"] += chunk.count(".")

    async with ContextFlow() as cf:
        print("\nProcessing chunks with callback...")

        full_text = ""
        async for chunk in cf.stream(
            task="Analyze the growth trends mentioned in the report",
            context=ANALYSIS_DOCUMENT,
        ):
            process_chunk(chunk)
            full_text += chunk

            # Real-time stats update
            if stats["chunks"] % 20 == 0:
                print(f"\r  Processed: {stats['chunks']} chunks, "
                      f"{stats['chars']} chars, "
                      f"{stats['sentences']} sentences", end="", flush=True)

        print(f"\n\nFinal Statistics:")
        print(f"  Total Chunks: {stats['chunks']}")
        print(f"  Total Characters: {stats['chars']}")
        print(f"  Sentences Detected: {stats['sentences']}")
        print(f"\n--- Response ---")
        print(full_text)


async def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ContextFlow Streaming Examples")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nWarning: ANTHROPIC_API_KEY not set.")
        print("Set the environment variable to run these examples.")
        return

    try:
        await basic_streaming_example()
        await streaming_with_progress()
        await streaming_to_buffer()
        await streaming_with_timeout()
        await streaming_callback_example()

        print("\n" + "=" * 60)
        print("All streaming examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
