#!/usr/bin/env python
"""
Multi-Provider Example.

This example demonstrates ContextFlow's multi-provider support:
- Switching between providers (Claude, OpenAI, Ollama, etc.)
- Provider-specific configuration
- Provider capabilities and features
- Fallback strategies
- Custom provider registration

Supported Providers:
    - claude: Anthropic Claude (requires ANTHROPIC_API_KEY)
    - openai: OpenAI GPT (requires OPENAI_API_KEY)
    - ollama: Local Ollama (no API key needed)
    - vllm: vLLM server (no API key needed)
    - groq: Groq API (requires GROQ_API_KEY)
    - gemini: Google Gemini (requires GOOGLE_API_KEY)
    - mistral: Mistral AI (requires MISTRAL_API_KEY)

Prerequisites:
    - Set appropriate API keys for providers you want to use
    - Install contextflow: pip install -e .

Run:
    python examples/07_providers.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextflow import ContextFlow, ContextFlowConfig
from contextflow.providers.factory import (
    get_provider,
    list_providers,
    is_provider_available,
    register_provider,
    ProviderFactory,
)
from contextflow.providers.base import BaseProvider


# =============================================================================
# Sample Data
# =============================================================================

SAMPLE_TEXT = """
# Machine Learning Pipeline

This document describes our ML pipeline for customer churn prediction.

## Data Processing
- Ingest data from PostgreSQL warehouse
- Feature engineering with 50+ features
- Train/test split with temporal validation

## Model Training
- Algorithm: XGBoost with hyperparameter tuning
- Cross-validation: 5-fold time-series CV
- Metrics: AUC-ROC, Precision@K, Recall@K

## Deployment
- Model serialized with joblib
- Served via FastAPI endpoint
- A/B testing framework for gradual rollout
"""


# =============================================================================
# Example Functions
# =============================================================================


async def list_available_providers() -> None:
    """
    List all available providers.

    Shows which providers are registered and potentially usable.
    """
    print("\n" + "=" * 60)
    print("Example 1: Available Providers")
    print("=" * 60)

    providers = list_providers()
    print(f"\nRegistered providers ({len(providers)}):")

    for name in providers:
        # Check for API key availability
        key_var_map = {
            "claude": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "ollama": None,  # No key needed
            "vllm": None,    # No key needed
        }

        key_var = key_var_map.get(name)
        if key_var is None:
            status = "Available (no key needed)"
        elif os.environ.get(key_var):
            status = "Ready (key found)"
        else:
            status = f"Missing {key_var}"

        print(f"  - {name}: {status}")


async def switch_providers_example() -> None:
    """
    Switch between different providers.

    Shows how to use different providers for the same task.
    """
    print("\n" + "=" * 60)
    print("Example 2: Switching Between Providers")
    print("=" * 60)

    # Determine which providers are available
    available = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        available.append("claude")
    if os.environ.get("OPENAI_API_KEY"):
        available.append("openai")
    if os.environ.get("GROQ_API_KEY"):
        available.append("groq")

    if not available:
        print("\nNo providers configured with API keys.")
        print("Set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY")
        return

    task = "Summarize the ML pipeline in 2-3 sentences"

    for provider_name in available:
        print(f"\n--- Using {provider_name.upper()} ---")

        try:
            async with ContextFlow(provider=provider_name) as cf:
                result = await cf.process(
                    task=task,
                    context=SAMPLE_TEXT,
                )

                print(f"Strategy: {result.strategy_used.value}")
                print(f"Tokens: {result.total_tokens}")
                print(f"Cost: ${result.total_cost:.6f}")
                print(f"Answer: {result.answer}")

        except Exception as e:
            print(f"Error with {provider_name}: {e}")


async def provider_configuration_example() -> None:
    """
    Configure providers with custom settings.

    Shows how to customize provider behavior.
    """
    print("\n" + "=" * 60)
    print("Example 3: Provider Configuration")
    print("=" * 60)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nSkipping: ANTHROPIC_API_KEY not set")
        return

    # Get provider with custom settings
    provider = get_provider(
        name="claude",
        model="claude-sonnet-4-20250514",  # Specific model
        # Additional settings can be passed
    )

    print(f"\nProvider: {provider.name}")
    print(f"Capabilities:")
    print(f"  - Max Context: {provider.capabilities.max_context_tokens:,} tokens")
    print(f"  - Max Output: {provider.capabilities.max_output_tokens:,} tokens")
    print(f"  - Streaming: {provider.capabilities.supports_streaming}")
    print(f"  - System Prompt: {provider.capabilities.supports_system_prompt}")
    print(f"  - Tools/Functions: {provider.capabilities.supports_tools}")

    # Use provider directly
    from contextflow.core.types import Message

    response = await provider.complete(
        messages=[Message(role="user", content="Say hello in 10 words or less")],
        system="You are a helpful assistant. Be very brief.",
        max_tokens=50,
    )

    print(f"\nDirect provider call:")
    print(f"  Response: {response.content}")
    print(f"  Tokens: {response.tokens_used}")
    print(f"  Cost: ${response.cost_usd:.6f}")


async def provider_factory_example() -> None:
    """
    Using the ProviderFactory class.

    Shows the factory pattern for provider creation.
    """
    print("\n" + "=" * 60)
    print("Example 4: Provider Factory")
    print("=" * 60)

    # Create factory
    factory = ProviderFactory()

    print(f"\nAvailable providers via factory: {factory.list()}")

    # Check availability
    for name in ["claude", "openai", "ollama"]:
        available = factory.is_available(name)
        print(f"  {name}: {'registered' if available else 'not registered'}")

    # Create provider if API key is available
    if os.environ.get("ANTHROPIC_API_KEY"):
        provider = factory.create(
            name="claude",
            model="claude-sonnet-4-20250514",
        )
        print(f"\nCreated provider: {provider.name}")


async def model_comparison_example() -> None:
    """
    Compare different models from the same provider.

    Shows how model choice affects results.
    """
    print("\n" + "=" * 60)
    print("Example 5: Model Comparison")
    print("=" * 60)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nSkipping: ANTHROPIC_API_KEY not set")
        return

    # Different Claude models to compare
    models = [
        ("claude-sonnet-4-20250514", "Sonnet 4 - Balanced"),
        ("claude-3-5-haiku-20241022", "Haiku 3.5 - Fast"),
    ]

    task = "What algorithms are used in the ML pipeline?"

    for model_id, description in models:
        print(f"\n--- {description} ({model_id}) ---")

        try:
            async with ContextFlow(provider="claude") as cf:
                # Note: In production, you'd configure the model at init
                result = await cf.process(
                    task=task,
                    context=SAMPLE_TEXT,
                )

                print(f"Tokens: {result.total_tokens}")
                print(f"Time: {result.execution_time:.2f}s")
                print(f"Answer: {result.answer[:150]}...")

        except Exception as e:
            print(f"Error: {e}")


async def ollama_local_example() -> None:
    """
    Using Ollama for local inference.

    Shows how to use local models without API keys.
    """
    print("\n" + "=" * 60)
    print("Example 6: Ollama Local Provider")
    print("=" * 60)

    print("\nNote: This requires Ollama running locally.")
    print("Install: https://ollama.ai")
    print("Start: ollama serve")
    print("Pull model: ollama pull llama2")

    try:
        # Try to connect to local Ollama
        provider = get_provider(
            name="ollama",
            model="llama2",  # or any model you have pulled
        )

        print(f"\nConnected to Ollama")
        print(f"Model: llama2")

        async with ContextFlow(provider=provider) as cf:
            result = await cf.process(
                task="What is this document about? One sentence.",
                context=SAMPLE_TEXT,
            )

            print(f"\nResult: {result.answer}")

    except Exception as e:
        print(f"\nCould not connect to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")


async def fallback_provider_example() -> None:
    """
    Implement fallback between providers.

    Shows how to handle provider failures gracefully.
    """
    print("\n" + "=" * 60)
    print("Example 7: Provider Fallback Strategy")
    print("=" * 60)

    # Define fallback order
    fallback_providers = ["claude", "openai", "groq"]

    # Filter to available providers
    available = [
        p for p in fallback_providers
        if (p == "claude" and os.environ.get("ANTHROPIC_API_KEY")) or
           (p == "openai" and os.environ.get("OPENAI_API_KEY")) or
           (p == "groq" and os.environ.get("GROQ_API_KEY"))
    ]

    if not available:
        print("\nNo providers available for fallback demo.")
        return

    print(f"\nFallback order: {available}")

    task = "Describe the deployment strategy in one sentence"
    result = None

    for provider_name in available:
        try:
            print(f"\nTrying {provider_name}...")

            async with ContextFlow(provider=provider_name) as cf:
                result = await cf.process(
                    task=task,
                    context=SAMPLE_TEXT,
                )

            print(f"Success with {provider_name}!")
            print(f"Answer: {result.answer}")
            break  # Success, stop trying

        except Exception as e:
            print(f"Failed: {e}")
            print("Falling back to next provider...")
            continue

    if result is None:
        print("\nAll providers failed!")


async def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ContextFlow Multi-Provider Examples")
    print("=" * 60)

    try:
        await list_available_providers()
        await switch_providers_example()
        await provider_configuration_example()
        await provider_factory_example()
        await model_comparison_example()
        await ollama_local_example()
        await fallback_provider_example()

        print("\n" + "=" * 60)
        print("All provider examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
