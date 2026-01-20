"""
ContextFlow Examples.

This package contains working examples demonstrating various ContextFlow
features and usage patterns.

Examples:
    01_basic_usage.py      - Basic ContextFlow usage
    02_strategy_selection.py - Strategy selection (auto vs manual)
    03_streaming.py        - Streaming output
    04_verification.py     - Verification protocol
    05_rag_processing.py   - RAG with document collections
    06_hooks.py            - Lifecycle hooks
    07_providers.py        - Multi-provider support
    08_cli_usage.md        - CLI command examples

Running Examples:
    python -m examples.01_basic_usage
    python examples/01_basic_usage.py

Prerequisites:
    - Set ANTHROPIC_API_KEY (or relevant provider key) in environment
    - Install contextflow: pip install -e .

Note:
    These examples require valid API keys for the providers being used.
    Set environment variables or use .env file:
    - ANTHROPIC_API_KEY for Claude
    - OPENAI_API_KEY for OpenAI
    - GROQ_API_KEY for Groq
    - GOOGLE_API_KEY for Gemini
    - MISTRAL_API_KEY for Mistral
"""

__all__ = [
    "EXAMPLE_MODULES",
]

EXAMPLE_MODULES = [
    "01_basic_usage",
    "02_strategy_selection",
    "03_streaming",
    "04_verification",
    "05_rag_processing",
    "06_hooks",
    "07_providers",
]
