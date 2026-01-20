# Getting Started with ContextFlow

This guide walks you through installing ContextFlow, basic configuration, and your first integration.

## Prerequisites

- **Python** 3.11 or higher
- **pip** or **poetry** package manager
- API key for at least one supported provider (Claude, OpenAI, etc.)

## Installation

### pip

```bash
pip install contextflow
```

### poetry

```bash
poetry add contextflow
```

### From Source

```bash
git clone https://github.com/contextflow/contextflow.git
cd contextflow
poetry install
```

## Quick Start

### 1. Set Up Environment Variables

Create a `.env` file in your project root:

```env
# Required: At least one provider API key
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional: Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Default settings
CONTEXTFLOW_DEFAULT_PROVIDER=claude
CONTEXTFLOW_DEFAULT_MODEL=claude-sonnet-4-20250514
CONTEXTFLOW_LOG_LEVEL=info
```

### 2. Basic Usage

```python
import os
import asyncio
from contextflow import ContextFlow

# Initialize ContextFlow
flow = ContextFlow(
    provider="claude",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    model="claude-sonnet-4-20250514"
)

# Process a simple request
async def main():
    result = await flow.process(
        input="Explain the concept of recursion in programming"
    )

    print(result.output)
    print(f"Strategy: {result.strategy}")
    print(f"Tokens used: {result.usage.total_tokens}")

asyncio.run(main())
```

### 3. With Context

```python
result = await flow.process(
    input="Summarize the key points",
    context={
        "document": """
            Artificial Intelligence (AI) has transformed numerous industries...
            [Your document content here]
        """,
        "style": "bullet points",
        "max_length": 500
    }
)
```

## Configuration

### Full Configuration Options

```python
import os
from contextflow import ContextFlow

# ContextFlowConfig type hint (for reference):
# provider: str
# api_key: str
# model: str
# base_url: Optional[str]
# default_strategy: Literal["auto", "gsd", "ralph", "rlm"]
# strategy_config: dict
# verification: dict
# log_level: Literal["debug", "info", "warn", "error"]
# timeout: int
# max_retries: int
# retry_delay: int

config = {
    # Provider settings
    "provider": "claude",
    "api_key": os.environ.get("ANTHROPIC_API_KEY"),
    "model": "claude-sonnet-4-20250514",
    "base_url": None,  # Custom API endpoint

    # Strategy settings
    "default_strategy": "auto",  # "auto" | "gsd" | "ralph" | "rlm"
    "strategy_config": {
        "gsd": {
            "max_tokens": 1000,
            "temperature": 0.3
        },
        "ralph": {
            "max_iterations": 5,
            "temperature": 0.5
        },
        "rlm": {
            "max_depth": 10,
            "temperature": 0.7
        }
    },

    # Verification settings
    "verification": {
        "enabled": True,
        "checks": ["format", "completeness", "accuracy"],
        "strict_mode": False
    },

    # Logging
    "log_level": "info",  # "debug" | "info" | "warn" | "error"

    # Timeouts and limits
    "timeout": 30000,  # 30 seconds
    "max_retries": 3,
    "retry_delay": 1000
}

flow = ContextFlow(**config)
```

### Configuration File

Create `contextflow_config.py` in your project root:

```python
config = {
    "provider": "claude",
    "model": "claude-sonnet-4-20250514",
    "default_strategy": "auto",
    "verification": {
        "enabled": True
    },
    "logging": {
        "level": "info",
        "format": "json"
    }
}
```

## Basic Examples

### Simple Query

```python
result = await flow.process(
    input="What is the capital of France?"
)
# Uses GSD strategy for simple factual query
```

### Analysis Task

```python
result = await flow.process(
    input="Analyze the sentiment of customer reviews",
    context={
        "reviews": [
            "Great product, works perfectly!",
            "Disappointed with the quality",
            "Average, nothing special"
        ]
    }
)
# Uses RALPH strategy for analysis
```

### Complex Reasoning

```python
result = await flow.process(
    input="Design a microservices architecture for an e-commerce platform",
    context={
        "requirements": {
            "scale": "10000 concurrent users",
            "features": ["catalog", "cart", "checkout", "recommendations"]
        }
    }
)
# Uses RLM strategy for complex design task
```

### Streaming Response

```python
import sys

stream = flow.stream(
    input="Write a detailed tutorial on React hooks"
)

async for chunk in stream:
    sys.stdout.write(chunk.content)
    sys.stdout.flush()

    if chunk.metadata:
        print(f"\nMetadata: {chunk.metadata}")
```

## Working with Strategies

### Auto-Routing (Default)

```python
# ContextFlow analyzes the input and selects the best strategy
result = await flow.process(
    input="Your query here",
    strategy="auto"  # This is the default
)

print(f"Selected strategy: {result.strategy}")
```

### Manual Strategy Selection

```python
# Force a specific strategy
result = await flow.process(
    input="Research AI ethics frameworks",
    strategy="ralph"
)
```

### Strategy Analysis

```python
# Analyze which strategy would be selected without processing
analysis = await flow.analyze(
    input="Build a recommendation engine"
)

print(analysis)
# {
#     "recommended_strategy": "rlm",
#     "confidence": 0.85,
#     "reasoning": "Complex system design requiring iterative refinement",
#     "alternative_strategies": ["ralph"]
# }
```

## Error Handling

```python
from contextflow import ContextFlow, ContextFlowError, ProviderError

flow = ContextFlow(...)  # config

try:
    result = await flow.process(
        input="Process this request"
    )
except ProviderError as error:
    print(f"Provider error: {error}")
    print(f"Provider: {error.provider}")
    print(f"Status: {error.status_code}")
except ContextFlowError as error:
    print(f"ContextFlow error: {error}")
    print(f"Code: {error.code}")
except Exception as error:
    raise error
```

## Python Type Hints

ContextFlow provides full type hint support:

```python
from contextflow import (
    ContextFlow,
    ContextFlowConfig,
    ProcessInput,
    ProcessResult,
    Strategy,
    Provider
)
from typing import Optional, List, Literal

# Type aliases for reference:
# Strategy = Literal["auto", "gsd", "ralph", "rlm"]
# Provider = Literal["claude", "openai", "ollama", "vllm", "groq", "gemini", "mistral"]

# All types are available
config: ContextFlowConfig = {...}
input_data: ProcessInput = {...}
result: ProcessResult = await flow.process(input_data)
```

## Next Steps

- [**Strategies Guide**](./strategies.md) - Deep dive into GSD, RALPH, and RLM
- [**API Reference**](./api-reference.md) - Complete API documentation
- [**Providers**](./providers.md) - Configure different AI providers
- [**Verification**](./verification.md) - Set up output verification
- [**CLI**](./cli.md) - Use ContextFlow from the command line
