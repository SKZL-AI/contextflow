# ContextFlow Documentation

> Intelligent context management for AI workflows with adaptive routing strategies

## Overview

ContextFlow is a powerful context management library designed to optimize how AI applications handle, route, and process contextual information. It provides:

- **Adaptive Strategy Routing** - Automatically selects the optimal processing strategy based on task complexity
- **Multi-Provider Support** - Works with Claude, OpenAI, Ollama, and custom providers
- **Built-in Verification** - Implements verification protocols to ensure output quality
- **Streaming Support** - Real-time streaming for responsive applications
- **CLI & MCP Server** - Multiple integration options for different use cases

## Key Features

### Intelligent Strategy Selection

ContextFlow automatically routes requests through the most appropriate strategy:

| Strategy | Best For | Complexity |
|----------|----------|------------|
| **GSD** | Quick tasks, simple queries | Low |
| **RALPH** | Research, analysis, multi-step reasoning | Medium |
| **RLM** | Complex reasoning, planning, problem-solving | High |

### Multi-Provider Architecture

```python
import os
from contextflow import ContextFlow

flow = ContextFlow(
    provider="claude",
    model="claude-sonnet-4-20250514"
)
```

## Quick Start

### Installation

```bash
pip install contextflow
```

### Basic Usage

```python
import os
from contextflow import ContextFlow

# Initialize with default settings
flow = ContextFlow(
    provider="claude",
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# Process a request
result = await flow.process(
    input="Explain quantum computing in simple terms",
    context={"audience": "beginners"}
)

print(result.output)
print(f"Strategy used: {result.strategy}")
```

### Streaming

```python
async for chunk in flow.stream(
    input="Write a story about a robot",
    strategy="ralph"
):
    print(chunk.content, end="", flush=True)
```

## Documentation Index

### Getting Started

- [**Getting Started Guide**](./getting-started.md) - Installation, setup, and first steps
- [**Configuration**](./getting-started.md#configuration) - Configure ContextFlow for your needs

### Core Concepts

- [**Strategies**](./strategies.md) - Understanding GSD, RALPH, and RLM strategies
- [**Providers**](./providers.md) - Supported AI providers and configuration
- [**Verification**](./verification.md) - Output verification protocols

### Reference

- [**API Reference**](./api-reference.md) - Complete API documentation
- [**CLI Reference**](./cli.md) - Command-line interface usage
- [**MCP Server**](./mcp-server.md) - Model Context Protocol server setup

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   ContextFlow                        │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │     GSD     │  │    RALPH    │  │     RLM     │ │
│  │  Strategy   │  │  Strategy   │  │  Strategy   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │         │
│         └────────────────┼────────────────┘         │
│                          ▼                          │
│              ┌───────────────────┐                  │
│              │  Strategy Router  │                  │
│              └─────────┬─────────┘                  │
│                        ▼                            │
│  ┌─────────────────────────────────────────────┐   │
│  │              Provider Layer                  │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────┐ │   │
│  │  │ Claude │ │ OpenAI │ │ Ollama │ │Custom│ │   │
│  │  └────────┘ └────────┘ └────────┘ └──────┘ │   │
│  └─────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────┤
│              Verification Layer                      │
└─────────────────────────────────────────────────────┘
```

## Examples

### Auto-Routing

```python
# ContextFlow automatically selects the best strategy
result = await flow.process(
    input="What is 2 + 2?"  # Uses GSD (simple)
)

result2 = await flow.process(
    input="Analyze market trends for Q4"  # Uses RALPH (analysis)
)

result3 = await flow.process(
    input="Design a distributed system architecture"  # Uses RLM (complex)
)
```

### Manual Strategy Selection

```python
result = await flow.process(
    input="Research quantum computing applications",
    strategy="ralph",  # Force RALPH strategy
    options={
        "max_iterations": 5,
        "verify_output": True
    }
)
```

## Support

- [GitHub Issues](https://github.com/contextflow/contextflow/issues)
- [Discussions](https://github.com/contextflow/contextflow/discussions)
- [Contributing Guide](https://github.com/contextflow/contextflow/blob/main/CONTRIBUTING.md)

## License

MIT License - see [LICENSE](https://github.com/contextflow/contextflow/blob/main/LICENSE) for details.
