# MCP Server

ContextFlow includes a Model Context Protocol (MCP) server for integration with Claude Code and other MCP-compatible clients.

## Overview

The MCP server exposes ContextFlow's capabilities as tools that can be invoked by AI assistants, enabling seamless integration into development workflows.

## Setup

### Installation

```bash
# Install with pip
pip install contextflow

# Or install with Poetry
poetry add contextflow
```

### Starting the Server

```bash
# Start MCP server
contextflow mcp start

# With options
contextflow mcp start --port 3000 --host localhost

# With specific provider
contextflow mcp start --provider claude --model claude-sonnet-4-20250514
```

### Environment Configuration

```bash
# Required
export ANTHROPIC_API_KEY=sk-ant-...

# Optional
export MCP_PORT=3000
export MCP_HOST=localhost
export CONTEXTFLOW_LOG_LEVEL=info
```

## Claude Code Integration

### Configuration

Add to your Claude Code MCP settings (`~/.claude/mcp_settings.json`):

```json
{
  "mcpServers": {
    "contextflow": {
      "command": "contextflow",
      "args": ["mcp", "start"],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

### Alternative: uvx Configuration

```json
{
  "mcpServers": {
    "contextflow": {
      "command": "uvx",
      "args": ["contextflow", "mcp", "start"],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

### Verify Integration

Once configured, Claude Code will have access to ContextFlow tools. Verify by asking Claude to list available MCP tools.

## Available Tools

### contextflow_process

Process input through ContextFlow with automatic strategy routing.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | string | Yes | The input to process |
| `strategy` | string | No | Strategy: auto, gsd, ralph, rlm |
| `context` | object | No | Additional context |
| `verify` | boolean | No | Enable verification |

**Example:**

```
Tool: contextflow_process
Input: {
  "input": "Explain the SOLID principles",
  "strategy": "ralph",
  "verify": true
}
```

### contextflow_stream

Stream responses in real-time.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | string | Yes | The input to process |
| `strategy` | string | No | Strategy to use |
| `context` | object | No | Additional context |

**Example:**

```
Tool: contextflow_stream
Input: {
  "input": "Write a tutorial on React hooks",
  "strategy": "rlm"
}
```

### contextflow_analyze

Analyze input routing without processing.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | string | Yes | The input to analyze |
| `context` | object | No | Additional context |

**Example:**

```
Tool: contextflow_analyze
Input: {
  "input": "Design a microservices architecture"
}
```

**Response:**

```json
{
  "recommendedStrategy": "rlm",
  "confidence": 0.87,
  "scores": {
    "gsd": 0.12,
    "ralph": 0.45,
    "rlm": 0.87
  },
  "reasoning": "Complex system design task requiring iterative refinement"
}
```

### contextflow_configure

Update ContextFlow configuration.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `provider` | string | No | Provider to use |
| `model` | string | No | Model to use |
| `strategy` | string | No | Default strategy |
| `verification` | object | No | Verification settings |

**Example:**

```
Tool: contextflow_configure
Input: {
  "provider": "openai",
  "model": "gpt-4-turbo",
  "verification": {
    "enabled": true,
    "strictMode": true
  }
}
```

### contextflow_status

Get current server status and configuration.

**Parameters:** None

**Response:**

```json
{
  "status": "running",
  "version": "1.0.0",
  "provider": "claude",
  "model": "claude-sonnet-4-20250514",
  "defaultStrategy": "auto",
  "verification": {
    "enabled": true
  },
  "uptime": 3600
}
```

## Usage Examples

### In Claude Code

When using Claude Code with ContextFlow MCP integration:

```
User: Use ContextFlow to explain dependency injection

Claude: I'll use the contextflow_process tool to explain dependency injection.

[Invokes contextflow_process with input "Explain dependency injection with examples"]

Dependency injection is a design pattern...
```

### Complex Tasks

```
User: Design a scalable API using ContextFlow's RLM strategy

Claude: I'll use ContextFlow with the RLM strategy for this complex design task.

[Invokes contextflow_process with strategy "rlm"]

Here's a comprehensive API design...
```

### Analysis Before Processing

```
User: Should I use RALPH or RLM for code review?

Claude: Let me analyze this with ContextFlow.

[Invokes contextflow_analyze with input "Review this code for best practices"]

Based on the analysis:
- Recommended: RALPH (confidence: 0.78)
- Reasoning: Code review benefits from structured analysis phases
```

## Server Options

### Command Line Options

```bash
contextflow mcp start [options]

Options:
  --port, -p      Port to listen on (default: 3000)
  --host, -h      Host to bind to (default: localhost)
  --provider      Default provider (default: claude)
  --model         Default model
  --strategy      Default strategy (default: auto)
  --debug         Enable debug logging
  --config        Path to config file
```

### Programmatic Usage

```python
import asyncio
from contextflow.mcp import start_mcp_server

async def main():
    server = await start_mcp_server(
        port=3000,
        host="localhost",
        contextflow_config={
            "provider": "claude",
            "model": "claude-sonnet-4-20250514",
            "verification": {"enabled": True}
        }
    )

    # Later: stop server
    await server.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Troubleshooting

### Common Issues

**Server won't start:**
- Check that the port is not in use
- Verify API keys are set correctly
- Check logs with `--debug` flag

**Tools not appearing in Claude Code:**
- Verify MCP settings configuration
- Restart Claude Code
- Check server is running

**Connection errors:**
- Ensure firewall allows connections
- Verify host/port settings match

### Debug Mode

```bash
# Enable verbose logging
contextflow mcp start --debug

# Check server logs
contextflow mcp logs
```

### Health Check

```bash
# Check server health
curl http://localhost:3000/health

# Expected response
{"status": "ok", "version": "1.0.0"}
```

### Health Check (Python)

```python
import httpx

async def check_health():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:3000/health")
        print(response.json())
        # Expected: {"status": "ok", "version": "1.0.0"}
```

---

## See Also

- [CLI Reference](./cli.md)
- [Getting Started](./getting-started.md)
- [API Reference](./api-reference.md)
