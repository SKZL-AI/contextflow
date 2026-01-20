# CLI Reference

ContextFlow provides a command-line interface for quick interactions and automation.

## Installation

The CLI is included with the ContextFlow package:

```bash
npm install -g contextflow
```

Or use via npx:

```bash
npx contextflow <command>
```

## Commands

### process

Process input through ContextFlow.

```bash
contextflow process [options] <input>
```

#### Options

| Option | Alias | Description | Default |
|--------|-------|-------------|---------|
| `--strategy` | `-s` | Strategy to use | auto |
| `--provider` | `-p` | Provider to use | claude |
| `--model` | `-m` | Model to use | provider default |
| `--output` | `-o` | Output file | stdout |
| `--format` | `-f` | Output format (text, json) | text |
| `--verify` | `-v` | Enable verification | false |
| `--context` | `-c` | Context file (JSON) | none |

#### Examples

```bash
# Simple query
contextflow process "What is machine learning?"

# With strategy
contextflow process -s ralph "Analyze the benefits of microservices"

# With provider
contextflow process -p openai -m gpt-4 "Explain quantum computing"

# With context file
contextflow process -c context.json "Summarize this document"

# Output to file
contextflow process -o result.txt "Write a poem about coding"

# JSON output with verification
contextflow process -f json -v "List 5 programming languages"
```

### stream

Stream responses in real-time.

```bash
contextflow stream [options] <input>
```

#### Options

Same as `process`, plus:

| Option | Alias | Description | Default |
|--------|-------|-------------|---------|
| `--no-color` | | Disable colored output | false |

#### Examples

```bash
# Stream a long response
contextflow stream "Write a detailed guide on Docker"

# Stream with specific strategy
contextflow stream -s rlm "Design a REST API architecture"
```

### analyze

Analyze input without processing.

```bash
contextflow analyze [options] <input>
```

#### Options

| Option | Alias | Description | Default |
|--------|-------|-------------|---------|
| `--format` | `-f` | Output format (text, json) | text |
| `--verbose` | | Show detailed analysis | false |

#### Examples

```bash
# Analyze routing decision
contextflow analyze "Build a recommendation system"

# Verbose analysis
contextflow analyze --verbose "Explain recursion"

# JSON output
contextflow analyze -f json "Design a database schema"
```

### config

Manage configuration.

```bash
contextflow config <subcommand>
```

#### Subcommands

```bash
# Show current config
contextflow config show

# Set a value
contextflow config set provider openai
contextflow config set model gpt-4-turbo
contextflow config set defaultStrategy ralph

# Get a value
contextflow config get provider

# Reset to defaults
contextflow config reset

# Initialize config file
contextflow config init
```

### providers

List and manage providers.

```bash
contextflow providers [subcommand]
```

#### Examples

```bash
# List available providers
contextflow providers list

# Check provider status
contextflow providers status

# Test provider connection
contextflow providers test claude
contextflow providers test openai
```

## Global Options

These options work with all commands:

| Option | Alias | Description |
|--------|-------|-------------|
| `--help` | `-h` | Show help |
| `--version` | | Show version |
| `--debug` | `-d` | Enable debug logging |
| `--quiet` | `-q` | Suppress non-essential output |
| `--config` | | Path to config file |

## Environment Variables

The CLI respects these environment variables:

```bash
# Provider API keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export OLLAMA_BASE_URL=http://localhost:11434

# Defaults
export CONTEXTFLOW_PROVIDER=claude
export CONTEXTFLOW_MODEL=claude-sonnet-4-20250514
export CONTEXTFLOW_STRATEGY=auto

# Logging
export CONTEXTFLOW_LOG_LEVEL=info
export CONTEXTFLOW_DEBUG=false
```

## Configuration File

Create `contextflow.config.js` or `.contextflowrc.json`:

```javascript
// contextflow.config.js
module.exports = {
  provider: 'claude',
  model: 'claude-sonnet-4-20250514',
  defaultStrategy: 'auto',
  verification: {
    enabled: true
  }
};
```

```json
// .contextflowrc.json
{
  "provider": "claude",
  "model": "claude-sonnet-4-20250514",
  "defaultStrategy": "auto",
  "verification": {
    "enabled": true
  }
}
```

## Piping and Scripting

### Pipe Input

```bash
# Pipe from file
cat document.txt | contextflow process "Summarize this"

# Pipe from command
git diff | contextflow process "Explain these changes"

# Chain commands
echo "Hello world" | contextflow process "Translate to Spanish" | tee output.txt
```

### JSON Output for Scripts

```bash
# Get JSON result
result=$(contextflow process -f json "What is 2+2?")
echo $result | jq '.output'

# Use in scripts
#!/bin/bash
response=$(contextflow process -f json -s gsd "Get current date")
strategy=$(echo $response | jq -r '.strategy')
echo "Used strategy: $strategy"
```

### Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Provider error |
| 4 | Verification failed |
| 5 | Timeout |

## Examples

### Daily Use

```bash
# Quick question
contextflow process "What's the syntax for async/await in JavaScript?"

# Code review
git diff HEAD~1 | contextflow process -s ralph "Review this code"

# Documentation
contextflow process -o README.md "Generate README for a Node.js API project"
```

### Automation

```bash
#!/bin/bash
# analyze_logs.sh

LOG_FILE=$1
CONTEXT=$(cat $LOG_FILE)

contextflow process \
  -s ralph \
  -f json \
  -c <(echo "{\"logs\": \"$CONTEXT\"}") \
  "Analyze these logs for errors and suggest fixes" \
  | jq '.output'
```

### Integration

```bash
# CI/CD pipeline
contextflow process -v -f json "Review PR #123 changes" > review.json

# Cron job
0 9 * * * contextflow process "Generate daily report" | mail -s "Daily Report" team@example.com
```

---

## See Also

- [Getting Started](./getting-started.md)
- [API Reference](./api-reference.md)
- [MCP Server](./mcp-server.md)
