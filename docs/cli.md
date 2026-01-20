# CLI Reference

ContextFlow provides a command-line interface for quick interactions and automation.

## Installation

The CLI is included with the ContextFlow package:

```bash
pip install contextflow
```

Or with Poetry:

```bash
poetry add contextflow
```

## Commands

### process

Process input through ContextFlow.

```bash
contextflow process [OPTIONS] TASK [DOCUMENTS]...
```

#### Options

| Option | Alias | Description | Default |
|--------|-------|-------------|---------|
| `--strategy` | `-s` | Strategy to use (auto, gsd, ralph, rlm) | auto |
| `--provider` | `-p` | Provider to use | claude |
| `--context` | `-c` | Direct context string | none |
| `--output` | `-o` | Output format (text, json, markdown) | text |
| `--output-file` | `-O` | Write output to file | stdout |
| `--stream` | | Enable streaming output | false |
| `--no-verify` | | Disable verification | false |
| `--verbose` | `-v` | Verbose output | false |

#### Examples

```bash
# Simple query with document
contextflow process "Summarize this document" document.txt

# With inline context
contextflow process --context "Python is a programming language" "What is Python?"

# With specific strategy
contextflow process -s ralph "Analyze the benefits of microservices" architecture.md

# With provider selection
contextflow process -p openai "Explain quantum computing" research.pdf

# Output to file as JSON
contextflow process -o json -O result.json "Extract key points" report.txt

# Streaming mode
contextflow process --stream "Write a detailed guide on Docker"

# Multiple documents
contextflow process "Compare these files" file1.py file2.py file3.py
```

### analyze

Analyze input without processing to get strategy recommendations.

```bash
contextflow analyze [OPTIONS] [DOCUMENTS]...
```

#### Options

| Option | Alias | Description | Default |
|--------|-------|-------------|---------|
| `--context` | `-c` | Direct context string | none |
| `--output` | `-o` | Output format (text, json) | text |
| `--output-file` | `-O` | Write output to file | stdout |
| `--verbose` | `-v` | Verbose output with LLM analysis | false |

#### Examples

```bash
# Analyze a document
contextflow analyze document.txt

# Analyze with verbose output
contextflow analyze --verbose large_report.pdf

# JSON output
contextflow analyze -o json -O analysis.json codebase/*.py

# Analyze inline context
contextflow analyze --context "Your text here"
```

### serve

Start the API server.

```bash
contextflow serve [OPTIONS]
```

#### Options

| Option | Alias | Description | Default |
|--------|-------|-------------|---------|
| `--host` | `-h` | Host to bind | 0.0.0.0 |
| `--port` | `-p` | Port to bind | 8000 |
| `--reload` | | Enable auto-reload | false |
| `--workers` | `-w` | Number of workers | 1 |
| `--cors-origins` | | CORS allowed origins | none |
| `--api-key` | | API key for authentication | none |

#### Examples

```bash
# Start with defaults
contextflow serve

# Custom port
contextflow serve --port 3000

# Development mode with auto-reload
contextflow serve --reload

# Production with multiple workers
contextflow serve --workers 4 --host 0.0.0.0 --port 8080

# With CORS
contextflow serve --cors-origins "http://localhost:3000,https://myapp.com"
```

### info

Show system information.

```bash
contextflow info
```

### providers

List available providers.

```bash
contextflow providers
```

### strategies

List available strategies.

```bash
contextflow strategies
```

## Global Options

These options work with all commands:

| Option | Alias | Description |
|--------|-------|-------------|
| `--help` | | Show help |
| `--version` | | Show version |

## Environment Variables

The CLI respects these environment variables:

```bash
# Provider API keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GROQ_API_KEY=gsk_...
export GOOGLE_API_KEY=...

# Ollama configuration
export OLLAMA_BASE_URL=http://localhost:11434

# Defaults
export CONTEXTFLOW_DEFAULT_PROVIDER=claude
export CONTEXTFLOW_GSD_MAX_TOKENS=10000
export CONTEXTFLOW_RLM_MAX_PARALLEL_AGENTS=10

# Logging
export CONTEXTFLOW_LOG_LEVEL=info
```

## Configuration

ContextFlow can be configured via environment variables or programmatically in Python:

```python
from contextflow import ContextFlow, ContextFlowConfig
from contextflow.core.config import StrategyConfig, RAGConfig

config = ContextFlowConfig(
    default_provider="claude",
    strategy=StrategyConfig(
        rlm_max_parallel_agents=20,
        rlm_max_iterations=100,
    ),
    rag=RAGConfig(
        chunk_size=5000,
        chunk_overlap=500,
    ),
)

cf = ContextFlow(config=config)
```

## Piping and Scripting

### Pipe Input

```bash
# Pipe from file
cat document.txt | contextflow process "Summarize this" -

# Pipe from command
git diff | contextflow process "Explain these changes" -

# Chain commands
echo "Hello world" | contextflow process "Translate to Spanish" - | tee output.txt
```

### JSON Output for Scripts

```bash
# Get JSON result
result=$(contextflow process -o json "What is 2+2?" --context "Math question")
echo $result | jq '.answer'

# Use in scripts
#!/bin/bash
response=$(contextflow process -o json -s gsd --context "Test" "Get info")
strategy=$(echo $response | jq -r '.strategy_used')
echo "Used strategy: $strategy"
```

### Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |

## Examples

### Daily Use

```bash
# Quick question
contextflow process --context "Python basics" "What's the syntax for list comprehension?"

# Code review
git diff HEAD~1 | contextflow process -s ralph "Review this code" -

# Summarize documentation
contextflow process "Summarize this README" README.md
```

### Automation

```bash
#!/bin/bash
# analyze_logs.sh

LOG_FILE=$1

contextflow process \
  -s ralph \
  -o json \
  "Analyze these logs for errors and suggest fixes" \
  "$LOG_FILE" \
  | jq '.answer'
```

### Integration with Python

```python
import subprocess
import json

# Call CLI from Python
result = subprocess.run(
    ["contextflow", "process", "-o", "json", "Summarize", "doc.txt"],
    capture_output=True,
    text=True
)
data = json.loads(result.stdout)
print(data["answer"])
```

---

## See Also

- [Getting Started](./getting-started.md)
- [API Reference](./api-reference.md)
- [MCP Server](./mcp-server.md)
