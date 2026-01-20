# ContextFlow CLI Usage Guide

This guide demonstrates common ContextFlow CLI commands and workflows.

## Installation

```bash
# Install from source
pip install -e .

# Verify installation
contextflow --version
```

## Basic Commands

### Get Help

```bash
# Main help
contextflow --help

# Command-specific help
contextflow process --help
contextflow analyze --help
contextflow serve --help
```

### System Information

```bash
# Display system info
contextflow info

# Example output:
# ContextFlow System Info
# Version      0.1.0
# Python       3.11.5
# Platform     Windows-10-...
# Providers    claude, openai, ollama, vllm, groq, gemini, mistral
# Default      claude
# Config       Yes
```

### List Providers

```bash
# Show available providers
contextflow providers

# Example output:
# Available Providers
# Provider   Status       Models   Max Context
# claude     Available    -        -
# openai     Configured   -        -
# ollama     Available    -        -
```

### Show Strategies

```bash
# Display strategy information
contextflow strategies

# Shows GSD, RALPH, and RLM strategy details
```

### View Configuration

```bash
# Basic configuration
contextflow config

# Full configuration
contextflow config --all
```

---

## Processing Documents

### Basic Processing

```bash
# Process a single file
contextflow process "Summarize this document" document.txt

# Process with explicit provider
contextflow process "Analyze the code" code.py --provider openai

# Process with specific strategy
contextflow process "Find all TODOs" code.py --strategy gsd_direct
```

### Processing Options

```bash
# Set output format
contextflow process "Summarize" doc.txt --output json
contextflow process "Summarize" doc.txt --output markdown

# Enable streaming output
contextflow process "Explain this" doc.txt --stream

# Set timeout
contextflow process "Analyze" large_doc.txt --timeout 300

# Verbose mode for debugging
contextflow process "Summarize" doc.txt --verbose
```

### Processing Multiple Files

```bash
# Process multiple files
contextflow process "Compare these files" file1.txt file2.txt

# Process with glob pattern
contextflow process "Analyze the codebase" "src/**/*.py"
```

### Processing with Constraints

```bash
# Add constraints to the task
contextflow process "Summarize the key points" report.pdf \
  --constraint "Keep under 500 words" \
  --constraint "Use bullet points"
```

---

## Analyzing Documents

### Basic Analysis

```bash
# Analyze a document
contextflow analyze document.txt

# Example output:
# Analysis Results
# Token Count    2,456
# Complexity     0.65
# Density        0.42
# Structure      markdown
# Recommended    gsd_direct
# Est. Cost      $0.0012
```

### Analysis Options

```bash
# Output as JSON
contextflow analyze document.txt --output json

# LLM-assisted analysis (more accurate)
contextflow analyze document.txt --use-llm

# Analyze multiple files
contextflow analyze "src/**/*.py" --output json
```

---

## Running the API Server

### Basic Server

```bash
# Start with defaults (localhost:8000)
contextflow serve

# Custom host and port
contextflow serve --host 0.0.0.0 --port 8080

# Enable auto-reload for development
contextflow serve --reload
```

### Server Options

```bash
# Set number of workers
contextflow serve --workers 4

# Set log level
contextflow serve --log-level debug

# Enable CORS for specific origins
contextflow serve --cors-origins "http://localhost:3000"
```

### API Endpoints

Once the server is running, you can access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

Example API calls:

```bash
# Process request
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Summarize this text",
    "context": "Your document content here...",
    "strategy": "auto"
  }'

# Analyze request
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Your document content here..."
  }'

# Streaming request (SSE)
curl -N http://localhost:8000/api/v1/stream \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Explain this in detail",
    "context": "Your content..."
  }'
```

---

## Common Workflows

### Document Summarization

```bash
# Summarize a single document
contextflow process "Provide a concise summary" report.pdf

# Summarize with length constraint
contextflow process "Summarize in 3 paragraphs" report.pdf \
  --constraint "Maximum 300 words"

# Summarize multiple documents
contextflow process "Compare and summarize" doc1.txt doc2.txt doc3.txt
```

### Code Analysis

```bash
# Analyze code structure
contextflow process "Describe the architecture" "src/**/*.py"

# Find issues
contextflow process "Identify potential bugs and improvements" code.py

# Generate documentation
contextflow process "Generate API documentation" "src/api/*.py" \
  --constraint "Use docstring format"
```

### Research and Q&A

```bash
# Ask questions about documents
contextflow process "What are the main findings?" research_paper.pdf

# Extract specific information
contextflow process "List all mentioned companies and their roles" report.txt

# Compare documents
contextflow process "What are the key differences?" v1.txt v2.txt
```

### Batch Processing

```bash
# Process all markdown files
for file in docs/*.md; do
  contextflow process "Summarize" "$file" --output json >> summaries.json
done

# Process with parallel execution (using xargs)
find . -name "*.py" | xargs -P 4 -I {} contextflow analyze {} --output json
```

---

## Environment Variables

Configure ContextFlow using environment variables:

```bash
# Provider API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
export GOOGLE_API_KEY="..."
export MISTRAL_API_KEY="..."

# Default settings
export CONTEXTFLOW_DEFAULT_PROVIDER="claude"
export CONTEXTFLOW_LOG_LEVEL="INFO"

# Strategy thresholds
export CONTEXTFLOW_GSD_MAX_TOKENS="10000"
export CONTEXTFLOW_RALPH_MAX_TOKENS="100000"
```

Or use a `.env` file:

```ini
# .env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
CONTEXTFLOW_DEFAULT_PROVIDER=claude
CONTEXTFLOW_LOG_LEVEL=INFO
```

---

## Output Formats

### Text Output (Default)

```bash
contextflow process "Summarize" doc.txt
# Output: Plain text response
```

### JSON Output

```bash
contextflow process "Summarize" doc.txt --output json
# Output:
# {
#   "id": "exec-abc123",
#   "answer": "...",
#   "strategy_used": "gsd_direct",
#   "total_tokens": 1234,
#   "total_cost": 0.00012,
#   "execution_time": 2.34
# }
```

### Markdown Output

```bash
contextflow process "Summarize" doc.txt --output markdown
# Output: Formatted markdown with metadata
```

---

## Troubleshooting

### Common Issues

**API Key Not Found**
```bash
# Check if key is set
echo $ANTHROPIC_API_KEY

# Set key
export ANTHROPIC_API_KEY="your-key-here"
```

**Provider Not Available**
```bash
# List available providers
contextflow providers

# Use a different provider
contextflow process "Task" doc.txt --provider openai
```

**Timeout Errors**
```bash
# Increase timeout for large documents
contextflow process "Task" large_doc.txt --timeout 600
```

**Memory Issues**
```bash
# Use RLM strategy for very large documents
contextflow process "Task" huge_doc.txt --strategy rlm_full
```

### Debug Mode

```bash
# Enable verbose logging
contextflow process "Task" doc.txt --verbose

# Set debug log level
CONTEXTFLOW_LOG_LEVEL=DEBUG contextflow process "Task" doc.txt
```

---

## Tips and Best Practices

1. **Start with Analysis**: Always analyze large documents first to understand complexity:
   ```bash
   contextflow analyze large_doc.txt
   ```

2. **Use Constraints**: Add constraints for better results:
   ```bash
   contextflow process "Summarize" doc.txt \
     --constraint "Use bullet points" \
     --constraint "Include statistics"
   ```

3. **Choose the Right Strategy**: Let AUTO choose for most cases, but override for specific needs:
   - `gsd_direct`: Fast, simple tasks
   - `ralph_structured`: Medium documents, detailed analysis
   - `rlm_full`: Large documents, complex synthesis

4. **Use Streaming for Long Tasks**: Enable streaming to see progress:
   ```bash
   contextflow process "Detailed analysis" large_doc.txt --stream
   ```

5. **Save Results**: Redirect output for later use:
   ```bash
   contextflow process "Summarize" doc.txt --output json > result.json
   ```

---

## More Examples

For more detailed examples, see the Python examples in this directory:

- `01_basic_usage.py` - Basic ContextFlow usage
- `02_strategy_selection.py` - Strategy selection patterns
- `03_streaming.py` - Streaming output
- `04_verification.py` - Verification protocol
- `05_rag_processing.py` - RAG processing
- `06_hooks.py` - Lifecycle hooks
- `07_providers.py` - Multi-provider support

Run them with:
```bash
python examples/01_basic_usage.py
```
