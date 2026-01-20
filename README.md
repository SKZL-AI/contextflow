# ContextFlow

**Intelligent LLM Context Orchestration with Auto-Strategy Selection**

ContextFlow is a Python framework for processing large contexts with LLMs using automatic strategy selection based on context size and complexity.

## Features

- **Auto-Strategy Selection**: Automatically chooses the best strategy based on context size
  - **GSD** (< 10K tokens): Direct, single LLM call
  - **RALPH** (10K-100K tokens): Iterative, structured processing
  - **RLM** (> 100K tokens): Recursive with sub-agents (based on MIT CSAIL paper)

- **Multi-Provider Support**: Claude, OpenAI, Ollama, vLLM, Groq, Gemini, Mistral

- **In-Memory RAG**: FAISS-based vector search (~1ms queries, 23x faster than external DBs)

- **Async-First**: Built on asyncio for high performance

- **Cost Tracking**: Automatic token counting and cost estimation

## Installation

```bash
# Using pip
pip install contextflow

# Using poetry
poetry add contextflow
```

## Quick Start

```python
from contextflow import ContextFlow

# Initialize with default provider (Claude)
cf = ContextFlow(provider="claude")

# Process a large document
result = await cf.process(
    task="Summarize this document and extract key insights",
    documents=["large_report.txt"],
    strategy="auto"  # Automatically selects best strategy
)

print(result.answer)
print(f"Strategy used: {result.strategy_used}")
print(f"Total tokens: {result.total_tokens}")
print(f"Cost: ${result.total_cost:.4f}")
```

## Strategy Selection

ContextFlow automatically selects the optimal strategy:

| Context Size | Strategy | Description |
|--------------|----------|-------------|
| < 10K tokens | GSD | Direct LLM call, fastest |
| 10K-100K tokens | RALPH | Iterative processing with chunking |
| > 100K tokens | RLM | Recursive sub-agents with aggregation |

Override automatic selection:

```python
# Force specific strategy
result = await cf.process(
    task="Find all mentions of 'revenue'",
    documents=["annual_report.txt"],
    strategy="rlm_full"  # Force RLM even for smaller contexts
)
```

## Providers

### Supported Providers

| Provider | Models | Features |
|----------|--------|----------|
| Claude | claude-3-opus, sonnet, haiku | 200K context, streaming |
| OpenAI | gpt-4o, gpt-4-turbo, gpt-3.5 | 128K context, streaming |
| Ollama | llama2, mistral, etc. | Local, free |
| vLLM | Any HF model | High throughput |
| Groq | mixtral, llama | Fast inference |
| Gemini | gemini-pro, 1.5-pro | 1M context |
| Mistral | mistral-large, medium | European provider |

### Using Different Providers

```python
# Claude (default)
cf = ContextFlow(provider="claude")

# OpenAI
cf = ContextFlow(provider="openai", model="gpt-4o")

# Local Ollama
cf = ContextFlow(provider="ollama", model="llama2")

# Direct provider instance
from contextflow.providers import ClaudeProvider
provider = ClaudeProvider(model="claude-3-5-sonnet-20241022")
cf = ContextFlow(provider=provider)
```

## Configuration

### Environment Variables

```bash
# Provider API keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...

# Default provider
CONTEXTFLOW_DEFAULT_PROVIDER=claude

# Strategy settings
CONTEXTFLOW_GSD_MAX_TOKENS=10000
CONTEXTFLOW_RLM_MAX_PARALLEL_AGENTS=10
```

### Programmatic Configuration

```python
from contextflow import ContextFlowConfig, ContextFlow

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

## CLI Usage

```bash
# Process a document
contextflow process "Summarize this" document.txt

# Analyze without processing
contextflow analyze document.txt

# Start API server
contextflow serve --port 8000
```

## API Server

```bash
# Start server
contextflow serve

# Or with uvicorn
uvicorn contextflow.api.server:app --port 8000
```

### Endpoints

- `POST /process` - Process task with documents
- `POST /analyze` - Analyze context
- `GET /health` - Health check
- `WS /ws` - WebSocket for streaming

## Development

```bash
# Clone repository
git clone https://github.com/SKZL-AI/contextflow.git
cd contextflow

# Install dependencies
poetry install --with dev,test

# Run tests
make test

# Run linting
make lint

# Type checking
make typecheck
```

## Architecture

```
ContextFlow
├── Core (orchestrator, analyzer, router)
├── Strategies (GSD, RALPH, RLM)
├── Providers (Claude, OpenAI, Ollama, ...)
├── RAG (TemporaryRAG, Embeddings, Chunker)
├── Agents (SubAgent, Pool, Aggregator)
├── API (FastAPI server)
└── CLI (Typer commands)
```

## License

MIT License - see [LICENSE](LICENSE) for details.

ContextFlow is free and open-source software. You can use it for any purpose,
including commercial applications.

### Dependency Licenses

ContextFlow uses the following major dependencies:

| Package | License | Notes |
|---------|---------|-------|
| anthropic | MIT | Claude API SDK |
| openai | MIT | OpenAI API SDK |
| faiss-cpu | MIT | Vector search |
| sentence-transformers | Apache 2.0 | Embeddings |
| pydantic | MIT | Data validation |
| fastapi | MIT | API framework |

All dependencies are MIT, Apache 2.0, or BSD licensed.

**Note on sentence-transformers models:** Some pre-trained models have training
data with commercial use restrictions. For commercial deployments, verify the
license of your chosen embedding model.

---

## Known Limitations

ContextFlow v0.1.0 has the following known limitations, planned for future versions:

### In-Memory RAG (By Design)

The FAISS-based RAG system runs entirely in-memory for maximum performance (~1ms queries).
This means:
- **No persistence**: Data is lost when the process ends
- **RAM-limited**: Large document collections require sufficient memory
- **Single-process**: Not suitable for distributed deployments

**Planned for v0.2.0**: Optional external vector database integration (Pinecone, Weaviate, Qdrant).

### Session Storage

The SessionManager uses synchronous SQLite for simplicity. For high-concurrency production
deployments, this may become a bottleneck.

**Planned for v0.2.0**: Async database backends (PostgreSQL, Redis).

### RLM Strategy Security

The RLM strategy uses a REPL environment for code execution. While sandboxed, it should
be used with caution in untrusted environments.

**Recommendations**:
- Don't expose RLM endpoints to untrusted users
- Review generated code before execution in sensitive contexts
- Consider using GSD or RALPH strategies for user-facing applications

### Strategy Thresholds

The automatic strategy selection thresholds (10K, 50K, 100K tokens) are based on
empirical testing but may not be optimal for all use cases. You can override them:

```python
result = await cf.process(
    task="...",
    documents=["..."],
    strategy="ralph"  # Force specific strategy
)
```

---

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

---

## Acknowledgments

ContextFlow is inspired by and builds upon the following research and methodologies:

### Research Papers

- **RLM (Recursive Language Models)**: Based on the research paper by
  [Zhang, Kraska, and Khattab (MIT CSAIL)](https://arxiv.org/abs/2512.24601).
  The official implementation is available at
  [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)
  under the **MIT License**.

### Methodologies

- **GSD (Get Shit Done)**: Strategy methodology inspired by the
  [TACHES framework](https://github.com/glittercowboy/get-shit-done),
  licensed under **MIT License**. ContextFlow implements the GSD principles independently.

- **RALPH (Rapid Adaptive Large-context Processing Hybrid)**: Iterative
  processing methodology based on general software engineering principles
  for structured decomposition and feedback loops.

### Best Practices

- **Boris' 13 Best Practices**: Community-shared recommendations for Claude Code
  development workflows, including the Verification Loop concept (Step 13).

### Note on Implementation

ContextFlow provides **independent implementations** of these methodologies.
No code has been directly copied from the referenced projects. The strategies
(GSD, RALPH, RLM) are original implementations following the published
principles and research.

---

## Citation

If you use ContextFlow in academic work, please cite:

```bibtex
@software{contextflow2026,
  title = {ContextFlow: Intelligent LLM Context Orchestration},
  author = {Sakizli, Furkan},
  year = {2026},
  url = {https://github.com/SKZL-AI/contextflow}
}
```

ContextFlow's RLM strategy is based on:

```bibtex
@article{zhang2025recursive,
  title = {Recursive Language Models},
  author = {Zhang, Alex L. and Kraska, Tim and Khattab, Omar},
  journal = {arXiv preprint arXiv:2512.24601},
  year = {2025}
}
```
