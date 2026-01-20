# Changelog

All notable changes to ContextFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Performance benchmarks
- Additional embedding providers
- Kubernetes deployment manifests

---

## [0.1.0] - 2026-01-20

### Added

#### Core Features
- **3-Strategy Architecture** - Intelligent auto-selection based on context size
  - GSD (Get Stuff Done): 0-10K tokens, single-pass processing
  - RALPH (Research-Analyze-Learn-Plan-Help): 10K-100K tokens, hierarchical processing
  - RLM (Recursive Language Model): 50K-10M+ tokens, recursive with REPL
- **Context Analyzer** - Automatic token counting, density analysis, complexity detection
- **Strategy Router** - Decision matrix for optimal strategy selection
- **Verification Protocol** - Boris Step 13 implementation with 6 check types
- **Lifecycle Hooks** - PRE_PROCESS, POST_PROCESS, ON_ERROR hooks
- **Session Manager** - SQLite-backed session persistence (Claude-Mem pattern)

#### Providers (7 LLM Providers)
- **Claude** (Anthropic) - 200K context, streaming, vision support
- **OpenAI** - GPT-4/GPT-3.5, tiktoken for exact token counting
- **Gemini** (Google) - 1M context window support
- **Groq** - LPU-based ultra-fast inference
- **Ollama** - Local model support with 40+ models
- **Mistral** - EU-based provider with JSON mode
- **vLLM** - Self-hosted high-throughput inference

#### RAG System
- **TemporaryRAG** - FAISS-based in-memory vector search
- **3-Layer Progressive Disclosure** - Compact/Timeline/Full retrieval
- **Smart Chunker** - Sentence/paragraph/semantic boundary detection
- **Embedding Providers** - OpenAI, Sentence Transformers, Ollama

#### API & CLI
- **FastAPI REST API** - 10+ endpoints with OpenAPI docs
- **WebSocket Support** - Real-time streaming
- **SSE Streaming** - Server-Sent Events for progress updates
- **Typer CLI** - 7 commands (process, stream, analyze, serve, info, providers, strategies)
- **MCP Server** - Model Context Protocol for Claude Code integration

#### Developer Experience
- **Type Hints** - 95%+ coverage, mypy strict mode
- **Structured Logging** - structlog with JSON output
- **Error Hierarchy** - Custom exception classes
- **Retry Logic** - tenacity-based with exponential backoff
- **CI/CD Pipeline** - GitHub Actions for lint, type-check, test, security, build

### Documentation
- API Reference (26 Pydantic models documented)
- Strategy Guide (decision matrix, use cases)
- Provider Setup Guide (7 providers)
- Verification Protocol Guide
- CLI Reference
- MCP Server Guide
- 7 Python examples

### Testing
- 377 tests total (260 unit + 117 integration)
- pytest + pytest-asyncio
- Mock providers for isolated testing
- Coverage target: 80%+

### Known Issues
- SessionManager uses synchronous SQLite (may block in high-concurrency scenarios)
- Token counting uses estimation for most providers (except OpenAI/Gemini)
- Large context tests (>100K tokens) may cause memory issues on systems with <32GB RAM

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2026-01-20 | Initial release with full feature set |

---

## Upgrade Guide

### From Pre-Release to 0.1.0

If you were using a pre-release version:

1. Update your installation:
   ```bash
   pip install --upgrade contextflow
   ```

2. Update imports (if changed):
   ```python
   # Old
   from contextflow import ContextFlow

   # New (same)
   from contextflow.core import ContextFlow
   ```

3. Check configuration:
   - Environment variables remain the same
   - API endpoints remain the same

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this changelog.

When adding entries:
- Use present tense ("Add feature" not "Added feature")
- Include issue/PR references where applicable
- Group changes by type (Added, Changed, Deprecated, Removed, Fixed, Security)
