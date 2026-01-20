# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**ContextFlow AI** is a world-class Open-Source Framework for intelligent LLM context orchestration.

- **Repository:** `D:\SAI_ULTRA\contextflow\`
- **Language:** Python 3.11+
- **Architecture:** Async-first with Pydantic 2.5
- **Package Manager:** Poetry

### Core Features
- **Auto-Strategy Selection:** GSD (<10K), RALPH (10K-100K), RLM (>100K tokens)
- **Class Names:** `GSDStrategy`, `RALPHStrategy` (uppercase), `RLMStrategy`
- **7 LLM Providers:** Claude, OpenAI, Ollama, vLLM, Groq, Gemini, Mistral
- **In-Memory FAISS RAG:** 23x faster than external vector DBs
- **Verification Loop:** All strategies include self-verification (Boris Step 13)
- **3-Layer Progressive Disclosure:** 10x token savings (Claude-Mem pattern)
- **Session Manager:** SQLite-backed observation history

---

## Current Project Status

```
PHASE 1: Foundation        [##########] 100% COMPLETE (12/12)
PHASE 2: Providers         [##########] 100% COMPLETE (6/6)
PHASE 3: Strategies        [##########] 100% COMPLETE (6/6)
PHASE 4: RAG & Agents      [##########] 100% COMPLETE (11/11)
PHASE 5: Orchestrator      [##########] 100% COMPLETE (4/4)
PHASE 6: API & CLI         [##########] 100% COMPLETE (5/5)
PHASE 7: Testing & Polish  [########  ]  85% IN PROGRESS

TOTAL: ~97% (47/48 Tasks) | 27,265+ LOC | 59+ Python Files
       + ~166K LOC Tests | ~75K LOC Docs | ~88K LOC Examples
```

### Phase 7 Status (Current)

| Task | Component | Status | Notes |
|------|-----------|--------|-------|
| 7.1 | Unit Tests | IMPLEMENTED | 6 files, ~260 tests |
| 7.2 | Integration Tests | IMPLEMENTED | 5 files, ~115 tests |
| 7.3 | Documentation | IMPLEMENTED | 8 files in docs/ |
| 7.4 | Examples | IMPLEMENTED | 8 files in examples/ |
| 7.5 | CI/CD Pipeline | IMPLEMENTED | 3 workflows |
| 7.6 | Dev Hooks Setup | IMPLEMENTED | 5 files in .claude/hooks/ |

**Remaining:** Test verification, Coverage report, Documentation sync, Checkpoint 008

---

## Commands

```bash
# Install dependencies
poetry install --with dev,test
# or: make dev

# Run all tests (WARNING: may cause memory issues - run individually)
make test

# Run single test file (RECOMMENDED)
poetry run pytest tests/unit/test_analyzer.py -v

# Run specific test
poetry run pytest tests/unit/test_analyzer.py::TestClassName::test_method -v

# Run tests with coverage
make test-cov

# Lint and format
make lint          # Check only
make format        # Auto-fix

# Type checking
make typecheck

# Quick check (format + lint + typecheck)
make check

# Start API server
make serve
# or: uvicorn contextflow.api.server:app --reload --port 8000

# CLI usage
poetry run contextflow process "Summarize this" document.txt
poetry run contextflow analyze document.txt
poetry run contextflow serve --port 8000
poetry run contextflow info
poetry run contextflow providers
poetry run contextflow strategies

# Quick verification
python -c "from contextflow.providers import list_providers; print(list_providers())"

# Build docs
poetry run mkdocs serve      # Local preview
poetry run mkdocs build      # Build static
```

---

## Architecture

```
src/contextflow/
├── core/
│   ├── orchestrator.py   # Main entry point - ContextFlow class
│   ├── analyzer.py       # Context analysis (size, complexity, density)
│   ├── router.py         # Strategy selection logic (decision matrix)
│   ├── hooks.py          # Lifecycle hooks (pre/post process, on_error)
│   ├── session.py        # SQLite-backed session tracking (Claude-Mem)
│   ├── config.py         # Pydantic-based configuration
│   └── types.py          # Shared type definitions
├── providers/
│   ├── base.py           # BaseProvider ABC - all providers inherit this
│   ├── factory.py        # get_provider() factory function
│   ├── claude.py         # Anthropic Claude
│   ├── openai_provider.py # OpenAI GPT models
│   ├── ollama.py         # Local Ollama
│   ├── vllm.py           # vLLM server
│   ├── groq.py           # Groq fast inference
│   ├── gemini.py         # Google Gemini (1M context!)
│   └── mistral.py        # Mistral AI
├── strategies/
│   ├── base.py           # BaseStrategy ABC with execute() + verify()
│   ├── verification.py   # VerificationProtocol (Boris Step 13)
│   ├── gsd.py            # GSD strategy (<10K tokens)
│   ├── ralph.py          # RALPH strategy (10K-100K tokens)
│   └── rlm.py            # RLM recursive strategy (>100K tokens)
├── rag/
│   ├── temp_rag.py       # FAISS-based TemporaryRAG with 3-Layer Search
│   ├── chunker.py        # Smart document chunking
│   └── embeddings/       # OpenAI, SentenceTransformers, Ollama embedders
├── agents/
│   ├── pool.py           # AgentPool for parallel task execution
│   ├── sub_agent.py      # SubAgent with verification
│   ├── aggregator.py     # Result aggregation for RLM
│   └── repl.py           # REPL environment for code execution
├── api/
│   ├── server.py         # FastAPI REST endpoints (10+)
│   ├── websocket.py      # WebSocket + SSE streaming
│   └── models.py         # Pydantic request/response models (26 models)
├── cli/
│   └── main.py           # Typer CLI (7 commands)
├── mcp/
│   └── server.py         # MCP server for Claude Code integration
└── utils/
    ├── errors.py         # Custom exception hierarchy
    ├── logging.py        # Structlog setup
    └── tokens.py         # Token estimation and cost calculation
```

### Data Flow

1. **Input** → `ContextFlow.process(task, documents)`
2. **Hooks** → PRE_PROCESS lifecycle hooks
3. **Analysis** → `ContextAnalyzer` determines token count, complexity, density
4. **Routing** → `StrategyRouter` selects GSD/RALPH/RLM based on decision matrix
5. **Execution** → Selected strategy processes with provider
6. **Verification** → `VerificationProtocol` validates output (iterates if needed)
7. **Hooks** → POST_PROCESS lifecycle hooks (ON_ERROR if failed)
8. **Session** → Save observation to SessionManager
9. **Output** → `ProcessResult` with answer, strategy_used, tokens, cost

### Strategy Decision Matrix

```
Token Count | Complexity | Density | Strategy
<10K        | Low        | *       | GSD_DIRECT
<10K        | High       | *       | GSD_GUIDED
10K-50K     | *          | <0.5    | RALPH_ITERATIVE
10K-50K     | *          | >=0.5   | RALPH_STRUCTURED
50K-100K    | *          | <0.7    | RALPH_STRUCTURED
50K-100K    | *          | >=0.7   | RLM_BASIC
>100K       | *          | *       | RLM_FULL
```

---

## Coding Standards

### Type Hints (REQUIRED)
```python
# GOOD
async def process(self, task: str, documents: List[str]) -> ProcessResult:
    ...

# BAD - Missing type hints
async def process(self, task, documents):
    ...
```

### Docstrings (REQUIRED for public methods)
```python
async def complete(
    self,
    messages: List[Message],
    system: Optional[str] = None,
) -> CompletionResponse:
    """
    Send completion request to provider.

    Args:
        messages: List of conversation messages
        system: Optional system prompt

    Returns:
        CompletionResponse with text and token usage

    Raises:
        ProviderError: If API call fails
        RateLimitError: If rate limit exceeded
    """
```

### Error Handling
```python
# Use custom exceptions from utils/errors.py
from contextflow.utils.errors import ProviderError, RateLimitError, TokenLimitError

# GOOD
try:
    response = await self._client.chat(messages)
except httpx.HTTPStatusError as e:
    raise ProviderError(f"API error: {e.response.status_code}") from e

# BAD - Generic exceptions
except Exception as e:
    print(f"Error: {e}")  # Never use print!
```

### Logging
```python
# Use ProviderLogger from utils/logging.py
from contextflow.utils.logging import ProviderLogger

logger = ProviderLogger("claude")
logger.info("Starting completion", model=model, tokens=token_count)
logger.error("API error", error=str(e), status_code=status)
```

---

## Patterns

### Provider Implementation
All providers MUST implement `BaseProvider` ABC:
```python
from contextflow.providers.base import BaseProvider

class NewProvider(BaseProvider):
    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> ProviderCapabilities: ...

    async def complete(self, messages, system, ...) -> CompletionResponse: ...

    async def stream(self, messages, ...) -> AsyncIterator[StreamChunk]: ...

    def count_tokens(self, text: str) -> int: ...
```

### Strategy Implementation
```python
from contextflow.strategies.base import BaseStrategy

class NewStrategy(BaseStrategy):
    async def execute(self, task: str, context: str) -> StrategyResult: ...
    async def verify(self, task: str, output: str) -> VerificationResult: ...
```

### Retry Logic
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
async def _call_api(self, ...):
    ...
```

### Token Counting
```python
# For OpenAI-compatible: Use tiktoken (exact)
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4")
tokens = len(encoding.encode(text))

# For others: Character estimation (~4 chars/token)
tokens = len(text) // 4
```

---

## Verification Protocol (CRITICAL - Boris Step 13)

Every strategy MUST implement verification:

```python
class BaseStrategy(ABC):
    @abstractmethod
    async def execute(self, task: str, context: str) -> StrategyResult:
        """Execute strategy and return result."""
        pass

    @abstractmethod
    async def verify(self, task: str, output: str) -> VerificationResult:
        """Verify output meets task requirements."""
        pass
```

**Why:** "Give Claude a way to verify its work - 2-3x quality improvement"

---

## DO NOT

### Never use print()
```python
# BAD
print(f"Processing {len(documents)} documents")

# GOOD
logger.info("Processing documents", count=len(documents))
```

### Never use blocking calls in async functions
```python
# BAD
async def process(self):
    time.sleep(1)  # Blocks event loop!
    requests.get(url)  # Blocking!

# GOOD
async def process(self):
    await asyncio.sleep(1)
    async with httpx.AsyncClient() as client:
        await client.get(url)
```

### Never hardcode API keys
```python
# BAD
client = Anthropic(api_key="sk-ant-...")

# GOOD
from contextflow.core.config import ContextFlowConfig
config = ContextFlowConfig.from_env()
client = Anthropic(api_key=config.claude.api_key)
```

### Never commit secrets
Files to never commit:
- `.env` (use `.env.example` as template)
- `credentials.json`
- Any file containing API keys

---

## Testing

Tests use pytest-asyncio with mock providers. **Memory-intensive** - run subsets when needed:

```bash
# Unit tests only (RECOMMENDED)
poetry run pytest tests/unit/ -v

# Single test file (for debugging)
poetry run pytest tests/unit/test_verification.py -v -s

# Integration tests only
poetry run pytest tests/integration/ -v

# With output capture disabled
poetry run pytest tests/unit/test_analyzer.py -v -s --capture=no

# Coverage report
poetry run pytest --cov=contextflow --cov-report=html
```

### Test Files Overview

**Unit Tests (~260 tests):**
- `test_analyzer.py` - Context Analyzer tests
- `test_hooks.py` - Lifecycle Hooks tests
- `test_providers.py` - Provider mock tests
- `test_router.py` - Strategy Router tests
- `test_session.py` - Session Manager tests
- `test_verification.py` - Verification Protocol tests

**Integration Tests (~115 tests):**
- `test_api_integration.py` - REST API tests
- `test_orchestrator_pipeline.py` - Full pipeline tests
- `test_rag_integration.py` - RAG system tests
- `test_strategy_routing.py` - Strategy selection tests
- `test_verification_integration.py` - End-to-end verification

**Coverage target:** 80%+

---

## Subagent Usage Pattern

This project uses subagents extensively. Pattern for Phase implementation:

### Execution Pattern
```
Wave 1: Parallel execution of independent tasks
        → Each task gets Execution Agent + Validation Agent

Wave 2: Sequential tasks depending on Wave 1
        → Execution + Validation per task

Wave 3: Integration/verification tasks
        → Final validation agents
```

### Subagent Statistics (Project Total)
| Phase | Execution Agents | Validation Agents | Total |
|-------|------------------|-------------------|-------|
| Phase 3 | 6 | 6 | 12 |
| Phase 4 | 11 | 11 | 22 |
| Phase 5 | 4 | 4 | 8 |
| Phase 6 | 5 | 5 | 10 |
| **Total** | **26** | **26** | **52** |

### When to Use Subagents
- **Parallel tasks:** Use multiple subagents simultaneously
- **Validation:** Always pair execution with validation agent
- **Complex files:** Dedicated agent per large component
- **Troubleshooting:** Isolate failing component in fresh context

---

## Troubleshooting

### Exit Code 137 (SIGKILL / Out of Memory)
**Problem:** Tests cause memory issues with full test suite
**Solution:** Run test files individually, not full suite:
```bash
# Instead of: poetry run pytest tests/
# Do: poetry run pytest tests/unit/test_analyzer.py -v
```

### Pydantic v2 conflicts with Anaconda
**Solution:** `pip install --ignore-installed pydantic`

### tiktoken not found
**Solution:** `pip install tiktoken` separately

### Windows Unicode issues in terminal
**Solution:** Use ASCII characters in verification scripts

### FAISS AVX2/AVX512 warnings
**Note:** AVX512 not available but AVX2 works fine - ignore warning

### Import errors after changes
**Solution:** Verify imports:
```python
python -c "from contextflow.providers import list_providers; print(list_providers())"
python -c "from contextflow.core import ContextFlow; print('OK')"
```

---

## Documentation References

### Project Documentation (External)
- **MASTERPLAN:** `D:\SAI_ULTRA\Researches\Claude MD\ContextFlow\MASTERPLAN.md`
- **PROJECT_STATUS:** `D:\SAI_ULTRA\Researches\Claude MD\ContextFlow\PROJECT_STATUS.md`
- **DECISIONS:** `D:\SAI_ULTRA\Researches\Claude MD\ContextFlow\DECISIONS.md`
- **Checkpoints:** `D:\SAI_ULTRA\Researches\Claude MD\ContextFlow\CHECKPOINTS\`
- **Final Plans:** `D:\SAI_ULTRA\Researches\Claude MD\Final\`

### In-Repository Documentation
- **API Docs:** `docs/api-reference.md`
- **Strategy Docs:** `docs/strategies.md`
- **Provider Docs:** `docs/providers.md`
- **Verification Docs:** `docs/verification.md`
- **CLI Docs:** `docs/cli.md`
- **MCP Docs:** `docs/mcp-server.md`
- **Examples:** `examples/` (8 examples)

---

## Environment Variables

```bash
# Required for providers
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=...

# Optional configuration
CONTEXTFLOW_DEFAULT_PROVIDER=claude
CONTEXTFLOW_GSD_MAX_TOKENS=10000
CONTEXTFLOW_RALPH_MAX_TOKENS=100000
CONTEXTFLOW_RLM_MAX_PARALLEL_AGENTS=10
CONTEXTFLOW_RLM_CHUNK_SIZE=50000
CONTEXTFLOW_RLM_CHUNK_OVERLAP=2500
```

---

## Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Three-Strategy Architecture | GSD/RALPH/RLM | Different sizes need different approaches |
| In-Memory FAISS RAG | FAISS IndexFlatIP | 23x faster, no infra dependencies |
| Async-First | asyncio throughout | Enables parallel sub-agents |
| Provider ABC Pattern | Abstract Base Class | Unified interface for 7+ providers |
| Structured Exceptions | Deep hierarchy | Precise error handling |
| Structlog Logging | JSON + Console | Production-grade structured logs |
| Poetry | pyproject.toml | Modern Python packaging |

---

## CI/CD Workflows

Located in `.github/workflows/`:

- **ci.yml:** Lint, type-check, test, security, build
- **docs.yml:** Documentation deployment
- **release.yml:** PyPI release automation

---

## Dev Hooks (Boris Step 9)

Located in `.claude/hooks/`:

- **index.js:** Hook registry
- **post-edit.js:** Auto-format on edit
- **post-write.js:** Package structure validation
- **pre-commit.js:** Tests and validation before commit

Configuration in `.claude/settings.json`

---

## Phase 7 Completion Checklist

### Remaining Tasks
- [ ] Run all unit tests individually and verify passing
- [ ] Run all integration tests individually and verify passing
- [ ] Generate coverage report (target: 80%+)
- [ ] Update PROJECT_STATUS.md to reflect actual ~85% Phase 7
- [ ] Create checkpoint_008.md
- [ ] Final linting pass (`make lint`)
- [ ] Final type-check (`make typecheck`)
- [ ] Build test (`poetry build`)

### Verification Commands
```bash
# Individual test verification
poetry run pytest tests/unit/test_analyzer.py -v
poetry run pytest tests/unit/test_hooks.py -v
poetry run pytest tests/unit/test_providers.py -v
poetry run pytest tests/unit/test_router.py -v
poetry run pytest tests/unit/test_session.py -v
poetry run pytest tests/unit/test_verification.py -v

# Integration tests (after unit tests pass)
poetry run pytest tests/integration/test_api_integration.py -v
poetry run pytest tests/integration/test_orchestrator_pipeline.py -v
poetry run pytest tests/integration/test_rag_integration.py -v
poetry run pytest tests/integration/test_strategy_routing.py -v
poetry run pytest tests/integration/test_verification_integration.py -v

# Quality checks
make lint
make typecheck
poetry build
```

---

*Last Updated: 2026-01-20*
*Project Version: 0.1.0*
*Phase: 7 (Testing & Polish) - 85% Complete*
